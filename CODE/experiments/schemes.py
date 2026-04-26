from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

# Baselines
from baselines.plain import PlainLogicalDelete
from baselines.static_key import StaticKeyEncryption
from baselines.sealed_no_rp import SealedNoRollbackProtection
from baselines.kms import TrustedKMSDesign
from baselines.sqlite_envelope import SQLiteEnvelopeEncryption

# SAGE wrapper
from baselines.sage import SAGE
from baselines.sage_tpm import SAGE_TPM
from tpm_epoch import TPMEpochStore
from tpm_sealer import TPMSealer


TPM_SAGE_SCHEMES = frozenset({"sage_tpm", "sage_tpm_naive"})
ROLLBACK_PROTECTED_STATE_SCHEMES = frozenset({"sage", "kms", *TPM_SAGE_SCHEMES})
ROLLBACK_RESTORABLE_AUX_STATE_SCHEMES = frozenset(
    {"static", "sealed_no_rp", "sqlite_envelope"}
)


class Scheme(Protocol):
    def put(self, scope_id: str, payload: dict[str, Any]) -> Any: ...
    def get_recent(self, scope_id: str, limit: int = 10) -> list[dict[str, Any]]: ...
    def forget_scope(self, scope_id: str, **kwargs: Any) -> Any: ...
    def close(self) -> None: ...


def _safe_close(obj: Any) -> None:
    if hasattr(obj, "close"):
        try:
            obj.close()
        except Exception:
            pass


@dataclass
class SchemeConfig:
    db_path: str
    scheme: str

    # SAGE
    sage_root_sealed: str = "sealed_root_key.sage.bin"
    sage_epochs: str = "epochs_eval.db"          # SQLiteEpochStore (separate DB)
    sage_dev_master: str = "dev_master_key_eval.bin"
    sage_tpm_epochs: str = "epochs_tpm.json"
    sage_tpm_blob_prefix: str = "results/tpm"

    # Baselines
    static_key_path: str = "static_key.bin"
    sealed_no_rp_root: str = "sealed_root_key.baseline.bin"
    sealed_no_rp_master: str = "dev_master_key_baseline.bin"   # was plaintext, now encrypted
    kms_state: str = "kms_state.json"
    sqlite_envelope_keys: str = "sqlite_envelope_keys.json"


def assign_artifact_paths(cfg: SchemeConfig, stem: str) -> SchemeConfig:
    cfg.static_key_path = f"{stem}.static_key.bin"
    cfg.sealed_no_rp_root = f"{stem}.sealed_no_rp_root.bin"
    cfg.sealed_no_rp_master = f"{stem}.sealed_no_rp_master.bin"
    cfg.kms_state = f"{stem}.kms_state.json"
    cfg.sqlite_envelope_keys = f"{stem}.sqlite_envelope_keys.json"
    cfg.sage_root_sealed = f"{stem}.sage_root.bin"
    cfg.sage_epochs = f"{stem}.sage_epochs.db"
    cfg.sage_dev_master = f"{stem}.sage_dev_master.bin"
    cfg.sage_tpm_epochs = f"{stem}.sage_tpm_epochs.json"
    cfg.sage_tpm_blob_prefix = f"{stem}.tpm"
    return cfg


def scheme_artifact_paths(
    cfg: SchemeConfig,
    scheme: str | None = None,
    *,
    include_wal: bool = False,
) -> list[str]:
    scheme_name = scheme or cfg.scheme
    paths: list[str] = [cfg.db_path]

    if scheme_name == "static":
        paths.append(cfg.static_key_path)
    elif scheme_name == "sealed_no_rp":
        paths.extend([cfg.sealed_no_rp_root, cfg.sealed_no_rp_master])
    elif scheme_name == "kms":
        paths.append(cfg.kms_state)
    elif scheme_name == "sqlite_envelope":
        paths.append(cfg.sqlite_envelope_keys)
    elif scheme_name == "sage":
        paths.extend([cfg.sage_root_sealed, cfg.sage_epochs, cfg.sage_dev_master])
    elif scheme_name in TPM_SAGE_SCHEMES:
        paths.extend(
            [
                cfg.sage_root_sealed,
                cfg.sage_tpm_epochs,
                f"{cfg.sage_tpm_epochs}.pending",
                f"{cfg.sage_tpm_epochs}.nvmeta.json",
                f"{cfg.sage_tpm_epochs}.mackey",
            ]
        )
        for sealed_path in (cfg.sage_root_sealed, f"{cfg.sage_tpm_epochs}.mackey"):
            artifacts = TPMSealer.artifact_paths_for(sealed_path)
            paths.extend(artifacts.values())

    if include_wal:
        wal_like: list[str] = []
        for path in paths:
            if path.endswith(".db"):
                wal_like.extend([f"{path}-wal", f"{path}-shm"])
        paths.extend(wal_like)

    seen: set[str] = set()
    deduped: list[str] = []
    for path in paths:
        if path and path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def rollback_restorable_aux_state_paths(
    cfg: SchemeConfig,
    scheme: str | None = None,
) -> list[str]:
    scheme_name = scheme or cfg.scheme
    if scheme_name == "static":
        return [cfg.static_key_path]
    if scheme_name == "sealed_no_rp":
        return [cfg.sealed_no_rp_root, cfg.sealed_no_rp_master]
    if scheme_name == "sqlite_envelope":
        return [cfg.sqlite_envelope_keys]
    return []


def scheme_has_rollback_protected_state(scheme: str) -> bool:
    return scheme in ROLLBACK_PROTECTED_STATE_SCHEMES


def destroy_scheme_persistent_state(cfg: SchemeConfig, scheme: str | None = None) -> None:
    scheme_name = scheme or cfg.scheme
    if scheme_name not in TPM_SAGE_SCHEMES:
        return
    TPMEpochStore.destroy_persistent_state(
        cfg.sage_tpm_epochs,
        blob_prefix=cfg.sage_tpm_blob_prefix,
    )


class SchemeHandle:
    """
    Wrap scheme objects and normalize minor API differences.
    Also exposes provenance-aware put_derived when available.
    """

    def __init__(self, impl: Any, scheme_name: str):
        self.impl = impl
        self.name = scheme_name

    def __getattr__(self, item: str):
        return getattr(self.impl, item)

    @property
    def kms(self):
        return getattr(self.impl, "kms")

    @kms.setter
    def kms(self, value):
        setattr(self.impl, "kms", value)

    def put(self, scope_id: str, payload: dict[str, Any], kind: str = "fact") -> Any:
        try:
            return self.impl.put(scope_id, payload, kind=kind)
        except TypeError:
            return self.impl.put(scope_id, payload)

    def put_derived(
        self,
        scope_id: str,
        payload: dict[str, Any],
        kind: str,
        derived_from_item_ids: Iterable[str],
        source_scope_ids: Iterable[str] | None = None,
    ) -> Any:
        # Preferred: scheme directly supports put_derived
        try:
            return self.impl.put_derived(
                scope_id,
                payload,
                kind=kind,
                derived_from_item_ids=list(derived_from_item_ids),
                source_scope_ids=list(source_scope_ids or []),
            )
        except Exception:
            pass

        # Common SAGE wrapper pattern: impl.mem exposes the richer API
        try:
            return self.impl.mem.put_derived(
                scope_id,
                payload,
                kind=kind,
                derived_from_item_ids=list(derived_from_item_ids),
                source_scope_ids=list(source_scope_ids or []),
            )
        except Exception:
            pass

        # For baselines without derived-item APIs, store the artifact as a
        # normal item.
        return self.put(scope_id, payload, kind=kind)

    def get_recent(self, scope_id: str, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        try:
            return self.impl.get_recent(scope_id, limit=limit, **kwargs)
        except TypeError:
            return self.impl.get_recent(scope_id, limit=limit)

    def forget_scope(self, scope_id: str, **kwargs: Any) -> Any:
        # Prefer physical deletion for SAGE if the wrapper supports it.
        if self.name in {"sage", *TPM_SAGE_SCHEMES} and "delete_ciphertext_rows" not in kwargs:
            kwargs = {**kwargs, "delete_ciphertext_rows": True}
        try:
            return self.impl.forget_scope(scope_id, **kwargs)
        except TypeError:
            return self.impl.forget_scope(scope_id)

    def close(self) -> None:
        _safe_close(self.impl)


def make_scheme(cfg: SchemeConfig) -> SchemeHandle:
    s = cfg.scheme

    if s == "plain":
        return SchemeHandle(PlainLogicalDelete(cfg.db_path), "plain")

    if s == "static":
        return SchemeHandle(
            StaticKeyEncryption(cfg.db_path, key_path=cfg.static_key_path),
            "static",
        )

    if s == "sealed_no_rp":
        return SchemeHandle(
            SealedNoRollbackProtection(
                cfg.db_path,
                sealed_root_path=cfg.sealed_no_rp_root,
                dev_master_path=cfg.sealed_no_rp_master,
            ),
            "sealed_no_rp",
        )

    if s == "kms":
        return SchemeHandle(
            TrustedKMSDesign(cfg.db_path, kms_state_path=cfg.kms_state),
            "kms",
        )

    if s == "sqlite_envelope":
        return SchemeHandle(
            SQLiteEnvelopeEncryption(
                cfg.db_path,
                key_store_path=cfg.sqlite_envelope_keys,
            ),
            "sqlite_envelope",
        )

    if s == "sage":
        return SchemeHandle(
            SAGE(
                db_path=cfg.db_path,
                root_key_sealed=cfg.sage_root_sealed,
                epochs_path=cfg.sage_epochs,
                dev_master_path=cfg.sage_dev_master,
            ),
            "sage",
        )

    if s in TPM_SAGE_SCHEMES:
        return SchemeHandle(
            SAGE_TPM(
                db_path=cfg.db_path,
                root_key_sealed=cfg.sage_root_sealed,
                epochs_path=cfg.sage_tpm_epochs,
                tpm_blob_prefix=cfg.sage_tpm_blob_prefix,
            ),
            s,
        )

    raise ValueError(f"Unknown scheme: {s}")
