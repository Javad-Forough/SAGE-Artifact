from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from crypto import aead_decrypt, aead_encrypt, derive_scope_key, make_aad
from store import SQLiteMemoryStore, StoredRecord


@dataclass
class ServiceConfig:
    root_key_sealed_path: str = "sealed_root_key_tpm.bin"
    db_path: str = "sealed_mem_tpm.db"


class SealedAgentMemoryServiceTPM:
    """
    TPM-backed SAGE service.

    This mirrors the provenance-aware behavior of service.py, but uses:
      - TPM-backed file sealing for the root key
      - TPM-backed rollback-protected epochs
    """

    def __init__(self, cfg: ServiceConfig, epoch: Any, sealer: Any):
        self.cfg = cfg
        self.epoch = epoch
        self.sealer = sealer
        self.store = SQLiteMemoryStore(cfg.db_path)
        self.root_key = self._load_or_create_root_key()

    def _load_or_create_root_key(self) -> bytes:
        if hasattr(self.sealer, "exists"):
            if self.sealer.exists(self.cfg.root_key_sealed_path):
                return self.sealer.unseal_from_file(self.cfg.root_key_sealed_path)
        elif os.path.exists(self.cfg.root_key_sealed_path):
            return self.sealer.unseal_from_file(self.cfg.root_key_sealed_path)

        root_key = os.urandom(32)
        self.sealer.seal_to_file(root_key, self.cfg.root_key_sealed_path)
        return root_key

    @staticmethod
    def _norm_list(values: Optional[Iterable[str]]) -> tuple[str, ...]:
        if values is None:
            return ()
        return tuple(sorted({str(v) for v in values if str(v)}))

    @staticmethod
    def _encode_scope_dep(scope_id: str, epoch: int) -> str:
        return json.dumps(
            {"scope": str(scope_id), "epoch": int(epoch)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_scope_dep(token: str) -> Optional[tuple[str, int]]:
        try:
            obj = json.loads(token)
            scope_id = str(obj["scope"])
            epoch = int(obj["epoch"])
            return scope_id, epoch
        except Exception:
            return None

    def _dep_tokens_to_scope_ids(self, dep_tokens: Iterable[str]) -> tuple[str, ...]:
        scope_ids: set[str] = set()
        for token in dep_tokens:
            parsed = self._decode_scope_dep(token)
            if parsed is None:
                continue
            scope_ids.add(parsed[0])
        return tuple(sorted(scope_ids))

    def _current_dep_tokens_for_scopes(self, scope_ids: Iterable[str]) -> tuple[str, ...]:
        tokens = {
            self._encode_scope_dep(scope_id, self.epoch.get_epoch(scope_id))
            for scope_id in scope_ids
            if str(scope_id)
        }
        return tuple(sorted(tokens))

    def _effective_source_scope_deps(
        self,
        derived_from_item_ids: Iterable[str],
        explicit_source_scope_ids: Iterable[str],
    ) -> tuple[str, ...]:
        dep_tokens: set[str] = set(self._current_dep_tokens_for_scopes(explicit_source_scope_ids))
        if derived_from_item_ids:
            dep_tokens.update(self.store.get_dependency_tokens_for_items(derived_from_item_ids))
        return tuple(sorted(dep_tokens))

    def _dependency_invalidation_reason(self, dep_tokens: Iterable[str]) -> Optional[str]:
        for token in dep_tokens:
            parsed = self._decode_scope_dep(token)
            if parsed is None:
                return "malformed source-scope dependency metadata"
            scope_id, expected_epoch = parsed
            current_epoch = self.epoch.get_epoch(scope_id)
            if current_epoch != expected_epoch:
                return (
                    f"source scope forgotten or advanced: {scope_id} "
                    f"(expected epoch {expected_epoch}, current epoch {current_epoch})"
                )
        return None

    def put(
        self,
        scope_id: str,
        payload: Any,
        kind: str = "text",
        item_id: Optional[str] = None,
        derived_from_item_ids: Optional[Iterable[str]] = None,
        source_scope_ids: Optional[Iterable[str]] = None,
    ) -> str:
        if item_id is None:
            item_id = str(uuid.uuid4())

        parents = self._norm_list(derived_from_item_ids)
        explicit_source_scopes = self._norm_list(source_scope_ids)

        source_scope_deps = self._effective_source_scope_deps(
            derived_from_item_ids=parents,
            explicit_source_scope_ids=explicit_source_scopes,
        )
        source_scope_ids_plain = self._dep_tokens_to_scope_ids(source_scope_deps)

        epoch = self.epoch.get_epoch(scope_id)
        scope_key = derive_scope_key(self.root_key, scope_id, epoch)

        created_ts = int(time.time())
        plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        aad = make_aad(
            scope_id,
            epoch,
            item_id,
            kind,
            created_ts,
            derived_from_item_ids=parents,
            source_scope_ids=source_scope_deps,
        )

        nonce, ct = aead_encrypt(scope_key, plaintext, aad)
        self.store.put(
            StoredRecord(
                scope_id=scope_id,
                epoch=epoch,
                item_id=item_id,
                kind=kind,
                created_ts=created_ts,
                nonce=nonce,
                ct=ct,
                derived_from_item_ids=parents,
                source_scope_ids=source_scope_ids_plain,
                source_scope_deps=source_scope_deps,
            )
        )
        return item_id

    def put_derived(
        self,
        scope_id: str,
        payload: Any,
        kind: str,
        derived_from_item_ids: Iterable[str],
        source_scope_ids: Optional[Iterable[str]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        return self.put(
            scope_id=scope_id,
            payload=payload,
            kind=kind,
            item_id=item_id,
            derived_from_item_ids=derived_from_item_ids,
            source_scope_ids=source_scope_ids,
        )

    def get_recent(
        self,
        scope_id: str,
        limit: int = 20,
        kinds: Optional[Iterable[str]] = None,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        epoch = self.epoch.get_epoch(scope_id)
        scope_key = derive_scope_key(self.root_key, scope_id, epoch)

        out: list[dict[str, Any]] = []
        for rec in self.store.get_by_scope_epoch(
            scope_id,
            epoch,
            limit=limit,
            active_only=False,
            kinds=kinds,
        ):
            aad = make_aad(
                rec.scope_id,
                rec.epoch,
                rec.item_id,
                rec.kind,
                rec.created_ts,
                derived_from_item_ids=rec.derived_from_item_ids,
                source_scope_ids=rec.source_scope_deps,
            )

            try:
                pt = aead_decrypt(scope_key, rec.nonce, rec.ct, aad)
            except Exception:
                continue

            dynamic_reason = self._dependency_invalidation_reason(rec.source_scope_deps)
            effective_active = bool(rec.is_active) and dynamic_reason is None

            if not include_inactive and not effective_active:
                continue

            invalidation_reason = dynamic_reason if dynamic_reason is not None else rec.invalidation_reason

            source_scope_dependencies: list[dict[str, Any]] = []
            for token in rec.source_scope_deps:
                parsed = self._decode_scope_dep(token)
                if parsed is not None:
                    source_scope_dependencies.append(
                        {"scope_id": parsed[0], "epoch": parsed[1]}
                    )

            out.append(
                {
                    "scope_id": rec.scope_id,
                    "epoch": rec.epoch,
                    "item_id": rec.item_id,
                    "kind": rec.kind,
                    "created_ts": rec.created_ts,
                    "payload": json.loads(pt.decode("utf-8")),
                    "derived_from_item_ids": list(rec.derived_from_item_ids),
                    "source_scope_ids": list(rec.source_scope_ids),
                    "source_scope_dependencies": source_scope_dependencies,
                    "is_active": effective_active,
                    "stored_is_active": rec.is_active,
                    "invalidated_ts": rec.invalidated_ts,
                    "invalidation_reason": invalidation_reason,
                }
            )
        return out

    def _collect_transitively_dependent_items(self, forgotten_scope_id: str) -> set[str]:
        _seed_items, dependent_items = self.store.collect_all_transitive_dependents(
            forgotten_scope_id
        )
        return dependent_items

    def forget_scope(
        self,
        scope_id: str,
        delete_ciphertext_rows: bool = False,
        propagate: bool = True,
    ) -> dict[str, Any]:
        old_epoch = self.epoch.get_epoch(scope_id)
        new_epoch = self.epoch.bump_epoch(scope_id)

        invalidated_items = 0
        deleted_dependent_rows = 0
        invalidated_item_ids: list[str] = []
        if propagate:
            dependent_ids = sorted(self._collect_transitively_dependent_items(scope_id))
            if dependent_ids:
                if delete_ciphertext_rows:
                    deleted_dependent_rows = self.store.delete_items_by_ids(dependent_ids)
                    invalidated_items = deleted_dependent_rows
                else:
                    invalidated_items = self.store.invalidate_items(
                        dependent_ids,
                        reason=f"source scope forgotten: {scope_id}",
                        invalidated_ts=int(time.time()),
                    )
                invalidated_item_ids = dependent_ids

        deleted_rows = 0
        if delete_ciphertext_rows:
            deleted_rows = self.store.delete_scope_all_epochs(scope_id)

        return {
            "scope_id": scope_id,
            "old_epoch": old_epoch,
            "new_epoch": new_epoch,
            "deleted_ciphertext_rows": deleted_rows,
            "deleted_dependent_rows": deleted_dependent_rows,
            "propagation_enabled": propagate,
            "invalidated_dependent_items": invalidated_items,
            "invalidated_item_ids": invalidated_item_ids,
        }

    def count_scope_rows(self, scope_id: str) -> int:
        return self.store.count_scope(scope_id)
