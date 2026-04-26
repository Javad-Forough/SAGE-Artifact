from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
from dataclasses import dataclass
from typing import Dict

from tpm_sealer import TPMSealer


@dataclass
class TPMEpochState:
    version: int
    epochs: Dict[str, int]
    mac_b64: str


class TPMEpochStore:
    """
    TPM-backed epoch store.

    - A MAC key is sealed with the TPM and cached in the process.
    - Per-scope epochs are stored in an authenticated JSON file.
    - A TPM NV counter stores the trusted version of that file.
    - Rolling back the JSON file without also rolling back the TPM counter is detected.
    """

    def __init__(
        self,
        state_path: str,
        blob_prefix: str = "results/tpm",
        sealer: TPMSealer | None = None,
    ):
        self.path = state_path
        self.pending_path = f"{state_path}.pending"
        self.meta_path = f"{state_path}.nvmeta.json"
        self.sealer = sealer or TPMSealer(blob_prefix=blob_prefix)
        self.mac_key_path = f"{state_path}.mackey"
        self._lock = threading.Lock()
        self._mac_key_cache: bytes | None = None
        self._state_cache: TPMEpochState | None = None
        self._counter_index_cache: int | None = None

        had_meta = os.path.exists(self.meta_path)
        self._ensure_mac_key()
        created_counter = self._ensure_counter_ref()
        if not had_meta and os.path.exists(self.path):
            # Existing state file predates NV metadata.
            self._migrate_existing_state_if_needed()
        else:
            self._ensure_state(allow_initial_create=created_counter)

    # ---------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------

    def _sealed_blob_exists(self, path: str) -> bool:
        if hasattr(self.sealer, "exists"):
            return bool(self.sealer.exists(path))
        return os.path.exists(path)

    def _ensure_mac_key(self) -> None:
        if self._sealed_blob_exists(self.mac_key_path):
            return

        key = os.urandom(32)
        self.sealer.seal_to_file(key, self.mac_key_path)

    def _load_mac_key(self) -> bytes:
        if self._mac_key_cache is None:
            self._mac_key_cache = self.sealer.unseal_from_file(self.mac_key_path)
        return self._mac_key_cache

    def _load_counter_index(self) -> int:
        if self._counter_index_cache is not None:
            return self._counter_index_cache

        with open(self.meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        index_raw = obj.get("nv_index")
        if isinstance(index_raw, str):
            index = int(index_raw, 16 if index_raw.startswith("0x") else 10)
        else:
            index = int(index_raw)

        self._counter_index_cache = index
        return index

    def _save_counter_index(self, index: int) -> None:
        parent = os.path.dirname(self.meta_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"nv_index": hex(index)}, f, sort_keys=True)
        os.replace(tmp, self.meta_path)
        self._counter_index_cache = index

    def _ensure_counter_ref(self) -> bool:
        if os.path.exists(self.meta_path):
            self.sealer.read_nv_counter(self._load_counter_index())
            return False

        index = self.sealer.find_free_nv_index()
        self.sealer.define_nv_counter(index)
        try:
            # On the current swtpm+tpm2-tools setup, a freshly defined NV
            # counter is not readable until it has been incremented once.
            # We therefore establish the initial trusted baseline at version 1.
            initial_counter = self.sealer.increment_nv_counter(index)
            if initial_counter < 1:
                raise RuntimeError(
                    "Fresh TPM NV counter did not advance to a readable baseline"
                )
            self._save_counter_index(index)
        except Exception:
            try:
                self.sealer.undefine_nv_index(index)
            except Exception:
                pass
            raise
        return True

    def _current_counter(self) -> int:
        return self.sealer.read_nv_counter(self._load_counter_index())

    # ---------------------------------------------------------
    # MAC helpers
    # ---------------------------------------------------------

    def _compute_mac(self, version: int, epochs: Dict[str, int]) -> bytes:
        key = self._load_mac_key()

        msg = json.dumps(
            {"version": version, "epochs": epochs},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hmac.new(key, msg, hashlib.sha256).digest()

    # ---------------------------------------------------------
    # State management
    # ---------------------------------------------------------

    def _build_state(self, version: int, epochs: Dict[str, int]) -> TPMEpochState:
        mac = self._compute_mac(version, epochs)
        return TPMEpochState(
            version=version,
            epochs=dict(epochs),
            mac_b64=base64.b64encode(mac).decode("ascii"),
        )

    def _write_state(self, path: str, st: TPMEpochState) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp = path + ".tmp"

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": st.version,
                    "epochs": st.epochs,
                    "mac": st.mac_b64,
                },
                f,
                sort_keys=True,
            )

        os.replace(tmp, path)

    def _read_state_file(self, path: str) -> TPMEpochState:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise RuntimeError("TPM epoch state is not a JSON object")

        if "version" not in obj or "epochs" not in obj or "mac" not in obj:
            raise RuntimeError("TPM epoch state format invalid")

        version = int(obj["version"])
        epochs = {str(k): int(v) for k, v in obj["epochs"].items()}
        mac_b64 = str(obj["mac"])

        expected = self._compute_mac(version, epochs)
        got = base64.b64decode(mac_b64.encode("ascii"))

        if not hmac.compare_digest(expected, got):
            raise RuntimeError("TPM epoch state MAC verification failed")

        return TPMEpochState(version=version, epochs=epochs, mac_b64=mac_b64)

    def _remove_pending(self) -> None:
        try:
            os.remove(self.pending_path)
        except FileNotFoundError:
            pass

    def _load_state_uncached(self) -> TPMEpochState:
        counter = self._current_counter()

        state: TPMEpochState | None = None
        if os.path.exists(self.path):
            state = self._read_state_file(self.path)

        pending: TPMEpochState | None = None
        if os.path.exists(self.pending_path):
            pending = self._read_state_file(self.pending_path)

        if state is not None and state.version == counter:
            if pending is not None and pending.version <= counter:
                self._remove_pending()
            return state

        if pending is not None and pending.version == counter:
            self._write_state(self.path, pending)
            self._remove_pending()
            return pending

        if state is None:
            raise RuntimeError(
                f"TPM epoch state missing or stale: counter={counter}, state file absent"
            )

        pending_desc = pending.version if pending is not None else "none"
        raise RuntimeError(
            "TPM epoch state rollback detected: "
            f"file version={state.version}, pending version={pending_desc}, counter={counter}"
        )

    def _ensure_state(self, *, allow_initial_create: bool = False) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if not os.path.exists(self.path):
            counter = self._current_counter()
            if not allow_initial_create:
                raise RuntimeError(
                    "TPM counter exists but epoch state file is missing. "
                    "Refusing to recreate trusted state automatically."
                )
            fresh = self._build_state(version=counter, epochs={})
            self._write_state(self.path, fresh)
            self._state_cache = fresh
            return

        self._state_cache = self._load_state_uncached()

    def _migrate_existing_state_if_needed(self) -> None:
        existing = self._read_state_file(self.path)
        migrated = self._build_state(version=self._current_counter(), epochs=existing.epochs)
        self._write_state(self.path, migrated)
        self._remove_pending()
        self._state_cache = migrated

    def _load_state(self) -> TPMEpochState:
        counter = self._current_counter()
        if self._state_cache is not None and self._state_cache.version == counter:
            return self._state_cache

        st = self._load_state_uncached()
        self._state_cache = st
        return st

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def get_epoch(self, scope_id: str) -> int:
        with self._lock:
            st = self._load_state()
            return int(st.epochs.get(scope_id, 0))

    def advance_epoch(self, scope_id: str) -> int:
        """
        Advance the epoch for a scope and persist the updated authenticated map.

        Crash recovery:
          1. Write the next state to .pending
          2. Increment the TPM NV counter
          3. Commit the new state file

        If the process dies after step 2, reopening will replay the .pending state.
        """
        with self._lock:
            st = self._load_state()
            current_counter = self._current_counter()
            if st.version != current_counter:
                self._state_cache = None
                st = self._load_state()
                current_counter = st.version

            new_epochs = dict(st.epochs)
            new_epochs[scope_id] = int(new_epochs.get(scope_id, 0)) + 1
            next_version = current_counter + 1
            new_state = self._build_state(version=next_version, epochs=new_epochs)

            self._write_state(self.pending_path, new_state)
            new_counter = self.sealer.increment_nv_counter(self._load_counter_index())
            if new_counter != next_version:
                self._state_cache = None
                raise RuntimeError(
                    "Unexpected TPM counter value after increment: "
                    f"expected {next_version}, got {new_counter}"
                )

            self._write_state(self.path, new_state)
            self._remove_pending()
            self._state_cache = new_state
            return new_epochs[scope_id]

    # ---------------------------------------------------------
    # Compatibility layer
    # ---------------------------------------------------------

    def bump_epoch(self, scope_id: str) -> int:
        """
        Compatibility alias expected by service_tpm.py.
        Internally delegates to advance_epoch().
        """
        return self.advance_epoch(scope_id)

    @classmethod
    def destroy_persistent_state(
        cls,
        state_path: str,
        *,
        blob_prefix: str = "results/tpm",
        sealer: TPMSealer | None = None,
    ) -> None:
        sealer = sealer or TPMSealer(blob_prefix=blob_prefix)
        meta_path = f"{state_path}.nvmeta.json"
        pending_path = f"{state_path}.pending"

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                index_raw = obj.get("nv_index")
                index = int(index_raw, 16 if str(index_raw).startswith("0x") else 10)
                sealer.undefine_nv_index(index)
            except Exception:
                pass

        for path in (meta_path, pending_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
