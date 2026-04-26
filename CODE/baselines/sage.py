from __future__ import annotations

from epoch import SQLiteEpochStore
from sealing import DevSealer
from service import SealedAgentMemoryService, ServiceConfig


class SAGE:
    """
    SAGE scheme wrapper so experiments treat it like a baseline scheme.

    Epoch store: SQLiteEpochStore (separate DB, WAL + FULL sync).
    Compared to the old LocalEpochStore (JSON file):
      - Atomic per-bump durability via PRAGMA synchronous=FULL
      - Epoch file is independent from the memory DB, so rolling back the
        memory DB alone does not revert epochs

    For full hardware-backed rollback protection use baselines/sage_tpm.py.
    """

    def __init__(
        self,
        db_path: str,
        root_key_sealed: str = "sealed_root_key.sage.bin",
        epochs_path: str = "epochs_eval.db",
        dev_master_path: str = "dev_master_key_eval.bin",
    ):
        # SQLiteEpochStore: separate DB, stronger than JSON file
        self.epoch = SQLiteEpochStore(db_path=epochs_path)
        self.sealer = DevSealer(dev_master_path)

        self.mem = SealedAgentMemoryService(
            ServiceConfig(
                root_key_sealed_path=root_key_sealed,
                db_path=db_path,
            ),
            epoch=self.epoch,
            sealer=self.sealer,
        )

    def put(self, scope_id: str, payload: dict):
        return self.mem.put(scope_id, payload, kind="fact")

    def get_recent(self, scope_id: str, limit: int = 10):
        return self.mem.get_recent(scope_id, limit=limit)

    def forget_scope(self, scope_id: str):
        return self.mem.forget_scope(scope_id, delete_ciphertext_rows=True)

    def close(self):
        if hasattr(self.epoch, "close"):
            try:
                self.epoch.close()
            except Exception:
                pass
        if hasattr(self.mem.store, "close"):
            try:
                self.mem.store.close()
            except Exception:
                pass
