# baselines/sealed_no_rp.py
from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, List

from .common import SQLiteKV, aead_encrypt, aead_decrypt, hkdf_32

# Resolve the project root so we can import sealing even when this module
# is imported from within the baselines package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sealing import DevSealer, SealedBlob, save_sealed_blob, load_sealed_blob  # noqa: E402


class SealedNoRollbackProtection:
    """
    Baseline 3: sealed root key + epoch versioning, BUT epoch is stored in the
    same database as the ciphertext.

    Design intent of this baseline:
      - The root key IS encrypted at rest (same DevSealer used by SAGE).
      - The epoch counter, however, is stored INSIDE the same SQLite database
        that holds the ciphertext.  Under a DB rollback, the epoch reverts
        together with the ciphertext rows, so old keys become valid again and
        cryptographic deletion fails.

    This isolates the rollback-protection property: the only difference between
    sealed_no_rp and SAGE is that SAGE stores epochs in a separate, protected
    store (SQLiteEpochStore / TPMEpochStore) while sealed_no_rp co-locates them.
    """

    def __init__(
        self,
        db_path: str,
        sealed_root_path: str = "sealed_root_key.baseline.bin",
        dev_master_path: str = "dev_master_key_baseline.bin",
    ):
        self.kv = SQLiteKV(db_path)
        self.db_path = db_path
        self.sealed_root_path = sealed_root_path
        self._sealer = DevSealer(master_key_path=dev_master_path)
        self.root = self._load_or_create_root()
        self._epoch_cache: dict[str, int] = {}   # in-memory epoch cache (per-process)
        self._init_epoch_table()

    def _load_or_create_root(self) -> bytes:
        """
        Load or generate the 32-byte root key, storing it AES-GCM encrypted
        via DevSealer (identical mechanism to the SAGE root key).
        """
        if os.path.exists(self.sealed_root_path):
            blob = load_sealed_blob(self.sealed_root_path)
            return self._sealer.unseal(blob)
        root = os.urandom(32)
        blob = self._sealer.seal(root)
        save_sealed_blob(self.sealed_root_path, blob)
        return root

    def _init_epoch_table(self) -> None:
        self.kv.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS epochs (
              scope_id TEXT PRIMARY KEY,
              epoch    INTEGER NOT NULL
            );
            """
        )
        self.kv.conn.commit()

    def _get_epoch(self, scope_id: str) -> int:
        if scope_id in self._epoch_cache:
            return self._epoch_cache[scope_id]
        cur = self.kv.conn.execute(
            "SELECT epoch FROM epochs WHERE scope_id=?", (scope_id,)
        )
        row = cur.fetchone()
        if row is None:
            self.kv.conn.execute(
                "INSERT INTO epochs(scope_id, epoch) VALUES(?,?)", (scope_id, 0)
            )
            self.kv.conn.commit()
            self._epoch_cache[scope_id] = 0
            return 0
        val = int(row[0])
        self._epoch_cache[scope_id] = val
        return val

    def _set_epoch(self, scope_id: str, epoch: int) -> None:
        self._epoch_cache[scope_id] = epoch
        self.kv.conn.execute(
            "INSERT OR REPLACE INTO epochs(scope_id, epoch) VALUES(?,?)",
            (scope_id, epoch),
        )
        self.kv.conn.commit()

    def _scope_key(self, scope_id: str, epoch: int) -> bytes:
        info = f"{scope_id}|{epoch}".encode("utf-8")
        return hkdf_32(self.root, info=info)

    def put(self, scope_id: str, payload: Dict[str, Any]) -> None:
        epoch = self._get_epoch(scope_id)
        key = self._scope_key(scope_id, epoch)
        pt = json.dumps(payload).encode("utf-8")
        aad = json.dumps(
            {"scheme": "sealed_no_rp", "scope": scope_id, "epoch": epoch},
            separators=(",", ":"),
        ).encode("utf-8")
        nonce, ct = aead_encrypt(key, pt, aad)
        self.kv.put_row(
            scope_id,
            nonce=nonce,
            blob=ct,
            aad={"scheme": "sealed_no_rp", "epoch": epoch},
        )

    def get_recent(self, scope_id: str, limit: int) -> List[Dict[str, Any]]:
        epoch = self._get_epoch(scope_id)
        key = self._scope_key(scope_id, epoch)
        aad = json.dumps(
            {"scheme": "sealed_no_rp", "scope": scope_id, "epoch": epoch},
            separators=(",", ":"),
        ).encode("utf-8")

        rows = self.kv.get_recent_rows(scope_id, limit)
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                pt = aead_decrypt(key, r.nonce, r.blob, aad)
                out.append(json.loads(pt.decode("utf-8")))
            except Exception:
                pass
        return out

    def forget_scope(self, scope_id: str) -> Dict[str, Any]:
        old = self._get_epoch(scope_id)
        new = old + 1
        self._set_epoch(scope_id, new)
        n = self.kv.delete_scope_rows(scope_id)
        return {
            "scheme": "sealed_no_rp",
            "scope_id": scope_id,
            "old_epoch": old,
            "new_epoch": new,
            "deleted_rows": n,
        }

    def close(self) -> None:
        self.kv.close()
