# baselines/static_key.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List

from .common import SQLiteKV, aead_encrypt, aead_decrypt


class StaticKeyEncryption:
    """
    Baseline 2: encrypt-at-rest with one static key K_static.
    Even if you logically delete, rollback resurrects ciphertext AND the key is still valid,
    so decryption succeeds.
    """
    def __init__(self, db_path: str, key_path: str = "static_key.bin"):
        self.kv = SQLiteKV(db_path)
        self.key_path = key_path
        self.key = self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        if os.path.exists(self.key_path):
            return open(self.key_path, "rb").read()
        key = os.urandom(32)
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def put(self, scope_id: str, payload: Dict[str, Any]) -> None:
        pt = json.dumps(payload).encode("utf-8")
        aad = json.dumps({"scheme": "static", "scope": scope_id}, separators=(",", ":")).encode("utf-8")
        nonce, ct = aead_encrypt(self.key, pt, aad)
        self.kv.put_row(scope_id, nonce=nonce, blob=ct, aad={"scheme": "static"})

    def get_recent(self, scope_id: str, limit: int) -> List[Dict[str, Any]]:
        rows = self.kv.get_recent_rows(scope_id, limit)
        out: List[Dict[str, Any]] = []
        aad = json.dumps({"scheme": "static", "scope": scope_id}, separators=(",", ":")).encode("utf-8")
        for r in rows:
            try:
                pt = aead_decrypt(self.key, r.nonce, r.blob, aad)
                out.append(json.loads(pt.decode("utf-8")))
            except Exception:
                # decryption failed
                pass
        return out

    def forget_scope(self, scope_id: str) -> Dict[str, Any]:
        # logical deletion only (key is unchanged)
        n = self.kv.delete_scope_rows(scope_id)
        return {"scheme": "static", "scope_id": scope_id, "deleted_rows": n}

    def close(self) -> None:
        self.kv.close()