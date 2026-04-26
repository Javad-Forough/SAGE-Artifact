# baselines/common.py
from __future__ import annotations

import os
import json
import time
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Tuple

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def now_ts() -> int:
    return int(time.time())


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def aead_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> Tuple[bytes, bytes]:
    """
    ChaCha20-Poly1305 returns (nonce, ciphertext).
    """
    aead = ChaCha20Poly1305(key)
    nonce = os.urandom(12)
    ct = aead.encrypt(nonce, plaintext, aad)
    return nonce, ct


def aead_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    aead = ChaCha20Poly1305(key)
    return aead.decrypt(nonce, ciphertext, aad)


def hkdf_32(root_key: bytes, info: bytes, salt: Optional[bytes] = None) -> bytes:
    """
    Derive a 32-byte key deterministically from root_key.
    """
    hk = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=info,
    )
    return hk.derive(root_key)


@dataclass
class Row:
    scope_id: str
    created_ts: int
    nonce: Optional[bytes]
    blob: bytes
    aad_json: str


class SQLiteKV:
    """
    SQLite helper that stores rows:
      mem(scope_id, created_ts, nonce, blob, aad_json)

    Baselines share this storage layout so comparison is fair.
    """
    def __init__(self, db_path: str):
        ensure_dir(db_path)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mem (
              scope_id   TEXT NOT NULL,
              created_ts INTEGER NOT NULL,
              nonce      BLOB,
              blob       BLOB NOT NULL,
              aad_json   TEXT NOT NULL
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope_ts ON mem(scope_id, created_ts DESC);")
        self.conn.commit()

    def put_row(self, scope_id: str, nonce: Optional[bytes], blob: bytes, aad: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO mem(scope_id, created_ts, nonce, blob, aad_json) VALUES(?,?,?,?,?)",
            (scope_id, now_ts(), nonce, blob, json.dumps(aad, separators=(",", ":"))),
        )
        self.conn.commit()

    def get_recent_rows(self, scope_id: str, limit: int) -> List[Row]:
        cur = self.conn.execute(
            "SELECT scope_id, created_ts, nonce, blob, aad_json FROM mem WHERE scope_id=? ORDER BY created_ts DESC LIMIT ?",
            (scope_id, limit),
        )
        out: List[Row] = []
        for scope_id, created_ts, nonce, blob, aad_json in cur.fetchall():
            out.append(Row(scope_id=scope_id, created_ts=created_ts, nonce=nonce, blob=blob, aad_json=aad_json))
        return out

    def delete_scope_rows(self, scope_id: str) -> int:
        cur = self.conn.execute("DELETE FROM mem WHERE scope_id=?", (scope_id,))
        self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()
