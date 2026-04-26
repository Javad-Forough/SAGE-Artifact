from __future__ import annotations

import json
import os
import secrets
import sqlite3
import tempfile
from contextlib import contextmanager
from typing import Any, List

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

try:
    import fcntl
except ImportError:  # non-POSIX fallback
    fcntl = None


class SQLiteEnvelopeEncryption:
    """
    Practical baseline:
    SQLite storage + per-scope AES envelope encryption.

    Each scope gets its own symmetric key stored in a local key file.
    Deleting a scope removes the key and the rows.

    If both the DB and key file are rolled back together,
    deleted data becomes recoverable.

    This implementation is made safe for concurrent multi-process
    evaluation by using:
      - a lock file for cross-process synchronization
      - tolerant JSON loading
      - atomic writes via os.replace()
    """

    def __init__(self, db_path: str, key_store_path: str = "sqlite_envelope_keys.json"):
        self.db_path = db_path
        self.key_store_path = key_store_path
        self.lock_path = key_store_path + ".lock"

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(key_store_path) or ".", exist_ok=True)

        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self._init_db()

        # Ensure key store exists in a valid format.
        with self._locked():
            if not os.path.exists(self.key_store_path):
                self._save_keys_unlocked({})

    # --------------------------------------------------

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                scope TEXT,
                nonce BLOB,
                ciphertext BLOB
            )
            """
        )
        self.conn.commit()

    # --------------------------------------------------
    # Locking / JSON helpers
    # --------------------------------------------------

    @contextmanager
    def _locked(self):
        """
        Cross-process lock guarding the key-store file.
        """
        with open(self.lock_path, "a+b") as lockf:
            if fcntl is not None:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)

    def _load_keys_unlocked(self) -> dict[str, str]:
        """
        Best-effort load. Empty/partial/corrupt files are treated as empty.
        """
        if not os.path.exists(self.key_store_path):
            return {}

        try:
            with open(self.key_store_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if not raw:
                    return {}
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
                return {}
        except Exception:
            return {}

    def _save_keys_unlocked(self, keys: dict[str, str]) -> None:
        """
        Atomic save to avoid readers observing partially written JSON.
        """
        parent = os.path.dirname(self.key_store_path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".envkeys.", suffix=".json.tmp", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(keys, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.key_store_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def _load_keys(self) -> dict[str, str]:
        with self._locked():
            return self._load_keys_unlocked()

    # --------------------------------------------------

    def _get_scope_key(self, scope: str) -> bytes:
        with self._locked():
            keys = self._load_keys_unlocked()

            if scope not in keys:
                key = secrets.token_bytes(32)
                keys[scope] = key.hex()
                self._save_keys_unlocked(keys)

            return bytes.fromhex(keys[scope])

    # --------------------------------------------------

    def put(self, scope: str, payload: Any):
        key = self._get_scope_key(scope)

        aes = AESGCM(key)
        nonce = secrets.token_bytes(12)

        plaintext = json.dumps(payload).encode("utf-8")
        ciphertext = aes.encrypt(nonce, plaintext, None)

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO memory VALUES (?, ?, ?)",
            (scope, nonce, ciphertext),
        )
        self.conn.commit()

    # --------------------------------------------------

    def get_recent(self, scope: str, limit: int = 50) -> List[Any]:
        keys = self._load_keys()

        if scope not in keys:
            return []

        key = bytes.fromhex(keys[scope])
        aes = AESGCM(key)

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT nonce, ciphertext
            FROM memory
            WHERE scope = ?
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (scope, limit),
        )

        rows = cur.fetchall()
        out = []

        for nonce, ct in rows:
            try:
                pt = aes.decrypt(nonce, ct, None)
                out.append(json.loads(pt))
            except Exception:
                continue

        return out

    # --------------------------------------------------

    def forget_scope(self, scope: str):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM memory WHERE scope = ?", (scope,))
        self.conn.commit()

        with self._locked():
            keys = self._load_keys_unlocked()
            if scope in keys:
                del keys[scope]
                self._save_keys_unlocked(keys)

    # --------------------------------------------------

    def close(self):
        self.conn.close()