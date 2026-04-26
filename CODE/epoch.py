# epoch.py
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Protocol


class EpochProvider(Protocol):
    def get_epoch(self, scope_id: str) -> int: ...
    def bump_epoch(self, scope_id: str) -> int: ...


class SQLiteEpochStore:
    """
    Epoch store backed by a SEPARATE SQLite database with WAL + FULL sync.

    Advantages over LocalEpochStore (JSON file):
      1. Atomic durability: PRAGMA synchronous=FULL + WAL fsyncs each bump
         before returning, preventing torn writes.
      2. File separation: epoch DB is independent from the memory DB.
         Rolling back only the memory DB is insufficient — an attacker must
         coordinate two separate files (ideally on different volumes).
      3. Atomic increment: INSERT...ON CONFLICT DO UPDATE SET epoch=epoch+1
         is a single atomic SQLite statement with no read-modify-write race.

    Limitation: still not hardware-backed.  A privileged attacker who can
    replace BOTH databases simultaneously retains rollback capability.
    For full rollback protection use TPMEpochStore.

    In-process cache: epoch values are cached after first read and invalidated
    on every bump_epoch(), so get_epoch() avoids a DB round-trip on the hot
    path. Cache is per-process (each worker has its own instance under mp).
    """

    def __init__(self, db_path: str = "epochs.db") -> None:
        self.path = db_path
        self._lock = threading.Lock()
        self._cache: dict[str, int] = {}
        self._conn = self._open()
        self._init()

    def _open(self) -> sqlite3.Connection:
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _init(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS epochs (
                scope_id TEXT PRIMARY KEY,
                epoch    INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def get_epoch(self, scope_id: str) -> int:
        with self._lock:
            if scope_id in self._cache:
                return self._cache[scope_id]
            row = self._conn.execute(
                "SELECT epoch FROM epochs WHERE scope_id = ?", (scope_id,)
            ).fetchone()
            val = int(row[0]) if row else 0
            self._cache[scope_id] = val
            return val

    def bump_epoch(self, scope_id: str) -> int:
        """Atomically increment and durably persist epoch. Returns new value."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO epochs(scope_id, epoch) VALUES(?, 1)
                ON CONFLICT(scope_id) DO UPDATE SET epoch = epoch + 1
                """,
                (scope_id,),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT epoch FROM epochs WHERE scope_id = ?", (scope_id,)
            ).fetchone()
            val = int(row[0])
            self._cache[scope_id] = val
            return val

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class LocalEpochStore:
    """
    JSON-backed epoch store.

    Not rollback-protected (host can roll back the JSON file alongside the DB).
    Use this for functional testing or file-backed compatibility scenarios.

    For a stronger software-only option use SQLiteEpochStore (separate DB, FULL sync).
    For hardware-backed rollback protection use TPMEpochStore.

    In-process cache: epoch values are cached after first read and invalidated
    on every bump_epoch(). Cache is per-process under multiprocessing.
    """

    def __init__(self, path: str = "epochs.json") -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: dict[str, int] = {}
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load(self) -> dict[str, int]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.items()}

    def _save(self, data: dict[str, int]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def get_epoch(self, scope_id: str) -> int:
        with self._lock:
            if scope_id in self._cache:
                return self._cache[scope_id]
            data = self._load()
            val = int(data.get(scope_id, 0))
            self._cache[scope_id] = val
            return val

    def bump_epoch(self, scope_id: str) -> int:
        with self._lock:
            data = self._load()
            data[scope_id] = int(data.get(scope_id, 0)) + 1
            self._save(data)
            self._cache[scope_id] = data[scope_id]
            return data[scope_id]


def make_epoch_store(mode: str = "sqlite", path: str = "epochs.db"):
    """
    Factory for selecting an epoch store at runtime.

    mode="sqlite" -> SQLiteEpochStore (separate DB, WAL+FULL sync — recommended)
    mode="json"   -> LocalEpochStore  (JSON file, functional testing only)
    mode="tpm"    -> TPMEpochStore    (TPM-backed, hardware rollback protection)
    """
    if mode == "sqlite":
        return SQLiteEpochStore(db_path=path)
    if mode == "json":
        return LocalEpochStore(path=path)
    if mode == "tpm":
        from tpm_epoch import TPMEpochStore
        return TPMEpochStore(state_path=path)
    raise ValueError(f"Unknown epoch store mode: {mode!r}. Choose: sqlite, json, tpm")
