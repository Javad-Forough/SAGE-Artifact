from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class StoredRecord:
    scope_id: str
    epoch: int
    item_id: str
    kind: str
    created_ts: int
    nonce: bytes
    ct: bytes
    derived_from_item_ids: tuple[str, ...] = field(default_factory=tuple)

    # Plain source scope IDs for graph traversal / convenience.
    source_scope_ids: tuple[str, ...] = field(default_factory=tuple)

    # Rollback-resilient dependency tokens, each encoding {"scope": ..., "epoch": ...}.
    source_scope_deps: tuple[str, ...] = field(default_factory=tuple)

    is_active: bool = True
    invalidated_ts: Optional[int] = None
    invalidation_reason: Optional[str] = None


class SQLiteMemoryStore:
    def __init__(self, db_path: str = "sealed_mem.db"):
        self.db_path = db_path
        # Persistent connection: opened once per instance (once per process in the
        # benchmark), reused for all operations.  This matches the baseline design
        # (SQLiteKV) and gives a fair apples-to-apples concurrency comparison.
        self._conn = self._open_connection()
        self._init_db()

    def _open_connection(self) -> sqlite3.Connection:
        import os
        d = os.path.dirname(os.path.abspath(self.db_path))
        if d:
            os.makedirs(d, exist_ok=True)
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _table_columns(self, table_name: str) -> set[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(r[1]) for r in rows}

    def _ensure_column(self, table_name: str, column_sql: str, column_name: str) -> None:
        cols = self._table_columns(table_name)
        if column_name not in cols:
            self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")

    @staticmethod
    def _to_json_list(values: Optional[Iterable[str]]) -> str:
        vals = [] if values is None else [str(v) for v in values if str(v)]
        return json.dumps(sorted(set(vals)), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _from_json_list(raw: Optional[str]) -> tuple[str, ...]:
        if not raw:
            return ()
        try:
            data = json.loads(raw)
        except Exception:
            return ()
        if not isinstance(data, list):
            return ()
        return tuple(str(v) for v in data if str(v))

    @staticmethod
    def _encode_scope_dep(scope_id: str, epoch: int) -> str:
        return json.dumps(
            {"scope": str(scope_id), "epoch": int(epoch)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
              scope_id               TEXT NOT NULL,
              epoch                  INTEGER NOT NULL,
              item_id                TEXT NOT NULL,
              kind                   TEXT NOT NULL,
              created_ts             INTEGER NOT NULL,
              nonce                  BLOB NOT NULL,
              ct                     BLOB NOT NULL,
              derived_from_json      TEXT NOT NULL DEFAULT '[]',
              source_scope_ids_json  TEXT NOT NULL DEFAULT '[]',
              source_scope_deps_json TEXT NOT NULL DEFAULT '[]',
              is_active              INTEGER NOT NULL DEFAULT 1,
              invalidated_ts         INTEGER,
              invalidation_reason    TEXT,
              PRIMARY KEY(scope_id, epoch, item_id)
            );
            """
        )

        self._ensure_column("memory", "derived_from_json TEXT NOT NULL DEFAULT '[]'", "derived_from_json")
        self._ensure_column("memory", "source_scope_ids_json TEXT NOT NULL DEFAULT '[]'", "source_scope_ids_json")
        self._ensure_column("memory", "source_scope_deps_json TEXT NOT NULL DEFAULT '[]'", "source_scope_deps_json")
        self._ensure_column("memory", "is_active INTEGER NOT NULL DEFAULT 1", "is_active")
        self._ensure_column("memory", "invalidated_ts INTEGER", "invalidated_ts")
        self._ensure_column("memory", "invalidation_reason TEXT", "invalidation_reason")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_item_edges (
              parent_item_id TEXT NOT NULL,
              child_item_id  TEXT NOT NULL,
              relation       TEXT NOT NULL DEFAULT 'derived_from',
              PRIMARY KEY(parent_item_id, child_item_id)
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_scope_edges (
              source_scope_id TEXT NOT NULL,
              child_item_id   TEXT NOT NULL,
              relation        TEXT NOT NULL DEFAULT 'source_scope',
              PRIMARY KEY(source_scope_id, child_item_id)
            );
            """
        )

        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_scope_epoch ON memory(scope_id, epoch, is_active, created_ts DESC);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_scope_kind ON memory(scope_id, kind, is_active);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_item_id ON memory(item_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_item_edges_parent ON memory_item_edges(parent_item_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_item_edges_child ON memory_item_edges(child_item_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_scope_edges_scope ON memory_scope_edges(source_scope_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_scope_edges_child ON memory_scope_edges(child_item_id);")
        self._conn.commit()

    def put(self, rec: StoredRecord) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memory(
                scope_id, epoch, item_id, kind, created_ts, nonce, ct,
                derived_from_json, source_scope_ids_json, source_scope_deps_json,
                is_active, invalidated_ts, invalidation_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.scope_id,
                rec.epoch,
                rec.item_id,
                rec.kind,
                rec.created_ts,
                rec.nonce,
                rec.ct,
                self._to_json_list(rec.derived_from_item_ids),
                self._to_json_list(rec.source_scope_ids),
                self._to_json_list(rec.source_scope_deps),
                1 if rec.is_active else 0,
                rec.invalidated_ts,
                rec.invalidation_reason,
            ),
        )

        # Refresh provenance edges for this child item.
        self._conn.execute("DELETE FROM memory_item_edges WHERE child_item_id = ?", (rec.item_id,))
        if rec.derived_from_item_ids:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO memory_item_edges(parent_item_id, child_item_id, relation)
                VALUES (?, ?, 'derived_from')
                """,
                [(parent_id, rec.item_id) for parent_id in rec.derived_from_item_ids],
            )

        self._conn.execute("DELETE FROM memory_scope_edges WHERE child_item_id = ?", (rec.item_id,))
        if rec.source_scope_ids:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO memory_scope_edges(source_scope_id, child_item_id, relation)
                VALUES (?, ?, 'source_scope')
                """,
                [(scope_id, rec.item_id) for scope_id in rec.source_scope_ids],
            )
        self._conn.commit()

    def get_by_scope_epoch(
        self,
        scope_id: str,
        epoch: int,
        limit: int = 100,
        active_only: bool = True,
        kinds: Optional[Iterable[str]] = None,
    ) -> list[StoredRecord]:
        params: list[object] = [scope_id, epoch]
        where = "WHERE scope_id = ? AND epoch = ?"
        if active_only:
            where += " AND is_active = 1"
        if kinds:
            kinds_list = [str(k) for k in kinds if str(k)]
            if kinds_list:
                placeholders = ",".join("?" for _ in kinds_list)
                where += f" AND kind IN ({placeholders})"
                params.extend(kinds_list)
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT scope_id, epoch, item_id, kind, created_ts, nonce, ct,
                   derived_from_json, source_scope_ids_json, source_scope_deps_json,
                   is_active, invalidated_ts, invalidation_reason
            FROM memory
            {where}
            ORDER BY created_ts DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        out: list[StoredRecord] = []
        for row in rows:
            out.append(
                StoredRecord(
                    scope_id=row[0],
                    epoch=row[1],
                    item_id=row[2],
                    kind=row[3],
                    created_ts=row[4],
                    nonce=row[5],
                    ct=row[6],
                    derived_from_item_ids=self._from_json_list(row[7]),
                    source_scope_ids=self._from_json_list(row[8]),
                    source_scope_deps=self._from_json_list(row[9]),
                    is_active=bool(row[10]),
                    invalidated_ts=row[11],
                    invalidation_reason=row[12],
                )
            )
        return out

    def get_item_ids_by_scope(self, scope_id: str, active_only: bool = False) -> list[str]:
        sql = "SELECT item_id FROM memory WHERE scope_id = ?"
        params: list[object] = [scope_id]
        if active_only:
            sql += " AND is_active = 1"
        rows = self._conn.execute(sql, params).fetchall()
        return [str(r[0]) for r in rows]

    def get_dependency_tokens_for_items(self, item_ids: Iterable[str]) -> tuple[str, ...]:
        """
        Return the full dependency token set for a collection of parent items.

        For each parent item, include:
          - the parent's own (scope, epoch) token
          - any transitive source-scope dependency tokens already stored on that parent
        """
        ids = [str(i) for i in item_ids if str(i)]
        if not ids:
            return ()

        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT scope_id, epoch, source_scope_deps_json
            FROM memory
            WHERE item_id IN ({placeholders})
            """,
            ids,
        ).fetchall()

        deps: set[str] = set()
        for scope_id, epoch, source_scope_deps_json in rows:
            deps.add(self._encode_scope_dep(str(scope_id), int(epoch)))
            deps.update(self._from_json_list(source_scope_deps_json))
        return tuple(sorted(deps))

    def get_child_item_ids_for_parents(self, parent_item_ids: Iterable[str], active_only: bool = True) -> list[str]:
        parents = [str(p) for p in parent_item_ids if str(p)]
        if not parents:
            return []

        placeholders = ",".join("?" for _ in parents)
        sql = f"""
            SELECT DISTINCT e.child_item_id
            FROM memory_item_edges e
            JOIN memory m ON m.item_id = e.child_item_id
            WHERE e.parent_item_id IN ({placeholders})
        """
        params: list[object] = list(parents)
        if active_only:
            sql += " AND m.is_active = 1"
        rows = self._conn.execute(sql, params).fetchall()
        return [str(r[0]) for r in rows]

    def get_child_item_ids_for_scopes(self, scope_ids: Iterable[str], active_only: bool = True) -> list[str]:
        scopes = [str(s) for s in scope_ids if str(s)]
        if not scopes:
            return []

        placeholders = ",".join("?" for _ in scopes)
        sql = f"""
            SELECT DISTINCT e.child_item_id
            FROM memory_scope_edges e
            JOIN memory m ON m.item_id = e.child_item_id
            WHERE e.source_scope_id IN ({placeholders})
        """
        params: list[object] = list(scopes)
        if active_only:
            sql += " AND m.is_active = 1"
        rows = self._conn.execute(sql, params).fetchall()
        return [str(r[0]) for r in rows]

    def invalidate_items(self, item_ids: Iterable[str], reason: str, invalidated_ts: Optional[int] = None) -> int:
        ids = [str(i) for i in item_ids if str(i)]
        if not ids:
            return 0

        invalidated_ts = int(invalidated_ts or __import__("time").time())
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"""
            UPDATE memory
            SET is_active = 0,
                invalidated_ts = COALESCE(invalidated_ts, ?),
                invalidation_reason = COALESCE(invalidation_reason, ?)
            WHERE item_id IN ({placeholders}) AND is_active = 1
            """,
            [invalidated_ts, reason, *ids],
        )
        self._conn.commit()
        return int(cur.rowcount)

    def delete_scope_all_epochs(self, scope_id: str) -> int:
        """
        Delete all ciphertext rows for a scope across all epochs, and clean up edge tables.
        Uses subqueries to avoid large IN-list parameters on big scopes.
        """
        self._conn.execute(
            """DELETE FROM memory_item_edges
               WHERE child_item_id  IN (SELECT item_id FROM memory WHERE scope_id = ?)
                  OR parent_item_id IN (SELECT item_id FROM memory WHERE scope_id = ?)""",
            (scope_id, scope_id),
        )
        self._conn.execute(
            "DELETE FROM memory_scope_edges WHERE child_item_id IN (SELECT item_id FROM memory WHERE scope_id = ?)",
            (scope_id,),
        )
        cur = self._conn.execute("DELETE FROM memory WHERE scope_id = ?", (scope_id,))
        self._conn.commit()
        return int(cur.rowcount)

    def delete_items_by_ids(self, item_ids: Iterable[str]) -> int:
        """
        Physically delete rows for the given item IDs from memory and all edge tables.
        Used to purge invalidated dependent artifacts rather than leaving is_active=0 rows.
        """
        ids = [str(i) for i in item_ids if str(i)]
        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"DELETE FROM memory_item_edges WHERE child_item_id IN ({placeholders}) OR parent_item_id IN ({placeholders})",
            [*ids, *ids],
        )
        self._conn.execute(
            f"DELETE FROM memory_scope_edges WHERE child_item_id IN ({placeholders})",
            ids,
        )
        cur = self._conn.execute(
            f"DELETE FROM memory WHERE item_id IN ({placeholders})",
            ids,
        )
        self._conn.commit()
        return int(cur.rowcount)

    def collect_all_transitive_dependents(
        self, scope_id: str
    ) -> tuple[set[str], set[str]]:
        """
        Return (seed_items, dependent_items) using a SINGLE recursive SQL CTE.

        seed_items      — all item IDs belonging to scope_id (all epochs)
        dependent_items — all items in OTHER scopes that transitively derive
                          from any seed item via memory_item_edges

        Compared to the per-node BFS in SealedAgentMemoryService:
          - O(1) SQL round-trips regardless of provenance graph depth
          - SQLite handles the graph traversal inside the engine in one pass
          - Correct for DAGs and isolated nodes (UNION deduplicates)

        Requires SQLite ≥ 3.8.3 (WITH RECURSIVE — available since 2014).
        """
        seed_rows = self._conn.execute(
            "SELECT item_id FROM memory WHERE scope_id = ?", (scope_id,)
        ).fetchall()
        seed_items = {str(r[0]) for r in seed_rows}

        # Use JOIN-based recursive CTE — avoids large IN-list parameters.
        # Starts from items that belong to scope_id (via memory JOIN) or that
        # explicitly declare scope_id as a source (via memory_scope_edges).
        rows = self._conn.execute(
            """
            WITH RECURSIVE deps(item_id) AS (
                SELECT DISTINCT e.child_item_id
                FROM memory_item_edges e
                JOIN memory m ON m.item_id = e.parent_item_id
                WHERE m.scope_id = ?
                UNION
                SELECT child_item_id
                FROM memory_scope_edges
                WHERE source_scope_id = ?
                UNION
                SELECT e.child_item_id
                FROM memory_item_edges e
                INNER JOIN deps d ON e.parent_item_id = d.item_id
            )
            SELECT DISTINCT item_id FROM deps
            """,
            (scope_id, scope_id),
        ).fetchall()

        all_descendants = {str(r[0]) for r in rows}
        dependent_items = all_descendants - seed_items
        return seed_items, dependent_items

    def count_scope(self, scope_id: str) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memory WHERE scope_id = ?", (scope_id,)).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
