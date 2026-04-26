# baselines/plain.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from .common import SQLiteKV


class PlainLogicalDelete:
    """
    Baseline 1: plaintext storage + logical deletion.
    Rollback resurrects deleted data.
    """
    def __init__(self, db_path: str):
        self.kv = SQLiteKV(db_path)

    def put(self, scope_id: str, payload: Dict[str, Any]) -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.kv.put_row(scope_id, nonce=None, blob=blob, aad={"scheme": "plain"})

    def get_recent(self, scope_id: str, limit: int) -> List[Dict[str, Any]]:
        rows = self.kv.get_recent_rows(scope_id, limit)
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                out.append(json.loads(r.blob.decode("utf-8")))
            except Exception:
                pass
        return out

    def forget_scope(self, scope_id: str) -> Dict[str, Any]:
        n = self.kv.delete_scope_rows(scope_id)
        return {"scheme": "plain", "scope_id": scope_id, "deleted_rows": n}

    def close(self) -> None:
        self.kv.close()