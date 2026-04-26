# baselines/kms.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from .common import SQLiteKV, aead_encrypt, aead_decrypt


class FileBackedKMS:
    """
    File-backed key service for the KMS baseline.

    For the rollback experiment, this state is treated as trusted and not
    subject to storage rollback.
    """

    def __init__(self, state_path: str = "kms_state.json"):
        self.state_path = state_path

        if not os.path.exists(self.state_path):
            with open(self.state_path, "w") as f:
                json.dump({"scopes": {}}, f)

        # Load once per process, robust to empty/partial files.
        try:
            if os.path.getsize(self.state_path) == 0:
                self.state = {"scopes": {}}
            else:
                with open(self.state_path, "r") as f:
                    data = f.read().strip()
                    self.state = json.loads(data) if data else {"scopes": {}}
        except Exception:
            self.state = {"scopes": {}}

        # Ensure expected shape
        if not isinstance(self.state, dict) or "scopes" not in self.state:
            self.state = {"scopes": {}}

    def _load(self) -> Dict[str, Any]:
        return self.state

    def _save(self, obj: Dict[str, Any]) -> None:
        self.state = obj
        with open(self.state_path, "w") as f:
            json.dump(obj, f, indent=2)

    def get_key(self, scope_id: str) -> Tuple[int, bytes]:
        obj = self._load()
        scopes = obj["scopes"]

        if scope_id not in scopes:
            scopes[scope_id] = {
                "ver": 0,
                "key": os.urandom(32).hex(),
                "deleted": False,
            }
            self._save(obj)

        rec = scopes[scope_id]

        if rec.get("deleted", False):
            raise KeyError(f"KMS: key for scope {scope_id} deleted")

        return int(rec["ver"]), bytes.fromhex(rec["key"])

    def delete_key(self, scope_id: str) -> Dict[str, Any]:
        obj = self._load()
        scopes = obj["scopes"]

        if scope_id not in scopes:
            scopes[scope_id] = {
                "ver": 0,
                "key": os.urandom(32).hex(),
                "deleted": True,
            }
        else:
            scopes[scope_id]["deleted"] = True
            scopes[scope_id]["ver"] = int(scopes[scope_id]["ver"]) + 1

        self._save(obj)

        return {
            "scope_id": scope_id,
            "kms_ver": scopes[scope_id]["ver"],
            "deleted": True,
        }


class TrustedKMSDesign:
    """
    Baseline 4: keys live in a trusted external service (KMS).
    Deletion is enforced by deleting the key at the KMS.
    """

    def __init__(self, db_path: str, kms_state_path: str = "kms_state.json"):
        self.kv = SQLiteKV(db_path)
        self.kms = FileBackedKMS(kms_state_path)

    def put(self, scope_id: str, payload: Dict[str, Any]) -> None:
        ver, key = self.kms.get_key(scope_id)
        pt = json.dumps(payload).encode("utf-8")
        aad = json.dumps(
            {"scheme": "kms", "scope": scope_id, "ver": ver},
            separators=(",", ":"),
        ).encode("utf-8")
        nonce, ct = aead_encrypt(key, pt, aad)
        self.kv.put_row(
            scope_id,
            nonce=nonce,
            blob=ct,
            aad={"scheme": "kms", "ver": ver},
        )

    def get_recent(self, scope_id: str, limit: int) -> List[Dict[str, Any]]:
        # If key is deleted, KMS denies. That enforces forgetting even after rollback.
        ver, key = self.kms.get_key(scope_id)  # raises if deleted
        aad = json.dumps(
            {"scheme": "kms", "scope": scope_id, "ver": ver},
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
        kms_res = self.kms.delete_key(scope_id)
        # optional: delete rows too (not required, but typical)
        n = self.kv.delete_scope_rows(scope_id)
        return {
            "scheme": "kms",
            "scope_id": scope_id,
            "deleted_rows": n,
            **kms_res,
        }

    def close(self) -> None:
        self.kv.close()
