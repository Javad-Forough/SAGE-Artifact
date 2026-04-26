from __future__ import annotations

import argparse
import csv
import os
import shutil
import uuid
from typing import Any, Iterable

from experiments.schemes import (
    SchemeConfig,
    assign_artifact_paths,
    destroy_scheme_persistent_state,
    make_scheme,
    rollback_restorable_aux_state_paths,
    scheme_artifact_paths,
)

try:
    from experiments.utils import clean
except Exception:
    def clean(paths: list[str]) -> None:
        for p in paths:
            if p and os.path.exists(p):
                os.remove(p)


def safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def safe_close(obj: Any) -> None:
    if hasattr(obj, "close"):
        try:
            obj.close()
        except Exception:
            pass


def snapshot_file(src: str) -> str | None:
    if src and os.path.exists(src):
        dst = src + ".snapshot"
        shutil.copy2(src, dst)
        return dst
    return None


def restore_file(dst: str, snap: str) -> None:
    if snap and os.path.exists(snap):
        shutil.copy2(snap, dst)


def snapshot_scheme_state(cfg: Any) -> dict[str, str]:
    snaps: dict[str, str] = {}
    for p in rollback_restorable_aux_state_paths(cfg):
        if isinstance(p, str) and os.path.exists(p):
            snap = snapshot_file(p)
            if snap:
                snaps[p] = snap
    return snaps


def restore_scheme_state(snaps: dict[str, str]) -> None:
    for orig, snap in snaps.items():
        restore_file(orig, snap)


def cleanup_snapshots(snaps: dict[str, str]) -> None:
    for snap in snaps.values():
        safe_remove(snap)


def snapshot(db_path: str, snap_path: str) -> None:
    if os.path.exists(db_path):
        shutil.copy2(db_path, snap_path)


def rollback(db_snap: str, db_path: str) -> None:
    if os.path.exists(db_snap):
        shutil.copy2(db_snap, db_path)


def make_trial_cfg(scheme: str, db_path: str, trial_tag: str) -> SchemeConfig:
    cfg = SchemeConfig(db_path=db_path, scheme=scheme)
    return assign_artifact_paths(cfg, f"{db_path}.{trial_tag}")


def cleanup_trial_files(cfg: SchemeConfig, db_path: str, snap_path: str) -> None:
    destroy_scheme_persistent_state(cfg)
    safe_remove(snap_path)
    for p in scheme_artifact_paths(cfg, include_wal=True):
        safe_remove(p)


SPECS = {
    "personal": {
        "source_kind": "fact",
        "source_payload": {"fact": "my favorite drink is tea"},
        "derived_kind": "profile_summary",
        "derived_payload": {"summary": "User profile summary: favorite drink is tea"},
        "expected": "tea",
    },
    "team": {
        "source_kind": "team_note",
        "source_payload": {"note": "the sprint deadline is Friday"},
        "derived_kind": "team_digest",
        "derived_payload": {"digest": "Team digest: the sprint deadline is Friday"},
        "expected": "friday",
    },
    "research": {
        "source_kind": "research_doc",
        "source_payload": {"doc": "SAGE uses epoch-based key derivation for rollback-resilient deletion."},
        "derived_kind": "research_summary",
        "derived_payload": {"summary": "Research summary: SAGE uses epoch-based key derivation for rollback-resilient deletion."},
        "expected": "epoch",
    },
}


def store_item(s, scope_id: str, payload: dict[str, Any], kind: str) -> str:
    try:
        return s.put(scope_id, payload, kind=kind)
    except TypeError:
        return s.put(scope_id, payload)


def store_derived(s, scope_id: str, payload: dict[str, Any], kind: str, derived_from_item_ids: Iterable[str], source_scope_ids: Iterable[str]) -> str:
    if hasattr(s, "put_derived"):
        try:
            return s.put_derived(
                scope_id,
                payload,
                kind=kind,
                derived_from_item_ids=list(derived_from_item_ids),
                source_scope_ids=list(source_scope_ids),
            )
        except Exception:
            pass
    if hasattr(s, "mem") and hasattr(s.mem, "put_derived"):
        return s.mem.put_derived(
            scope_id,
            payload,
            kind=kind,
            derived_from_item_ids=list(derived_from_item_ids),
            source_scope_ids=list(source_scope_ids),
        )
    return store_item(s, scope_id, payload, kind)


def _extract_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, str):
                out.append(v)
    elif isinstance(obj, str):
        out.append(obj)
    return out


def contains_token(items: list[dict[str, Any]], token: str) -> bool:
    t = token.lower()
    for it in items:
        payload = it.get("payload", it)
        for s in _extract_strings(payload):
            if t in s.lower():
                return True
    return False


def run_one(scheme: str, agent_kind: str, db: str, snap: str) -> dict[str, Any]:
    trial_tag = uuid.uuid4().hex
    cfg = make_trial_cfg(scheme, db, trial_tag)
    cleanup_trial_files(cfg, db, snap)

    spec = SPECS[agent_kind]
    source_scope = f"{agent_kind}:{scheme}:{trial_tag}:source"
    derived_scope = f"{agent_kind}:{scheme}:{trial_tag}:derived"

    s = make_scheme(cfg)
    source_id = store_item(s, source_scope, spec["source_payload"], spec["source_kind"])
    store_derived(
        s,
        derived_scope,
        spec["derived_payload"],
        spec["derived_kind"],
        derived_from_item_ids=[source_id],
        source_scope_ids=[source_scope],
    )
    safe_close(s)

    snapshot(db, snap)
    state_snaps = snapshot_scheme_state(cfg)

    s = make_scheme(cfg)
    try:
        s.forget_scope(source_scope, delete_ciphertext_rows=True)
    except TypeError:
        s.forget_scope(source_scope)
    derived_hidden_after_forget = not contains_token(s.get_recent(derived_scope, limit=10), spec["expected"])
    safe_close(s)

    rollback(snap, db)
    if rollback_restorable_aux_state_paths(cfg):
        restore_scheme_state(state_snaps)

    s = make_scheme(cfg)
    try:
        source_recovered = contains_token(s.get_recent(source_scope, limit=10), spec["expected"])
    except KeyError:
        source_recovered = False
    try:
        derived_recovered = contains_token(s.get_recent(derived_scope, limit=10), spec["expected"])
    except KeyError:
        derived_recovered = False
    safe_close(s)

    cleanup_snapshots(state_snaps)
    cleanup_trial_files(cfg, db, snap)

    return {
        "scheme": scheme,
        "agent": agent_kind,
        "source_scope": source_scope,
        "derived_scope": derived_scope,
        "derived_hidden_after_forget": int(derived_hidden_after_forget),
        "source_recovered_after_rollback": int(source_recovered),
        "derived_recovered_after_rollback": int(derived_recovered),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rollback_matrix.csv")
    ap.add_argument("--db", default="results/rollback_mem.db")
    ap.add_argument("--snap", default="results/rollback_mem.db.snapshot")
    ap.add_argument("--schemes", default="plain,static,sealed_no_rp,sqlite_envelope,kms,sage")
    ap.add_argument("--agents", default="personal,team,research")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]

    rows = [run_one(scheme, agent_kind, args.db, args.snap) for agent_kind in agents for scheme in schemes]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
