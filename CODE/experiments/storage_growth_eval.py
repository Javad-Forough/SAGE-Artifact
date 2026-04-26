from __future__ import annotations

import argparse
import csv
import os
import random
import sqlite3
from typing import Dict, List

from experiments.schemes import (
    SchemeConfig,
    assign_artifact_paths,
    make_scheme,
    scheme_artifact_paths,
)

WORKLOADS = ["chat", "assistant", "research"]


def db_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def _user_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        """
    )
    return [r[0] for r in cur.fetchall()]


def row_count(db_path: str) -> int:
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    total = 0
    try:
        cur = conn.cursor()
        for t in _user_tables(conn):
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{t}"')
                total += int(cur.fetchone()[0])
            except Exception:
                pass
    finally:
        conn.close()
    return total


def table_count(db_path: str) -> int:
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        return len(_user_tables(conn))
    finally:
        conn.close()


def workload_profile(workload: str) -> Dict:
    if workload == "chat":
        return {"put_prob": 0.40, "derived_prob": 0.20, "forget_prob": 0.05, "scopes": 40}
    if workload == "assistant":
        return {"put_prob": 0.55, "derived_prob": 0.25, "forget_prob": 0.15, "scopes": 24}
    # research: put_prob + derived_prob must be < 1.0 so the else branch (forgets) is reachable.
    # 0.60 + 0.30 = 0.90 → 10% of events reach the forget logic; effective forget rate ≈ 1%.
    return {"put_prob": 0.60, "derived_prob": 0.30, "forget_prob": 0.10, "scopes": 14}


def make_payload(workload: str, i: int, scope: str) -> Dict:
    if workload == "chat":
        return {"kind": "chat_memory", "scope": scope, "summary": f"Short conversational summary {i}", "ts": i}
    if workload == "assistant":
        return {"kind": "assistant_memory", "scope": scope, "task_id": f"{scope}-task-{i}", "ts": i}
    return {"kind": "research_memory", "scope": scope, "topic": f"{scope}-topic-{i}", "summary": f"Dense research synthesis note {i}", "ts": i}


def run(cfg: SchemeConfig, workload: str, events: int) -> List[Dict]:
    s = make_scheme(cfg)
    profile = workload_profile(workload)
    active_scopes = [f"{workload}_scope_{i}" for i in range(profile["scopes"])]
    next_scope_id = profile["scopes"]
    deletions = 0
    derived_puts = 0
    checkpoints: List[Dict] = []
    recent_item_by_scope: Dict[str, str] = {}

    for i in range(events):
        if not active_scopes:
            active_scopes.append(f"{workload}_scope_{next_scope_id}")
            next_scope_id += 1
        scope = random.choice(active_scopes)
        r = random.random()

        if r < profile["put_prob"]:
            payload = make_payload(workload, i, scope)
            try:
                item_id = s.put(scope, payload, kind=str(payload.get("kind", "fact")))
            except TypeError:
                item_id = s.put(scope, payload)
            except KeyError:
                if scope in active_scopes:
                    active_scopes.remove(scope)
                fresh = f"{workload}_scope_{next_scope_id}"
                next_scope_id += 1
                active_scopes.append(fresh)
                item_id = s.put(fresh, payload)
                scope = fresh
            recent_item_by_scope[scope] = item_id

        elif r < profile["put_prob"] + profile["derived_prob"] and len(active_scopes) >= 2:
            src_scope = random.choice(active_scopes)
            dst_scope = random.choice([x for x in active_scopes if x != src_scope])
            parent_id = recent_item_by_scope.get(src_scope)
            if parent_id is not None:
                payload = make_payload(workload, i, dst_scope)
                try:
                    s.put_derived(
                        dst_scope,
                        payload,
                        kind=str(payload.get("kind", "fact")),
                        derived_from_item_ids=[parent_id],
                        source_scope_ids=[src_scope],
                    )
                    derived_puts += 1
                except Exception:
                    try:
                        s.put(dst_scope, payload, kind=str(payload.get("kind", "fact")))
                    except TypeError:
                        s.put(dst_scope, payload)

        else:
            if random.random() < profile["forget_prob"] and len(active_scopes) > 5:
                try:
                    s.forget_scope(scope)
                except KeyError:
                    pass
                deletions += 1
                if scope in active_scopes:
                    active_scopes.remove(scope)
                fresh = f"{workload}_scope_{next_scope_id}"
                next_scope_id += 1
                active_scopes.append(fresh)

        if i % 100 == 0:
            checkpoints.append({
                "workload": workload,
                "event": i,
                "deletions": deletions,
                "derived_puts": derived_puts,
                "db_size_bytes": db_size(cfg.db_path),
                "rows": row_count(cfg.db_path),
                "table_count": table_count(cfg.db_path),
                "active_scopes": len(active_scopes),
                "scopes_created_total": next_scope_id,
                "experiment_variant": "provenance_aware_storage_growth",
            })

    if hasattr(s, "close"):
        s.close()
    return checkpoints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="plain,static,sealed_no_rp,sqlite_envelope,kms,sage")
    ap.add_argument("--events", type=int, default=10000)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--out", default="results/storage_growth_eval.csv")
    ap.add_argument("--db_prefix", default="results/storage_growth")
    args = ap.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    os.makedirs("results", exist_ok=True)
    rows = []

    for scheme in schemes:
        for workload in WORKLOADS:
            for trial in range(args.trials):
                stem = f"{args.db_prefix}_{scheme}_{workload}_trial{trial}"
                cfg = assign_artifact_paths(
                    SchemeConfig(db_path=f"{stem}.db", scheme=scheme),
                    stem,
                )
                for p in scheme_artifact_paths(cfg, scheme, include_wal=True):
                    if isinstance(p, str) and os.path.exists(p):
                        os.remove(p)
                checkpoints = run(cfg, workload, args.events)
                for c in checkpoints:
                    rows.append({"scheme": scheme, "trial": trial, **c})
                print(
                    f"[storage_growth_eval] completed "
                    f"scheme={scheme} workload={workload} trial={trial}"
                )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
