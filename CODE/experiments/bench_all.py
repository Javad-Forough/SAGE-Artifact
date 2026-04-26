# experiments/bench_all.py
from __future__ import annotations

import argparse
import csv
import os
import secrets
import statistics
import time
from typing import Any, Optional

from experiments.schemes import SchemeConfig, make_scheme
from experiments.utils import clean, Timer

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def payload(nbytes: int) -> dict[str, Any]:
    # fixed-size-ish blob
    return {"blob": secrets.token_hex(max(1, nbytes // 2))[:nbytes]}


def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    xs2 = sorted(xs)
    med = statistics.median(xs2)
    q1 = statistics.median(xs2[: len(xs2) // 2]) if len(xs2) >= 2 else med
    q3 = statistics.median(xs2[(len(xs2) + 1) // 2 :]) if len(xs2) >= 2 else med
    return med, q1, q3


# ----------------------------
# Transaction helpers (duck typing)
# ----------------------------
def _begin_txn(s: Any) -> bool:
    """
    Try to begin a transaction for batching.
    Returns True if we successfully started a txn (so caller should commit/rollback).
    """
    if hasattr(s, "begin") and callable(getattr(s, "begin")):
        s.begin()
        return True
    if hasattr(s, "conn"):
        conn = getattr(s, "conn")
        try:
            conn.execute("BEGIN")
            return True
        except Exception:
            return False
    return False


def _commit_txn(s: Any) -> None:
    if hasattr(s, "commit") and callable(getattr(s, "commit")):
        s.commit()
        return
    if hasattr(s, "conn"):
        conn = getattr(s, "conn")
        try:
            conn.commit()
        except Exception:
            pass


def _rollback_txn(s: Any) -> None:
    if hasattr(s, "rollback") and callable(getattr(s, "rollback")):
        s.rollback()
        return
    if hasattr(s, "conn"):
        conn = getattr(s, "conn")
        try:
            conn.rollback()
        except Exception:
            pass


def _close_scheme(s: Any) -> None:
    # Support different APIs across schemes
    if hasattr(s, "close") and callable(getattr(s, "close")):
        try:
            s.close()
        except Exception:
            pass


# ----------------------------
# Bench blocks
# ----------------------------
def bench_put(s: Any, scope: str, payload_bytes: int, n_ops: int, batch_size: int, progress: bool) -> float:
    it = range(n_ops)
    if progress and tqdm is not None:
        it = tqdm(it, desc=f"put n={n_ops} pb={payload_bytes}", leave=False)

    started = _begin_txn(s)
    try:
        with Timer() as t:
            for i in it:
                s.put(scope, payload(payload_bytes))
                if batch_size > 0 and (i + 1) % batch_size == 0:
                    if started:
                        _commit_txn(s)
                        _begin_txn(s)
        if started:
            _commit_txn(s)
        return t.ms / n_ops
    except Exception:
        if started:
            _rollback_txn(s)
        raise


def bench_populate(s: Any, scope: str, n_items: int, batch_size: int, progress: bool) -> None:
    it = range(n_items)
    if progress and tqdm is not None:
        it = tqdm(it, desc=f"populate n={n_items}", leave=False)

    started = _begin_txn(s)
    try:
        for i in it:
            s.put(scope, {"fact": "x"})
            if batch_size > 0 and (i + 1) % batch_size == 0:
                if started:
                    _commit_txn(s)
                    _begin_txn(s)
        if started:
            _commit_txn(s)
    except Exception:
        if started:
            _rollback_txn(s)
        raise


def bench_get(s: Any, scope: str, k: int, n_queries: int, progress: bool) -> float:
    it = range(n_queries)
    if progress and tqdm is not None:
        it = tqdm(it, desc=f"get q={n_queries} k={k}", leave=False)

    with Timer() as t:
        for _ in it:
            _ = s.get_recent(scope, limit=k)
    return t.ms / n_queries


def bench_forget(s: Any, scope: str) -> float:
    with Timer() as t:
        s.forget_scope(scope)
    return t.ms


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench_all.csv")
    ap.add_argument("--db", default="results/bench_mem.db")
    ap.add_argument("--scope", default="user:bench")
    ap.add_argument("--schemes", default="plain,static,sealed_no_rp,kms,sage")

    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)

    ap.add_argument("--put_ops", type=int, default=2000)
    ap.add_argument("--get_queries", type=int, default=2000)
    ap.add_argument("--populate", type=int, default=20000)  # ensure enough items for gets

    ap.add_argument("--payloads", default="32,256,1024,4096")
    ap.add_argument("--ks", default="1,5,10,25,50,100")
    ap.add_argument("--forget_ns", default="100,1000,5000,20000")

    # Optional execution controls.
    ap.add_argument("--progress", action="store_true", help="Show progress bars (tqdm if available).")
    ap.add_argument("--batch_size", type=int, default=1000, help="Commit every N puts during put/populate blocks (0 disables).")
    ap.add_argument("--seed_sleep", type=float, default=0.0, help="Optional delay between schemes.")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    payloads = [int(x) for x in args.payloads.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    forget_ns = [int(x) for x in args.forget_ns.split(",")]

    rows: list[dict[str, Any]] = []

    for scheme in schemes:
        log(f"\n===== Scheme: {scheme} =====")
        cfg = SchemeConfig(db_path=args.db, scheme=scheme)

        cleanup = [args.db]
        if scheme == "static":
            cleanup += [cfg.static_key_path]
        if scheme == "sealed_no_rp":
            cleanup += [cfg.sealed_no_rp_root]
        if scheme == "kms":
            cleanup += [cfg.kms_state]
        if scheme == "sage":
            cleanup += [cfg.sage_root_sealed, cfg.sage_dev_master, cfg.sage_epochs]
        clean(cleanup)

        s = make_scheme(cfg)

        # ---- PUT latency vs payload size ----
        log("[stage] put_latency")
        for pb in payloads:
            samples: list[float] = []
            total_runs = args.repeats + args.warmup
            for r in range(total_runs):
                if args.progress and tqdm is None:
                    log(f"  put pb={pb} run {r+1}/{total_runs}")
                ms = bench_put(
                    s, args.scope, pb,
                    n_ops=args.put_ops,
                    batch_size=args.batch_size,
                    progress=args.progress
                )
                if r >= args.warmup:
                    samples.append(ms)
            med, q1, q3 = median_iqr(samples)
            rows += [
                {"scheme": scheme, "experiment": "put_latency", "payload_bytes": pb,
                 "n_items": args.put_ops, "k": "", "metric": "median_ms_per_put", "value": med},
                {"scheme": scheme, "experiment": "put_latency", "payload_bytes": pb,
                 "n_items": args.put_ops, "k": "", "metric": "q1_ms_per_put", "value": q1},
                {"scheme": scheme, "experiment": "put_latency", "payload_bytes": pb,
                 "n_items": args.put_ops, "k": "", "metric": "q3_ms_per_put", "value": q3},
            ]

        # ---- Populate ----
        log("[stage] populate")
        bench_populate(s, args.scope, n_items=args.populate, batch_size=args.batch_size, progress=args.progress)

        # ---- GET latency vs k ----
        log("[stage] get_latency")
        for k in ks:
            samples = []
            total_runs = args.repeats + args.warmup
            for r in range(total_runs):
                if args.progress and tqdm is None:
                    log(f"  get k={k} run {r+1}/{total_runs}")
                ms = bench_get(s, args.scope, k=k, n_queries=args.get_queries, progress=args.progress)
                if r >= args.warmup:
                    samples.append(ms)
            med, q1, q3 = median_iqr(samples)
            rows += [
                {"scheme": scheme, "experiment": "get_latency", "payload_bytes": 256,
                 "n_items": args.get_queries, "k": k, "metric": "median_ms_per_get", "value": med},
                {"scheme": scheme, "experiment": "get_latency", "payload_bytes": 256,
                 "n_items": args.get_queries, "k": k, "metric": "q1_ms_per_get", "value": q1},
                {"scheme": scheme, "experiment": "get_latency", "payload_bytes": 256,
                 "n_items": args.get_queries, "k": k, "metric": "q3_ms_per_get", "value": q3},
            ]

        # ---- FORGET: one-shot latency ----
        log("[stage] forget_latency")
        samples = []
        total_runs = args.repeats + args.warmup
        for r in range(total_runs):
            if args.progress and tqdm is None:
                log(f"  forget one-shot run {r+1}/{total_runs}")
            ms = bench_forget(s, args.scope)
            if r >= args.warmup:
                samples.append(ms)
        med, q1, q3 = median_iqr(samples)
        rows += [
            {"scheme": scheme, "experiment": "forget_latency", "payload_bytes": "",
             "n_items": "", "k": "", "metric": "median_ms_per_forget", "value": med},
            {"scheme": scheme, "experiment": "forget_latency", "payload_bytes": "",
             "n_items": "", "k": "", "metric": "q1_ms_per_forget", "value": q1},
            {"scheme": scheme, "experiment": "forget_latency", "payload_bytes": "",
             "n_items": "", "k": "", "metric": "q3_ms_per_forget", "value": q3},
        ]

        # ---- FORGET vs N items ----
        log("[stage] forget_vs_n")
        for n_items in forget_ns:
            scopeN = f"{args.scope}:n{n_items}"
            bench_populate(s, scopeN, n_items=n_items, batch_size=args.batch_size, progress=args.progress)

            samples = []
            total_runs = args.repeats + args.warmup
            for r in range(total_runs):
                if args.progress and tqdm is None:
                    log(f"  forget_vs_n n={n_items} run {r+1}/{total_runs}")
                ms = bench_forget(s, scopeN)
                if r >= args.warmup:
                    samples.append(ms)
            med, q1, q3 = median_iqr(samples)
            rows += [
                {"scheme": scheme, "experiment": "forget_vs_n", "payload_bytes": 128,
                 "n_items": n_items, "k": "", "metric": "median_forget_ms", "value": med},
                {"scheme": scheme, "experiment": "forget_vs_n", "payload_bytes": 128,
                 "n_items": n_items, "k": "", "metric": "q1_forget_ms", "value": q1},
                {"scheme": scheme, "experiment": "forget_vs_n", "payload_bytes": 128,
                 "n_items": n_items, "k": "", "metric": "q3_forget_ms", "value": q3},
            ]

        _close_scheme(s)

        if args.seed_sleep > 0:
            time.sleep(args.seed_sleep)

    # Write CSV
    if not rows:
        raise RuntimeError("No benchmark rows collected; check schemes list.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
