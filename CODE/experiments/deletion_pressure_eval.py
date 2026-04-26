"""
Deletion-pressure evaluation.

This experiment measures foreground put/get latency and throughput while a
background thread issues scope deletions at a controlled rate.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from experiments.schemes import SchemeConfig, make_scheme

try:
    from experiments.utils import clean
except Exception:
    def clean(paths: list[str]) -> None:
        for p in paths:
            if p and os.path.exists(p):
                os.remove(p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.perf_counter()


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


class Progress:
    def __init__(self, total: int, desc: str = ""):
        self.total = max(1, int(total))
        self.desc = desc
        self.start = _now()

    def update(self, i: int) -> None:
        pct = 100.0 * i / self.total
        elapsed = _now() - self.start
        print(f"\r{self.desc} {pct:5.1f}% ({i}/{self.total}) {elapsed:.1f}s",
              end="", flush=True)

    def done(self) -> None:
        print("", flush=True)


def _make_payload(nbytes: int) -> dict[str, Any]:
    return {"blob": "a" * nbytes}


def _close(s: Any) -> None:
    if hasattr(s, "close"):
        try:
            s.close()
        except Exception:
            pass


def _cleanup_cfg(cfg: SchemeConfig, scheme: str) -> list[str]:
    paths = [cfg.db_path]
    if scheme == "static":
        paths.append(cfg.static_key_path)
    if scheme == "sealed_no_rp":
        paths.append(cfg.sealed_no_rp_root)
    if scheme == "kms":
        paths.append(cfg.kms_state)
    if scheme == "sqlite_envelope":
        paths.append(cfg.sqlite_envelope_keys)
    if scheme == "sage":
        paths += [cfg.sage_root_sealed, cfg.sage_epochs, cfg.sage_dev_master]
    return paths


# ---------------------------------------------------------------------------
# Shared state for the experiment threads
# ---------------------------------------------------------------------------

@dataclass
class WorkerStats:
    put_lat:    list[float] = field(default_factory=list)
    get_lat:    list[float] = field(default_factory=list)
    ops_done:   int = 0
    lock:       threading.Lock = field(default_factory=threading.Lock)


@dataclass
class DeleterStats:
    forget_ops:  int = 0
    forget_fail: int = 0
    lock:        threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Background deleter thread
# ---------------------------------------------------------------------------

def _deleter_thread(
    scheme_factory,          # callable () -> SchemeHandle
    scope_pool: list[str],   # pool of scopes to delete from
    forget_rate_hz: float,   # target deletions per second (0 = no deletions)
    stop_event: threading.Event,
    stats: DeleterStats,
    payload: dict[str, Any],
    seed: int,
) -> None:
    """
    Runs in a background thread. Opens its own scheme handle so there is no
    lock contention on the foreground handle, then deletes scopes at the
    requested rate by cycling through scope_pool.
    """
    if forget_rate_hz <= 0:
        return

    rng = random.Random(seed)
    s = scheme_factory()
    interval_s = 1.0 / forget_rate_hz
    idx = 0

    try:
        while not stop_event.is_set():
            t0 = _now()
            scope = scope_pool[idx % len(scope_pool)]
            idx += 1

            # Re-seed the scope with a few items so forget is non-trivial
            try:
                for _ in range(10):
                    s.put(scope, payload)
            except Exception:
                pass

            try:
                s.forget_scope(scope)
                with stats.lock:
                    stats.forget_ops += 1
            except Exception:
                with stats.lock:
                    stats.forget_fail += 1

            # Re-populate after deletion so the scope is usable again
            try:
                for _ in range(5):
                    s.put(scope, payload)
            except Exception:
                pass

            elapsed = _now() - t0
            sleep_time = interval_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        _close(s)


# ---------------------------------------------------------------------------
# Foreground worker threads
# ---------------------------------------------------------------------------

def _worker_thread(
    scheme_factory,
    scope_pool: list[str],
    duration_s: float,
    stop_event: threading.Event,
    stats: WorkerStats,
    payload: dict[str, Any],
    k: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    s = scheme_factory()

    # Warm up
    scope = rng.choice(scope_pool)
    try:
        for _ in range(50):
            s.put(scope, payload)
    except Exception:
        pass

    t_end = _now() + duration_s
    try:
        while _now() < t_end and not stop_event.is_set():
            scope = rng.choice(scope_pool)

            t0 = _now()
            try:
                s.put(scope, payload)
                put_ms = (_now() - t0) * 1000.0
            except Exception:
                put_ms = float("nan")

            t0 = _now()
            try:
                s.get_recent(scope, limit=k)
                get_ms = (_now() - t0) * 1000.0
            except Exception:
                get_ms = float("nan")

            with stats.lock:
                if put_ms == put_ms:   # nan check
                    stats.put_lat.append(put_ms)
                if get_ms == get_ms:
                    stats.get_lat.append(get_ms)
                stats.ops_done += 1
    finally:
        _close(s)


# ---------------------------------------------------------------------------
# One (scheme, forget_rate) configuration
# ---------------------------------------------------------------------------

def run_one_config(
    scheme: str,
    forget_rate_hz: float,
    db_prefix: str,
    num_scopes: int,
    workers: int,
    duration_s: float,
    payload_bytes: int,
    k: int,
    seed: int,
) -> dict[str, Any]:

    stem = f"{db_prefix}_{scheme}_fr{forget_rate_hz:.0f}"
    cfg = SchemeConfig(
        db_path=f"{stem}.db",
        scheme=scheme,
        sage_root_sealed=f"{stem}.sage_root.bin",
        sage_epochs=f"{stem}.sage_epochs.json",
        sage_dev_master=f"{stem}.sage_dev_master.bin",
        static_key_path=f"{stem}.static_key.bin",
        sealed_no_rp_root=f"{stem}.sealed_no_rp_root.bin",
        kms_state=f"{stem}.kms_state.json",
        sqlite_envelope_keys=f"{stem}.sqlite_envelope_keys.json",
    )
    clean(_cleanup_cfg(cfg, scheme))

    payload = _make_payload(payload_bytes)

    # Pre-populate all scopes so gets are non-trivial from the start
    s_init = make_scheme(cfg)
    scope_pool = [f"scope:{i}" for i in range(num_scopes)]
    for sc in scope_pool:
        for _ in range(20):
            s_init.put(sc, payload)
    _close(s_init)

    def scheme_factory():
        return make_scheme(cfg)

    # Split scope pool: foreground workers use even indices, deleter uses odd
    fg_scopes  = scope_pool[::2]  or scope_pool
    del_scopes = scope_pool[1::2] or scope_pool

    stop_event = threading.Event()
    w_stats = WorkerStats()
    d_stats = DeleterStats()

    threads: list[threading.Thread] = []

    # Deleter thread (single, background)
    if forget_rate_hz > 0:
        dt = threading.Thread(
            target=_deleter_thread,
            args=(scheme_factory, del_scopes, forget_rate_hz,
                  stop_event, d_stats, payload, seed + 9999),
            daemon=True,
        )
        threads.append(dt)

    # Foreground worker threads
    for i in range(workers):
        wt = threading.Thread(
            target=_worker_thread,
            args=(scheme_factory, fg_scopes, duration_s,
                  stop_event, w_stats, payload, k, seed + i),
            daemon=True,
        )
        threads.append(wt)

    t0 = _now()
    for t in threads:
        t.start()

    # Wait for foreground duration, then signal stop
    time.sleep(duration_s + 1.0)
    stop_event.set()
    for t in threads:
        t.join(timeout=10.0)

    actual_duration = _now() - t0

    put_lat = w_stats.put_lat
    get_lat = w_stats.get_lat
    total_ops = w_stats.ops_done

    return {
        "scheme":           scheme,
        "forget_rate_hz":   forget_rate_hz,
        "workers":          workers,
        "duration_s":       actual_duration,
        "forget_ops":       d_stats.forget_ops,
        "forget_fail":      d_stats.forget_fail,
        "put_p50_ms":       _pct(put_lat, 50),
        "put_p95_ms":       _pct(put_lat, 95),
        "put_p99_ms":       _pct(put_lat, 99),
        "get_p50_ms":       _pct(get_lat, 50),
        "get_p95_ms":       _pct(get_lat, 95),
        "get_p99_ms":       _pct(get_lat, 99),
        "throughput_ops_s": total_ops / max(1e-9, actual_duration),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",          default="results/deletion_pressure_eval.csv")
    ap.add_argument("--db_prefix",    default="results/del_pressure")
    ap.add_argument("--schemes",      default="plain,static,sealed_no_rp,sqlite_envelope,kms,sage")
    ap.add_argument("--forget_rates", default="0,1,5,10,20,50",
                    help="Forget operations per second to test")
    ap.add_argument("--workers",      type=int,   default=4)
    ap.add_argument("--duration_s",   type=float, default=15.0)
    ap.add_argument("--num_scopes",   type=int,   default=200)
    ap.add_argument("--payload_bytes",type=int,   default=256)
    ap.add_argument("--k",            type=int,   default=10)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--trials",       type=int,   default=10,
                    help="Repeat each config N times and take the median")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    schemes      = [s.strip() for s in args.schemes.split(",")      if s.strip()]
    forget_rates = [float(x)  for x in args.forget_rates.split(",") if x.strip()]

    total = len(schemes) * len(forget_rates) * args.trials
    prog  = Progress(total, "[deletion_pressure]")
    step  = 0
    rows: list[dict[str, Any]] = []

    for scheme in schemes:
        for fr in forget_rates:
            trial_rows: list[dict[str, Any]] = []
            for trial in range(args.trials):
                step += 1
                prog.update(step)
                print(
                    f"\n[deletion_pressure] scheme={scheme} forget_rate={fr}hz "
                    f"trial={trial+1}/{args.trials}",
                    flush=True,
                )
                r = run_one_config(
                    scheme=scheme,
                    forget_rate_hz=fr,
                    db_prefix=f"{args.db_prefix}_trial{trial}",
                    num_scopes=args.num_scopes,
                    workers=args.workers,
                    duration_s=args.duration_s,
                    payload_bytes=args.payload_bytes,
                    k=args.k,
                    seed=args.seed + trial * 1000,
                )
                trial_rows.append(r)
                print(
                    f"  put_p95={r['put_p95_ms']:.2f}ms  "
                    f"get_p95={r['get_p95_ms']:.2f}ms  "
                    f"thr={r['throughput_ops_s']:.1f}ops/s  "
                    f"forget_ops={r['forget_ops']}",
                    flush=True,
                )

            # Median across trials
            def med(key: str) -> float:
                vals = sorted(tr[key] for tr in trial_rows
                              if isinstance(tr[key], float) and tr[key] == tr[key])
                if not vals:
                    return float("nan")
                mid = len(vals) // 2
                return vals[mid] if len(vals) % 2 else (vals[mid-1] + vals[mid]) / 2

            rows.append({
                "scheme":           scheme,
                "forget_rate_hz":   fr,
                "workers":          args.workers,
                "trials":           args.trials,
                "put_p50_ms":       med("put_p50_ms"),
                "put_p95_ms":       med("put_p95_ms"),
                "put_p99_ms":       med("put_p99_ms"),
                "get_p50_ms":       med("get_p50_ms"),
                "get_p95_ms":       med("get_p95_ms"),
                "get_p99_ms":       med("get_p99_ms"),
                "throughput_ops_s": med("throughput_ops_s"),
                "forget_ops":       sum(tr["forget_ops"]  for tr in trial_rows),
                "forget_fail":      sum(tr["forget_fail"] for tr in trial_rows),
            })

    prog.done()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
