# experiments/kms_latency_eval.py
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from experiments.schemes import SchemeConfig, make_scheme
from experiments.utils import clean


class Progress:
    def __init__(self, total: int, desc: str = "", every: int | None = None):
        self.total = max(1, int(total))
        self.desc = desc
        self.start = time.perf_counter()
        self.last_print = 0.0
        self.every = every if every is not None else max(1, self.total // 100)

    def _fmt(self, secs: float) -> str:
        if secs < 60:
            return f"{secs:0.1f}s"
        if secs < 3600:
            return f"{secs/60:0.1f}m"
        return f"{secs/3600:0.2f}h"

    def update(self, i: int) -> None:
        if i % self.every != 0 and i != self.total:
            return
        now_t = time.perf_counter()
        if now_t - self.last_print < 0.2 and i != self.total:
            return
        self.last_print = now_t
        elapsed = now_t - self.start
        rate = i / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - i) / rate if rate > 0 else float("inf")
        pct = 100.0 * i / self.total
        print(
            f"\r{self.desc} {pct:6.2f}% ({i}/{self.total})"
            f"  elapsed={self._fmt(elapsed)}  eta={self._fmt(remaining)}",
            end="",
            flush=True,
        )

    def done(self) -> None:
        self.update(self.total)
        print("", flush=True)


def now() -> float:
    return time.perf_counter()


def sleep_ms(ms: float) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = int((p / 100.0) * (len(xs) - 1))
    return xs[max(0, min(len(xs) - 1, i))]


def batch_populate(s: Any, scope_id: str, n: int, batch: int = 500) -> None:
    """Populate scope with n items.

    store.put() commits after every insert.  To batch those commits we
    temporarily replace conn.commit with a no-op so that Python's sqlite3
    keeps the transaction open, then flush every `batch` inserts.
    """
    if n <= 0:
        return

    store = getattr(getattr(s, "mem", None), "store", None)
    conn = getattr(store, "_conn", None)  # SQLiteMemoryStore uses _conn

    if conn is not None:
        orig_commit = conn.commit
        conn.commit = lambda: None  # suppress per-put commits
        try:
            for i in range(n):
                s.put(scope_id, {"fact": "x"})
                if (i + 1) % batch == 0:
                    orig_commit()
            orig_commit()
        finally:
            conn.commit = orig_commit
    else:
        for i in range(n):
            s.put(scope_id, {"fact": "x"})


@dataclass
class RTTModel:
    base_ms: float
    jitter_ms: float = 0.0

    def sample(self, rng: random.Random) -> float:
        if self.jitter_ms <= 0:
            return self.base_ms
        return self.base_ms + rng.uniform(-self.jitter_ms, self.jitter_ms)


class KMSNetworkError(RuntimeError):
    pass


class KMSClientSim:
    def __init__(
        self,
        inner: Any,
        rng: random.Random,
        rtt: RTTModel,
        failure_rate: float,
        cache_ttl_s: float,
    ):
        self.inner = inner
        self.rng = rng
        self.rtt = rtt
        self.failure_rate = failure_rate
        self.cache_ttl_s = cache_ttl_s
        self.cache: dict[str, tuple[float, tuple[int, bytes]]] = {}
        self.calls_get = 0
        self.calls_delete = 0
        self.failures = 0

    def _delay_and_maybe_fail(self) -> None:
        sleep_ms(self.rtt.sample(self.rng))
        if self.failure_rate > 0 and self.rng.random() < self.failure_rate:
            self.failures += 1
            raise KMSNetworkError("simulated KMS failure")

    def get_key(self, scope_id: str):
        self.calls_get += 1
        if self.cache_ttl_s > 0:
            ent = self.cache.get(scope_id)
            if ent and now() < ent[0]:
                return ent[1]
        self._delay_and_maybe_fail()
        val = self.inner.get_key(scope_id)
        if self.cache_ttl_s > 0:
            self.cache[scope_id] = (now() + self.cache_ttl_s, val)
        return val

    def delete_key(self, scope_id: str):
        self.calls_delete += 1
        self.cache.pop(scope_id, None)
        self._delay_and_maybe_fail()
        return self.inner.delete_key(scope_id)

    def forget(self, scope_id: str):
        return self.delete_key(scope_id)


def attach_kms_sim_strict(scheme_obj: Any, sim: KMSClientSim) -> None:
    if not hasattr(scheme_obj, "kms"):
        raise RuntimeError("scheme has no .kms attribute")
    kms = scheme_obj.kms
    if not hasattr(kms, "get_key") or not hasattr(kms, "delete_key"):
        raise RuntimeError("scheme.kms does not implement KMS API")
    scheme_obj.kms = sim


def time_call(fn) -> float:
    t0 = now()
    fn()
    return (now() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kms_latency_eval.csv")
    ap.add_argument("--db", default="results/kms_latency.db")
    ap.add_argument("--scope", default="user:kmslat")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--schemes", default="kms,sage")
    ap.add_argument("--payload_bytes", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)

    ap.add_argument("--ops", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--populate", type=int, default=5000)

    # Forget-scope sizing. The default measures forget on an empty scope,
    # matching the microbenchmark configuration. Increase this value to include
    # populated-scope paths together with --forget_populate > 0.
    ap.add_argument(
        "--forget_populate",
        type=int,
        default=0,
        help="Items pre-populated per forget scope before timing forget().",
    )
    ap.add_argument(
        "--forget_reps",
        type=int,
        default=5,
        help="Number of forget scope repetitions per config.",
    )
    ap.add_argument("--forget_batch", type=int, default=500,
                    help="Commit batch size during forget-scope pre-population.")

    ap.add_argument("--rtts_ms", default="0,1,5,10,25")
    ap.add_argument("--jitter_ms", type=float, default=0.2)
    ap.add_argument("--failure_rate", type=float, default=0.0)
    ap.add_argument("--cache_ttls", default="0,0.01,0.1,1.0")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = random.Random(args.seed)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    rtts = [float(x) for x in args.rtts_ms.split(",") if x.strip()]
    ttls = [float(x) for x in args.cache_ttls.split(",") if x.strip()]

    payload = {"blob": "a" * args.payload_bytes}
    rows: list[dict[str, Any]] = []

    total_cfgs = len(schemes) * len(rtts) * len(ttls)
    cfg_prog = Progress(total_cfgs, desc="[kms_latency_eval] configs")

    cfg_idx = 0
    for scheme in schemes:
        for rtt in rtts:
            for ttl in ttls:
                cfg_idx += 1
                cfg_prog.update(cfg_idx)
                print(
                    f"\n[kms_latency_eval] cfg {cfg_idx}/{total_cfgs}:"
                    f" scheme={scheme} rtt_ms={rtt} ttl_s={ttl}",
                    flush=True,
                )

                cfg = SchemeConfig(db_path=args.db, scheme=scheme)

                cleanup = [args.db]
                if scheme == "kms":
                    cleanup.append(cfg.kms_state)
                if scheme == "sage":
                    cleanup += [cfg.sage_root_sealed, cfg.sage_epochs, cfg.sage_dev_master]
                clean(cleanup)

                s = make_scheme(cfg)

                # --- Pre-populate forget scopes BEFORE attaching KMS sim ---
                # This ensures forget-scope puts are not slowed by simulated RTT,
                # matching the microbench setup where populate happens outside
                # the timed paths. Only the forget() call itself is timed.
                forget_scope_ids = [f"{args.scope}:f{i}" for i in range(args.forget_reps)]
                pre_pop_total = args.forget_reps * args.forget_populate
                pre_pop_prog = Progress(pre_pop_total, desc="  pre-pop forget scopes")
                pre_pop_done = 0
                for sc in forget_scope_ids:
                    batch_populate(s, sc, args.forget_populate, batch=args.forget_batch)
                    pre_pop_done += args.forget_populate
                    pre_pop_prog.update(pre_pop_done)
                pre_pop_prog.done()

                # --- Attach KMS sim (affects all subsequent puts/gets/forgets) ---
                kms_sim = None
                if scheme == "kms":
                    kms_sim = KMSClientSim(
                        inner=s.kms,
                        rng=rng,
                        rtt=RTTModel(rtt, args.jitter_ms),
                        failure_rate=args.failure_rate,
                        cache_ttl_s=ttl,
                    )
                    attach_kms_sim_strict(s, kms_sim)

                # Main populate
                pop_prog = Progress(args.populate, desc="  populate")
                for i in range(1, args.populate + 1):
                    s.put(args.scope, {"fact": "x"})
                    pop_prog.update(i)
                pop_prog.done()

                # Warmup
                warm_prog = Progress(args.warmup, desc="  warmup")
                for i in range(1, args.warmup + 1):
                    s.put(args.scope, payload)
                    _ = s.get_recent(args.scope, limit=args.k)
                    warm_prog.update(i)
                warm_prog.done()

                put_lat, get_lat, forget_lat = [], [], []

                ops_prog = Progress(args.ops, desc="  put ops")
                for i in range(1, args.ops + 1):
                    put_lat.append(time_call(lambda: s.put(args.scope, payload)))
                    ops_prog.update(i)
                ops_prog.done()

                ops_prog = Progress(args.ops, desc="  get ops")
                for i in range(1, args.ops + 1):
                    get_lat.append(time_call(lambda: s.get_recent(args.scope, limit=args.k)))
                    ops_prog.update(i)
                ops_prog.done()

                # Forget timing — scopes already pre-populated above
                forget_prog = Progress(args.forget_reps, desc="  forget scopes")
                for i, sc in enumerate(forget_scope_ids):
                    sc_fixed = sc  # capture loop variable
                    forget_lat.append(time_call(lambda: s.forget_scope(sc_fixed)))
                    forget_prog.update(i + 1)
                forget_prog.done()

                def add(metric, xs):
                    rows.append({
                        "scheme": scheme,
                        "rtt_ms": rtt,
                        "cache_ttl_s": ttl,
                        "metric": metric,
                        "p50_ms": pct(xs, 50),
                        "p95_ms": pct(xs, 95),
                        "p99_ms": pct(xs, 99),
                        "n": len(xs),
                        "kms_gets": getattr(kms_sim, "calls_get", ""),
                        "kms_deletes": getattr(kms_sim, "calls_delete", ""),
                        "kms_failures": getattr(kms_sim, "failures", ""),
                    })

                add("put", put_lat)
                add("get", get_lat)
                add("forget", forget_lat)

                if hasattr(s, "close"):
                    try:
                        s.close()
                    except Exception:
                        pass

    cfg_prog.done()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
