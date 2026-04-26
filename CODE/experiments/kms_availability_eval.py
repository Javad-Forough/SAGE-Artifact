"""
KMS-availability evaluation.

This experiment compares a remote-KMS design with SAGE under simulated KMS
network outages. It reports operation success rates and latency summaries for
put, get, and delete.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from experiments.schemes import SchemeConfig, make_scheme

try:
    from experiments.utils import clean
except Exception:
    def clean(paths: list[str]) -> None:
        import os
        for p in paths:
            if p and os.path.exists(p):
                os.remove(p)


# ----------------------------
# Minimal progress helper (no deps)
# ----------------------------
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
        msg = f"{self.desc} {pct:6.2f}% ({i}/{self.total})  elapsed={self._fmt(elapsed)}  eta={self._fmt(remaining)}"
        print("\r" + msg, end="", flush=True)

    def done(self) -> None:
        self.update(self.total)
        print("", flush=True)


def _now() -> float:
    return time.perf_counter()


def _sleep_ms(ms: float) -> None:
    if ms <= 0:
        return
    time.sleep(ms / 1000.0)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


@dataclass
class RTTModel:
    base_ms: float
    jitter_ms: float = 0.0
    def sample_ms(self, rng: random.Random) -> float:
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
        outage_duration_s: float,
        outage_start_after_s: float,
    ):
        self.inner = inner
        self.rng = rng
        self.rtt = rtt
        self.failure_rate = max(0.0, min(1.0, failure_rate))
        self.cache_ttl_s = max(0.0, cache_ttl_s)

        self.t0_wall = time.time()
        self.outage_start_after_s = max(0.0, outage_start_after_s)
        self.outage_duration_s = max(0.0, outage_duration_s)

        self._cache: dict[str, tuple[float, tuple[Any, Any]]] = {}
        self.calls_get = 0
        self.calls_delete = 0
        self.failures = 0

    def _in_outage(self) -> bool:
        t = time.time() - self.t0_wall
        return (t >= self.outage_start_after_s) and (t < self.outage_start_after_s + self.outage_duration_s)

    def _maybe_fail(self) -> None:
        if self._in_outage():
            self.failures += 1
            raise KMSNetworkError("KMS unreachable (simulated outage)")
        if self.failure_rate > 0 and self.rng.random() < self.failure_rate:
            self.failures += 1
            raise KMSNetworkError("KMS request failed (simulated)")
        _sleep_ms(self.rtt.sample_ms(self.rng))

    def get_key(self, scope_id: str):
        self.calls_get += 1
        if self.cache_ttl_s > 0:
            ent = self._cache.get(scope_id)
            if ent:
                exp, val = ent
                if _now() < exp:
                    return val
        self._maybe_fail()
        val = self.inner.get_key(scope_id)
        if self.cache_ttl_s > 0:
            self._cache[scope_id] = (_now() + self.cache_ttl_s, val)
        return val

    def delete_key(self, scope_id: str):
        self.calls_delete += 1
        if scope_id in self._cache:
            del self._cache[scope_id]
        self._maybe_fail()
        return self.inner.delete_key(scope_id)


def attach_kms_sim_strict(
    scheme_obj: Any,
    sim: "KMSClientSim",
) -> bool:
    """
    Strict attachment: fail fast if this is not a TrustedKMSDesign-like scheme.
    Returns True on success.
    """
    if not hasattr(scheme_obj, "kms"):
        raise RuntimeError("Scheme has no .kms attribute (expected TrustedKMSDesign)")

    kms = scheme_obj.kms
    if not hasattr(kms, "get_key") or not hasattr(kms, "delete_key"):
        raise RuntimeError("Scheme .kms does not implement the file-backed KMS API")

    scheme_obj.kms = sim
    return True


def time_call(fn) -> float:
    t0 = _now()
    fn()
    return (_now() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kms_availability_eval.csv")
    ap.add_argument("--db", default="results/kms_avail_mem.db")
    ap.add_argument("--scope", default="user:kmsavail")
    ap.add_argument("--seed", type=int, default=11)

    ap.add_argument("--schemes", default="kms,sage")
    ap.add_argument("--ops", type=int, default=5000)
    ap.add_argument("--populate", type=int, default=5000)
    ap.add_argument("--payload_bytes", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)

    ap.add_argument("--rtt_ms", type=float, default=5.0)
    ap.add_argument("--jitter_ms", type=float, default=0.5)
    ap.add_argument("--failure_rate", type=float, default=0.0)
    ap.add_argument("--cache_ttl_s", type=float, default=0.1)

    ap.add_argument("--outage_start_after_s", type=float, default=1.0)
    ap.add_argument("--outage_duration_s", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    rng = random.Random(args.seed)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    payload = {"blob": "a" * args.payload_bytes}

    rows: list[dict[str, Any]] = []

    for scheme in schemes:
        print(f"\n[kms_availability_eval] scheme={scheme}", flush=True)

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

        # populate
        print(f"[kms_availability_eval] populate={args.populate}", flush=True)
        pop_prog = Progress(args.populate, desc="  populate")
        for i in range(1, args.populate + 1):
            s.put(args.scope, {"fact": "x"})
            pop_prog.update(i)
        pop_prog.done()

        # attach sim for kms
        sim: Optional[KMSClientSim] = None
        if scheme == "kms":
            if not hasattr(s, "kms"):
                raise RuntimeError("KMS scheme selected but scheme has no .kms attribute")

            sim = KMSClientSim(
                inner=s.kms,
                rng=rng,
                rtt=RTTModel(args.rtt_ms, args.jitter_ms),
                failure_rate=args.failure_rate,
                cache_ttl_s=args.cache_ttl_s,
                outage_duration_s=args.outage_duration_s,
                outage_start_after_s=args.outage_start_after_s,
            )

            ok = attach_kms_sim_strict(s, sim)
            if not ok:
                print("[warn] could not attach outage simulator to KMS baseline. "
                      "Availability experiment will be less meaningful.", flush=True)

        # workload: interleave put/get + occasional forget
        ok_put, ok_get, ok_forget = 0, 0, 0
        fail_put, fail_get, fail_forget = 0, 0, 0
        put_lat: list[float] = []
        get_lat: list[float] = []
        forget_lat: list[float] = []

        print(
            f"[kms_availability_eval] running ops={args.ops} (outage starts at {args.outage_start_after_s}s for {args.outage_duration_s}s)",
            flush=True,
        )
        op_prog = Progress(args.ops, desc="  ops")

        for i in range(args.ops):
            # PUT
            try:
                put_lat.append(time_call(lambda: s.put(args.scope, payload)))
                ok_put += 1
            except Exception:
                fail_put += 1

            # GET
            try:
                get_lat.append(time_call(lambda: s.get_recent(args.scope, limit=args.k)))
                ok_get += 1
            except Exception:
                fail_get += 1

            # FORGET occasionally on fresh scopes
            if i % 500 == 0:
                sc = f"{args.scope}:f{i}"
                for _ in range(200):
                    try:
                        s.put(sc, {"fact": "x"})
                    except Exception:
                        pass
                try:
                    forget_lat.append(time_call(lambda: s.forget_scope(sc)))
                    ok_forget += 1
                except Exception:
                    fail_forget += 1

            op_prog.update(i + 1)

        op_prog.done()

        rows.append({
            "scheme": scheme,
            "ops": args.ops,
            "rtt_ms": args.rtt_ms,
            "jitter_ms": args.jitter_ms,
            "failure_rate": args.failure_rate,
            "cache_ttl_s": args.cache_ttl_s,
            "outage_start_after_s": args.outage_start_after_s,
            "outage_duration_s": args.outage_duration_s,

            "put_ok": ok_put, "put_fail": fail_put,
            "get_ok": ok_get, "get_fail": fail_get,
            "forget_ok": ok_forget, "forget_fail": fail_forget,

            "put_p50_ms": _pct(put_lat, 50), "put_p99_ms": _pct(put_lat, 99),
            "get_p50_ms": _pct(get_lat, 50), "get_p99_ms": _pct(get_lat, 99),
            "forget_p50_ms": _pct(forget_lat, 50), "forget_p99_ms": _pct(forget_lat, 99),

            "kms_calls_get": getattr(sim, "calls_get", ""),
            "kms_calls_delete": getattr(sim, "calls_delete", ""),
            "kms_failures": getattr(sim, "failures", ""),
        })

        if hasattr(s, "close"):
            try:
                s.close()
            except Exception:
                pass

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
