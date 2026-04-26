from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from typing import Any

from experiments.schemes import SchemeConfig, make_scheme

try:
    from experiments.utils import clean
except Exception:
    def clean(paths: list[str]) -> None:
        for p in paths:
            if p and os.path.exists(p):
                os.remove(p)


# ----------------------------
# Progress helper
# ----------------------------
class Progress:
    def __init__(self, total: int, desc: str = ""):
        self.total = max(1, int(total))
        self.desc = desc
        self.start = time.perf_counter()

    def update(self, i: int) -> None:
        pct = 100.0 * i / self.total
        elapsed = time.perf_counter() - self.start
        print(
            f"\r{self.desc} {pct:6.2f}% ({i}/{self.total}) elapsed={elapsed:0.1f}s",
            end="",
            flush=True,
        )

    def done(self) -> None:
        print("", flush=True)


# ----------------------------
# Stats helpers
# ----------------------------
def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


def summarize_latencies_ms(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {
            "mean_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
        }
    return {
        "mean_ms": statistics.mean(xs),
        "p50_ms": pct(xs, 50),
        "p95_ms": pct(xs, 95),
        "p99_ms": pct(xs, 99),
    }


# ----------------------------
# Payloads
# ----------------------------
def make_payload(seq: int, scope_id: str) -> dict[str, Any]:
    return {
        "kind": "fact",
        "scope": scope_id,
        "fact": f"memory item {seq} for {scope_id}",
        "summary": f"Persistent fact {seq}",
        "ts": seq,
    }


# ----------------------------
# Scheme cleanup
# ----------------------------
def cleanup_paths(cfg: SchemeConfig, scheme: str) -> list[str]:
    paths = [cfg.db_path]

    if scheme == "static":
        paths += [cfg.static_key_path]

    if scheme == "sealed_no_rp":
        paths += [cfg.sealed_no_rp_root]

    if scheme == "kms":
        paths += [cfg.kms_state]

    if scheme == "sage":
        paths += [cfg.sage_root_sealed, cfg.sage_epochs, cfg.sage_dev_master]

    if scheme == "sqlite_envelope":
        if hasattr(cfg, "sqlite_envelope_keys"):
            paths += [cfg.sqlite_envelope_keys]

    return paths


# ----------------------------
# Core experiment
# ----------------------------
def prepopulate_scopes(s, scope_prefix: str, num_scopes: int) -> list[str]:
    scopes: list[str] = []
    prog = Progress(num_scopes, f"  [populate {num_scopes} scopes]")

    for i in range(num_scopes):
        scope = f"{scope_prefix}:scope{i}"
        s.put(scope, make_payload(i, scope))
        scopes.append(scope)
        prog.update(i + 1)

    prog.done()
    return scopes


def run_one_setting(
    scheme: str,
    cfg: SchemeConfig,
    num_scopes: int,
    scope_prefix: str,
    measure_ops: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)

    s = make_scheme(cfg)

    active_scopes = prepopulate_scopes(s, scope_prefix, num_scopes)

    put_lat_ms: list[float] = []
    get_lat_ms: list[float] = []
    forget_lat_ms: list[float] = []

    prog = Progress(measure_ops, f"  [measure {scheme} scopes={num_scopes}]")

    next_scope_id = num_scopes

    for i in range(measure_ops):
        target = rng.choice(active_scopes)

        # PUT on random active scope
        t0 = time.perf_counter()
        s.put(target, make_payload(1000000 + i, target))
        put_lat_ms.append((time.perf_counter() - t0) * 1000.0)

        # GET on random active scope
        t0 = time.perf_counter()
        _ = s.get_recent(target, limit=1)
        get_lat_ms.append((time.perf_counter() - t0) * 1000.0)

        # FORGET on random active scope, then replace it with a fresh scope
        # so the number of live scopes stays approximately constant.
        target = rng.choice(active_scopes)

        t0 = time.perf_counter()
        s.forget_scope(target)
        forget_lat_ms.append((time.perf_counter() - t0) * 1000.0)

        active_scopes.remove(target)

        new_scope = f"{scope_prefix}:scope{next_scope_id}"
        next_scope_id += 1
        s.put(new_scope, make_payload(2000000 + i, new_scope))
        active_scopes.append(new_scope)

        prog.update(i + 1)

    prog.done()

    if hasattr(s, "close"):
        s.close()

    return {
        "put": summarize_latencies_ms(put_lat_ms),
        "get": summarize_latencies_ms(get_lat_ms),
        "forget": summarize_latencies_ms(forget_lat_ms),
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", default="results/scope_scaling_eval.csv")
    ap.add_argument("--db_prefix", default="results/scope_scaling")
    ap.add_argument("--schemes", default="plain,static,sealed_no_rp,kms,sage")
    ap.add_argument("--scope_sizes", default="10,100,1000,10000")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--measure_ops", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    scope_sizes = [int(x) for x in args.scope_sizes.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []

    total = len(schemes) * len(scope_sizes) * args.trials
    overall = Progress(total, "[scope_scaling]")
    step = 0

    for scheme in schemes:
        for num_scopes in scope_sizes:
            for trial in range(args.trials):
                step += 1
                overall.update(step)

                stem = f"{args.db_prefix}_{scheme}_{num_scopes}_trial{trial}"

                cfg = SchemeConfig(
                    db_path=f"{stem}.db",
                    scheme=scheme,
                    sage_root_sealed=f"{stem}.sage_root.bin",
                    sage_epochs=f"{stem}.sage_epochs.json",
                    sage_dev_master=f"{stem}.sage_dev_master.bin",
                    static_key_path=f"{stem}.static_key.bin",
                    sealed_no_rp_root=f"{stem}.sealed_no_rp_root.bin",
                    kms_state=f"{stem}.kms_state.json",
                )

                clean(cleanup_paths(cfg, scheme))

                stats = run_one_setting(
                    scheme=scheme,
                    cfg=cfg,
                    num_scopes=num_scopes,
                    scope_prefix=f"user:multiscope:{scheme}:trial{trial}",
                    measure_ops=args.measure_ops,
                    seed=args.seed + trial,
                )

                for op_name, vals in stats.items():
                    rows.append(
                        {
                            "scheme": scheme,
                            "num_scopes": num_scopes,
                            "trial": trial,
                            "op": op_name,
                            **vals,
                        }
                    )

                print(
                    f"[{scheme}] scopes={num_scopes} trial={trial} "
                    f"put_p50={stats['put']['p50_ms']:.2f} "
                    f"get_p50={stats['get']['p50_ms']:.2f} "
                    f"forget_p50={stats['forget']['p50_ms']:.2f}",
                    flush=True,
                )

    overall.done()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()