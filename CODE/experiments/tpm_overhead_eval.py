from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Any

from experiments.schemes import SchemeConfig, make_scheme
from experiments.utils import Timer, clean

from baselines.sage_tpm import SAGE_TPM
from tpm_epoch import TPMEpochStore
from tpm_sealer import TPMSealer


def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    xs = sorted(xs)
    med = statistics.median(xs)
    q1 = statistics.median(xs[: len(xs) // 2]) if len(xs) > 1 else med
    q3 = statistics.median(xs[(len(xs) + 1) // 2 :]) if len(xs) > 1 else med
    return med, q1, q3


def payload(nbytes: int) -> dict[str, Any]:
    return {"blob": "x" * nbytes}


def safe_close(obj: Any) -> None:
    if hasattr(obj, "close"):
        try:
            obj.close()
        except Exception:
            pass


def bench_put(s: Any, scope: str, payload_bytes: int, n_ops: int) -> float:
    samples = []
    for i in range(n_ops):
        with Timer() as t:
            s.put(f"{scope}:put:{i}", payload(payload_bytes))
        samples.append(t.ms)
    return statistics.mean(samples)


def bench_get(s: Any, scope: str, k: int, n_queries: int) -> float:
    samples = []
    for _ in range(n_queries):
        with Timer() as t:
            s.get_recent(scope, limit=k)
        samples.append(t.ms)
    return statistics.mean(samples)


def bench_forget(s: Any, scope: str) -> float:
    with Timer() as t:
        s.forget_scope(scope)
    return t.ms


def make_run_paths(base_dir: str, scheme: str, run_id: str) -> dict[str, str]:
    prefix = os.path.join(base_dir, f"{scheme}_{run_id}")
    return {
        "db": f"{prefix}.db",
        "root": f"{prefix}.root.bin",
        "epochs": f"{prefix}.epochs.db",
        "epochs_tpm": f"{prefix}.epochs_tpm.json",
        "blob_prefix": f"{prefix}.tpm",
        "dev_master": f"{prefix}.dev_master.bin",
    }


def sealed_blob_artifacts(path: str) -> list[str]:
    a = TPMSealer.artifact_paths_for(path)
    return [a["pub"], a["priv"], a["primary"], a["ctx"]]


def cleanup_run_paths(paths: dict[str, str]) -> list[str]:
    return [
        paths["db"],
        f"{paths['db']}-wal",
        f"{paths['db']}-shm",
        paths["root"],
        paths["epochs"],
        f"{paths['epochs']}-wal",
        f"{paths['epochs']}-shm",
        paths["epochs_tpm"],
        f"{paths['epochs_tpm']}.pending",
        f"{paths['epochs_tpm']}.nvmeta.json",
        f"{paths['epochs_tpm']}.mackey",
        paths["dev_master"],
        *sealed_blob_artifacts(paths["root"]),
        *sealed_blob_artifacts(f"{paths['epochs_tpm']}.mackey"),
    ]


def destroy_tpm_state(paths: dict[str, str]) -> None:
    TPMEpochStore.destroy_persistent_state(
        paths["epochs_tpm"],
        blob_prefix=paths["blob_prefix"],
    )


def make_sage(paths: dict[str, str]) -> Any:
    cfg = SchemeConfig(db_path=paths["db"], scheme="sage")

    # Force software SAGE to use per-run paths.
    if hasattr(cfg, "sage_root_sealed"):
        cfg.sage_root_sealed = paths["root"]
    if hasattr(cfg, "sage_epochs"):
        cfg.sage_epochs = paths["epochs"]
    if hasattr(cfg, "sage_dev_master"):
        cfg.sage_dev_master = paths["dev_master"]

    return make_scheme(cfg)


def make_sage_tpm(paths: dict[str, str]) -> Any:
    return SAGE_TPM(
        db_path=paths["db"],
        root_key_sealed=paths["root"],
        epochs_path=paths["epochs_tpm"],
        tpm_blob_prefix=paths["blob_prefix"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tpm_overhead.csv")
    ap.add_argument("--workdir", default="results/tpm_overhead_runs")
    ap.add_argument("--scope", default="user:tpmbench")
    ap.add_argument(
        "--schemes",
        default="sage,sage_tpm_realistic,sage_tpm_naive",
        help="Options: sage, sage_tpm_realistic, sage_tpm_naive",
    )
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--put_ops", type=int, default=100)
    ap.add_argument("--get_queries", type=int, default=100)
    ap.add_argument("--populate", type=int, default=200)
    ap.add_argument("--payload_bytes", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    total_runs = args.repeats + args.warmup

    for scheme in schemes:
        print(f"\n[tpm_overhead] {scheme}", flush=True)

        put_samples: list[float] = []
        get_samples: list[float] = []
        forget_samples: list[float] = []

        for r in range(total_runs):
            run_id = f"run{r}"
            paths = make_run_paths(args.workdir, scheme, run_id)
            destroy_tpm_state(paths)
            clean(cleanup_run_paths(paths))

            if scheme == "sage":
                s = make_sage(paths)

                # Populate once for get/forget on the benchmark scope.
                for _ in range(args.populate):
                    s.put(args.scope, payload(args.payload_bytes))

                put_ms = bench_put(s, args.scope, args.payload_bytes, args.put_ops)
                get_ms = bench_get(s, args.scope, args.k, args.get_queries)
                forget_ms = bench_forget(s, args.scope)

                safe_close(s)

            elif scheme == "sage_tpm_naive":
                s = make_sage_tpm(paths)

                for _ in range(args.populate):
                    s.put(args.scope, payload(args.payload_bytes))

                put_ms = bench_put(s, args.scope, args.payload_bytes, args.put_ops)
                get_ms = bench_get(s, args.scope, args.k, args.get_queries)
                forget_ms = bench_forget(s, args.scope)

                safe_close(s)

            elif scheme == "sage_tpm_realistic":
                # Realistic TPM-backed deployment model:
                # - After service open, steady-state Put/Get use cached in-process
                #   key and epoch state, so they do not invoke the TPM directly.
                # - Forget still advances the protected epoch through the TPM.

                # Measure steady-state Put/Get on the cached-state path.
                s_soft = make_sage(paths)

                for _ in range(args.populate):
                    s_soft.put(args.scope, payload(args.payload_bytes))

                put_ms = bench_put(s_soft, args.scope, args.payload_bytes, args.put_ops)
                get_ms = bench_get(s_soft, args.scope, args.k, args.get_queries)

                safe_close(s_soft)

                # Measure Forget on the TPM-backed epoch-advance path.
                s_tpm = make_sage_tpm(paths)
                forget_ms = bench_forget(s_tpm, args.scope)
                safe_close(s_tpm)

            else:
                raise ValueError(f"Unsupported scheme: {scheme}")

            if r >= args.warmup:
                put_samples.append(put_ms)
                get_samples.append(get_ms)
                forget_samples.append(forget_ms)

            print(
                f"run {r+1}/{total_runs}  put={put_ms:.2f}  get={get_ms:.2f}  forget={forget_ms:.2f}",
                flush=True,
            )

            destroy_tpm_state(paths)
            clean(cleanup_run_paths(paths))

        put_med, put_q1, put_q3 = median_iqr(put_samples)
        get_med, get_q1, get_q3 = median_iqr(get_samples)
        forget_med, forget_q1, forget_q3 = median_iqr(forget_samples)

        rows.extend(
            [
                {"scheme": scheme, "experiment": "put_latency", "metric": "median_ms_per_put", "value": put_med},
                {"scheme": scheme, "experiment": "put_latency", "metric": "q1_ms_per_put", "value": put_q1},
                {"scheme": scheme, "experiment": "put_latency", "metric": "q3_ms_per_put", "value": put_q3},
                {"scheme": scheme, "experiment": "get_latency", "metric": "median_ms_per_get", "value": get_med},
                {"scheme": scheme, "experiment": "get_latency", "metric": "q1_ms_per_get", "value": get_q1},
                {"scheme": scheme, "experiment": "get_latency", "metric": "q3_ms_per_get", "value": get_q3},
                {"scheme": scheme, "experiment": "forget_latency", "metric": "median_ms_per_forget", "value": forget_med},
                {"scheme": scheme, "experiment": "forget_latency", "metric": "q1_ms_per_forget", "value": forget_q1},
                {"scheme": scheme, "experiment": "forget_latency", "metric": "q3_ms_per_forget", "value": forget_q3},
            ]
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scheme", "experiment", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()
