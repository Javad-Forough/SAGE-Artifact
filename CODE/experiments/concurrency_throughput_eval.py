from __future__ import annotations

import argparse
import csv
import os
import random
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any
import statistics

from experiments.schemes import SchemeConfig, make_scheme

try:
    from experiments.utils import clean
except Exception:
    def clean(paths: list[str]) -> None:
        for p in paths:
            if p and os.path.exists(p):
                os.remove(p)


AGENTS = ["assistant", "chat", "research"]


def _now() -> float:
    return time.perf_counter()


class Progress:
    def __init__(self, total: int, desc: str = "", every: int | None = None):
        self.total = max(1, int(total))
        self.desc = desc
        self.start = _now()
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
        now_t = _now()
        if now_t - self.last_print < 0.2 and i != self.total:
            return
        self.last_print = now_t

        elapsed = now_t - self.start
        rate = i / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - i) / rate if rate > 0 else float("inf")
        pct = 100.0 * i / self.total

        msg = (
            f"{self.desc} {pct:6.2f}% ({i}/{self.total}) "
            f"elapsed={self._fmt(elapsed)} eta={self._fmt(remaining)}"
        )
        print("\r" + msg, end="", flush=True)

    def done(self) -> None:
        self.update(self.total)
        print("", flush=True)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


@dataclass
class WorkerResult:
    ops_done: int
    ok: int
    fail: int
    lat_ms_p50: float
    lat_ms_p99: float


def agent_profile(agent: str) -> dict[str, Any]:
    if agent == "chat":
        return {
            "get_prob": 0.75,
            "put_prob": 0.25,
            "recent_limit": 8,
        }

    if agent == "assistant":
        return {
            "get_prob": 0.55,
            "put_prob": 0.45,
            "recent_limit": 5,
        }

    return {
        "get_prob": 0.35,
        "put_prob": 0.65,
        "recent_limit": 4,
    }


def make_payload(agent: str, seq: int, scope: str) -> dict[str, Any]:
    if agent == "chat":
        return {
            "kind": "chat_memory",
            "scope": scope,
            "turn_id": seq,
            "speaker": "user" if seq % 2 == 0 else "assistant",
            "summary": (
                f"Conversation memory {seq}: scheduling preferences, reminders, "
                f"tone, and short personal context."
            ),
            "message": (
                f"Turn {seq}: user discussed preferences, timing constraints, "
                f"and a small profile detail."
            ),
            "entities": ["meeting", "preference", "reminder", scope],
            "ts": seq,
        }

    if agent == "assistant":
        return {
            "kind": "assistant_memory",
            "scope": scope,
            "task_id": f"{scope}-task-{seq}",
            "tool_output": {
                "tool": random.choice(["calendar", "email", "tasks", "docs"]),
                "status": "ok",
                "result": (
                    f"Assistant workflow step {seq}: updated task metadata, "
                    f"follow-up reminder state, and tool output."
                ),
            },
            "action_items": [
                f"send reminder for task {seq}",
                f"update attendee list {seq}",
                f"record follow-up note {seq}",
            ],
            "priority": random.choice(["low", "medium", "high"]),
            "ts": seq,
        }

    return {
        "kind": "research_memory",
        "scope": scope,
        "topic": f"{scope}-topic-{seq}",
        "title": f"Research synthesis note {seq}",
        "abstract_summary": (
            f"Intermediate research note {seq} with extracted evidence, "
            f"comparison points, unresolved questions, and candidate citations."
        ),
        "evidence": [
            f"finding-{seq}-A: robustness improved",
            f"finding-{seq}-B: baseline degraded on long contexts",
            f"finding-{seq}-C: follow-up evaluation required",
        ],
        "notes": (
            "Long-form research artifact containing synthesis and retrieved "
            "evidence. Larger than assistant/chat payloads."
        ),
        "ts": seq,
    }


def choose_op(rng: random.Random, agent: str) -> str:
    p = agent_profile(agent)
    r = rng.random()
    return "get" if r < p["get_prob"] else "put"


def trial_cfg(base_db_prefix: str, scheme: str, agent: str, workers: int, trial: int) -> SchemeConfig:
    stem = f"{base_db_prefix}_{scheme}_{agent}_w{workers}_t{trial}"
    return SchemeConfig(
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


def cleanup_paths_for_cfg(cfg: SchemeConfig, scheme: str) -> list[str]:
    paths = [cfg.db_path]

    if scheme == "static":
        paths += [cfg.static_key_path]

    if scheme == "sealed_no_rp":
        paths += [cfg.sealed_no_rp_root, cfg.sealed_no_rp_master]

    if scheme == "kms":
        paths += [cfg.kms_state]

    if scheme == "sage":
        paths += [cfg.sage_root_sealed, cfg.sage_dev_master, cfg.sage_epochs]

    if scheme == "sqlite_envelope":
        paths += [cfg.sqlite_envelope_keys]

    return paths


def preinitialize_scheme(cfg: SchemeConfig, agent: str, seed: int) -> None:
    rng = random.Random(seed)
    s = make_scheme(cfg)
    scope = f"warmup:{agent}:init"
    try:
        for i in range(8):
            s.put(scope, make_payload(agent, i, scope))
        _ = s.get_recent(scope, limit=3)
    finally:
        if hasattr(s, "close"):
            try:
                s.close()
            except Exception:
                pass


def worker_main(
    cfg: SchemeConfig,
    agent: str,
    scope_prefix: str,
    duration_s: float,
    seed: int,
    q: mp.Queue,
    barrier: mp.Barrier,
):
    rng = random.Random(seed)
    s = make_scheme(cfg)

    profile = agent_profile(agent)
    scope = f"{scope_prefix}:{agent}:{mp.current_process().name}:{seed}"

    try:
        for i in range(50):
            payload = make_payload(agent, i, scope)
            s.put(scope, payload)
        _ = s.get_recent(scope, limit=profile["recent_limit"])
    except Exception:
        pass

    barrier.wait()

    lat: list[float] = []
    ok = fail = 0
    ops = 0

    t_end = _now() + duration_s
    seq = 1000

    while _now() < t_end:
        op = choose_op(rng, agent)
        t0 = _now()

        try:
            if op == "put":
                payload = make_payload(agent, seq, scope)
                s.put(scope, payload)
                seq += 1
            else:
                _ = s.get_recent(scope, limit=profile["recent_limit"])

            ok += 1
        except Exception:
            fail += 1

        lat.append((_now() - t0) * 1000.0)
        ops += 1

    if hasattr(s, "close"):
        try:
            s.close()
        except Exception:
            pass

    q.put(
        WorkerResult(
            ops_done=ops,
            ok=ok,
            fail=fail,
            lat_ms_p50=_pct(lat, 50),
            lat_ms_p99=_pct(lat, 99),
        )
    )


def parse_csv_arg(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", default="results/concurrency_throughput.csv")
    ap.add_argument("--db_prefix", default="results/concurrency_mem")
    ap.add_argument("--scope", default="user:conc")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument(
        "--schemes",
        default="plain,static,sealed_no_rp,sqlite_envelope,kms,sage",
    )
    ap.add_argument("--agents", default="assistant,chat,research")
    ap.add_argument("--workers", default="1,2,4,6,8,10,12")

    ap.add_argument("--duration_s", type=float, default=15.0)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--heartbeat_s", type=float, default=1.0)

    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    schemes = parse_csv_arg(args.schemes)
    agents = parse_csv_arg(args.agents)
    workers_list = [int(x) for x in parse_csv_arg(args.workers)]

    print(f"[concurrency] schemes: {schemes}", flush=True)
    print(f"[concurrency] agents: {agents}", flush=True)
    print(f"[concurrency] workers: {workers_list}", flush=True)

    rows: list[dict[str, Any]] = []

    total_steps = len(schemes) * len(agents) * len(workers_list)
    overall = Progress(total_steps, desc="[concurrency] configs", every=1)

    step = 0

    for scheme in schemes:
        for agent in agents:
            print(f"\n[concurrency] scheme={scheme} agent={agent}", flush=True)

            for w in workers_list:
                step += 1
                overall.update(step)

                print(
                    f"\n[concurrency] scheme={scheme} agent={agent} "
                    f"workers={w} duration_s={args.duration_s} trials={args.trials}",
                    flush=True,
                )

                trial_throughputs: list[float] = []
                trial_p50: list[float] = []
                trial_p99: list[float] = []
                trial_fail_rates: list[float] = []

                for trial in range(args.trials):
                    print(f"[concurrency] trial {trial+1}/{args.trials}", flush=True)

                    cfg = trial_cfg(args.db_prefix, scheme, agent, w, trial)
                    clean(cleanup_paths_for_cfg(cfg, scheme))

                    preinitialize_scheme(cfg, agent, args.seed + trial)

                    q: mp.Queue = mp.Queue()
                    barrier = mp.Barrier(w + 1)
                    procs: list[mp.Process] = []

                    for i in range(w):
                        p = mp.Process(
                            target=worker_main,
                            args=(
                                cfg,
                                agent,
                                args.scope,
                                args.duration_s,
                                args.seed + i + trial * 1000,
                                q,
                                barrier,
                            ),
                        )
                        p.start()
                        procs.append(p)

                    barrier.wait()
                    t0 = _now()

                    results: list[WorkerResult] = []
                    next_hb = _now() + max(0.1, args.heartbeat_s)
                    max_wait_s = args.duration_s + 30

                    while len(results) < w:
                        try:
                            results.append(q.get(timeout=0.2))
                        except Exception:
                            pass

                        now_t = _now()
                        if now_t >= next_hb:
                            alive = sum(1 for p in procs if p.is_alive())
                            elapsed = now_t - t0
                            remaining = max(0.0, args.duration_s - elapsed)

                            print(
                                f"[concurrency] running... alive={alive}/{w} "
                                f"elapsed={elapsed:0.1f}s approx_remaining={remaining:0.1f}s",
                                flush=True,
                            )
                            next_hb = now_t + max(0.1, args.heartbeat_s)

                        if (now_t - t0) > max_wait_s:
                            raise RuntimeError(
                                f"Timed out waiting for worker results (got {len(results)}/{w})"
                            )

                    for p in procs:
                        p.join()

                    elapsed = _now() - t0

                    total_ok = sum(r.ok for r in results)
                    total_fail = sum(r.fail for r in results)
                    total_ops = sum(r.ops_done for r in results)

                    throughput = total_ok / max(1e-9, elapsed)
                    fail_rate = total_fail / max(1, total_ops)

                    lat_p50 = float(sum(r.lat_ms_p50 for r in results) / len(results))
                    lat_p99 = float(sum(r.lat_ms_p99 for r in results) / len(results))

                    trial_throughputs.append(throughput)
                    trial_p50.append(lat_p50)
                    trial_p99.append(lat_p99)
                    trial_fail_rates.append(fail_rate)

                    print(
                        f"[trial {trial+1}] throughput={throughput:.1f} ops/s "
                        f"p50={lat_p50:.2f}ms p99={lat_p99:.2f}ms fail_rate={fail_rate:.3f}",
                        flush=True,
                    )

                    # Clean up trial files immediately after results are collected.
                    # Also removes SQLite WAL/SHM sidecars produced during the run.
                    trial_paths = cleanup_paths_for_cfg(cfg, scheme)
                    sidecar_paths = [p + ext for p in trial_paths for ext in ("-wal", "-shm")]
                    clean(trial_paths + sidecar_paths)

                median_thr = statistics.median(trial_throughputs)
                median_p50 = statistics.median(trial_p50)
                median_p99 = statistics.median(trial_p99)
                median_fail = statistics.median(trial_fail_rates)

                rows.append(
                    {
                        "scheme": scheme,
                        "agent": agent,
                        "workers": w,
                        "duration_s": args.duration_s,
                        "trials": args.trials,
                        "throughput_ops_per_s": median_thr,
                        "lat_ms_p50": median_p50,
                        "lat_ms_p99": median_p99,
                        "fail_rate": median_fail,
                    }
                )

                print(
                    f"[{scheme}:{agent}] workers={w} "
                    f"median_throughput={median_thr:.1f} ops/s "
                    f"median_p50={median_p50:.2f}ms "
                    f"median_p99={median_p99:.2f}ms "
                    f"median_fail_rate={median_fail:.3f}",
                    flush=True,
                )

    overall.done()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote: {args.out}", flush=True)

    # Final sweep: remove any leftover temp files whose names start with the db_prefix stem.
    # This catches files left behind by interrupted trials or unexpected sidecars.
    db_dir = os.path.dirname(args.db_prefix) or "."
    db_stem = os.path.basename(args.db_prefix)
    leftover: list[str] = []
    try:
        for fname in os.listdir(db_dir):
            if fname.startswith(db_stem + "_"):
                leftover.append(os.path.join(db_dir, fname))
    except OSError:
        pass
    if leftover:
        clean(leftover)
        print(f"[concurrency] cleaned up {len(leftover)} leftover temp file(s).", flush=True)


if __name__ == "__main__":
    main()
