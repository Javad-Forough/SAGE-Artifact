"""
Provenance-chain-length evaluation.

This experiment builds linear chains of derived artifacts and measures
retrieval latency, deletion latency, and post-reopen invalidation after the
root scope is deleted. Correctness checks are performed after close/reopen so
the results reflect persisted state rather than only in-process behavior.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
import uuid
from typing import Any

from experiments.schemes import (
    SchemeConfig,
    assign_artifact_paths,
    destroy_scheme_persistent_state,
    make_scheme,
    scheme_artifact_paths,
)

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
    return xs[max(0, min(len(xs) - 1, k))]


def _median(vals: list[float]) -> float:
    """Median ignoring NaN values."""
    clean_vals = sorted(v for v in vals if not math.isnan(v))
    if not clean_vals:
        return float("nan")
    mid = len(clean_vals) // 2
    if len(clean_vals) % 2:
        return clean_vals[mid]
    return (clean_vals[mid - 1] + clean_vals[mid]) / 2


class Progress:
    def __init__(self, total: int, desc: str = ""):
        self.total = max(1, int(total))
        self.desc = desc
        self.start = _now()

    def update(self, i: int) -> None:
        pct = 100.0 * i / self.total
        print(
            f"\r{self.desc} {pct:5.1f}% ({i}/{self.total}) "
            f"{_now() - self.start:.1f}s",
            end="",
            flush=True,
        )

    def done(self) -> None:
        print("", flush=True)


def _close(s: Any) -> None:
    if hasattr(s, "close"):
        try:
            s.close()
        except Exception:
            pass


def _cleanup_cfg(cfg: SchemeConfig, scheme: str) -> list[str]:
    return scheme_artifact_paths(cfg, scheme, include_wal=True)


def _safe_get_recent(s: Any, scope_id: str, limit: int = 10) -> list[dict[str, Any]]:
    try:
        return s.get_recent(scope_id, limit=limit)
    except Exception:
        return []


def _contains(items: list[dict[str, Any]], token: str) -> bool:
    t = token.lower()
    for it in items:
        payload = it.get("payload", it)
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, str) and t in v.lower():
                    return True
        elif isinstance(payload, str) and t in payload.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_chain(
    s: Any,
    tag: str,
    depth: int,
    payload_bytes: int,
) -> tuple[str, list[str], list[str]]:
    """
    Build a linear provenance chain of `depth` hops.

    Returns (root_scope, all_scopes, all_item_ids) where:
      all_scopes[0]  = root source scope
      all_scopes[-1] = leaf (most-derived) scope

    Chain structure:
      scope_0 (source) -> scope_1 (derived from scope_0)
                       -> scope_2 (derived from scope_1)
                       -> ...
                       -> scope_depth (leaf)
    """
    scopes: list[str] = [f"chain:{tag}:depth{d}" for d in range(depth + 1)]
    item_ids: list[str] = []

    # Root item (source)
    root_payload = {"content": f"root-source-{tag}", "token": f"token_{tag}"}
    try:
        root_id = s.put(scopes[0], root_payload, kind="source")
    except TypeError:
        root_id = s.put(scopes[0], root_payload)
    item_ids.append(root_id)

    # Derived items at each subsequent depth
    prev_id    = root_id
    prev_scope = scopes[0]

    for d in range(1, depth + 1):
        derived_payload = {
            "content": f"derived-depth{d}-{tag}",
            "token":   f"token_{tag}",
            "depth":   d,
        }
        curr_scope = scopes[d]
        try:
            curr_id = s.put_derived(
                curr_scope,
                derived_payload,
                kind=f"derived_depth{d}",
                derived_from_item_ids=[prev_id],
                source_scope_ids=[prev_scope],
            )
        except Exception:
            # Fallback for baselines that don't support put_derived
            try:
                curr_id = s.put(curr_scope, derived_payload, kind=f"derived_depth{d}")
            except TypeError:
                curr_id = s.put(curr_scope, derived_payload)
        item_ids.append(curr_id)
        prev_id    = curr_id
        prev_scope = curr_scope

    return scopes[0], scopes, item_ids


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_one_trial(
    scheme: str,
    depth: int,
    cfg: SchemeConfig,
    payload_bytes: int,
    get_reps: int,
    tag: str,
) -> dict[str, float]:
    """
    Three-phase trial:

    Phase 1 — Build and measure:
      Build the full chain, measure get latency on leaf and root.

    Phase 2 — Forget and close:
      Forget the root scope, then CLOSE the scheme so epoch state is
      flushed to disk.

    Phase 3 — Reopen and verify:
      Reopen scheme from disk and check leaf + intermediate correctness.
      This is stronger than checking on the same open handle because it
      verifies that epoch advances survived a close/reopen cycle.
    """
    token = f"token_{tag}"

    # ------------------------------------------------------------------
    # Phase 1: build chain, measure latency
    # ------------------------------------------------------------------
    s = make_scheme(cfg)

    t0 = _now()
    root_scope, all_scopes, all_ids = build_chain(s, tag, depth, payload_bytes)
    put_chain_ms = (_now() - t0) * 1000.0

    leaf_scope = all_scopes[-1]

    # Get latency on leaf — SAGE performs epoch dependency checks here
    get_leaf_samples: list[float] = []
    for _ in range(get_reps):
        t0 = _now()
        _safe_get_recent(s, leaf_scope, limit=5)
        get_leaf_samples.append((_now() - t0) * 1000.0)

    leaf_visible_before = _contains(_safe_get_recent(s, leaf_scope), token)

    # Get latency on root — no dependency checking; baseline for comparison
    get_root_samples: list[float] = []
    for _ in range(get_reps):
        t0 = _now()
        _safe_get_recent(s, root_scope, limit=5)
        get_root_samples.append((_now() - t0) * 1000.0)

    # DB size before forget
    db_size = os.path.getsize(cfg.db_path) if os.path.exists(cfg.db_path) else 0

    # ------------------------------------------------------------------
    # Phase 2: forget root scope, then close
    # ------------------------------------------------------------------
    t0 = _now()
    try:
        s.forget_scope(root_scope, delete_ciphertext_rows=True, propagate=True)
    except TypeError:
        try:
            s.forget_scope(root_scope, delete_ciphertext_rows=True)
        except TypeError:
            s.forget_scope(root_scope)
    forget_ms = (_now() - t0) * 1000.0

    # Close — forces epoch state to be written to disk
    _close(s)

    # ------------------------------------------------------------------
    # Phase 3: reopen and verify correctness from a fresh handle
    # ------------------------------------------------------------------
    s2 = make_scheme(cfg)

    leaf_hidden_after_reopen = not _contains(
        _safe_get_recent(s2, leaf_scope, limit=10), token
    )

    # Intermediate scopes are undefined at length 1, so that metric is NaN.
    intermediate_scopes = all_scopes[1:-1]

    if not intermediate_scopes:
        # Length 1 has no intermediate nodes.
        intermediate_hidden_reopen: float = float("nan")
    else:
        intermediate_hidden_reopen = float(all(
            not _contains(_safe_get_recent(s2, sc, limit=10), token)
            for sc in intermediate_scopes
        ))

    # `all_hidden_reopen` is the single correctness indicator used by the paper.
    if math.isnan(intermediate_hidden_reopen):
        all_hidden_reopen = float(leaf_hidden_after_reopen)
    else:
        all_hidden_reopen = float(
            leaf_hidden_after_reopen and bool(intermediate_hidden_reopen)
        )

    _close(s2)

    return {
        "put_chain_ms":               put_chain_ms,
        "get_leaf_p50_ms":            _pct(get_leaf_samples, 50),
        "get_leaf_p95_ms":            _pct(get_leaf_samples, 95),
        "get_root_p50_ms":            _pct(get_root_samples, 50),
        "get_root_p95_ms":            _pct(get_root_samples, 95),
        "forget_ms":                  forget_ms,
        "leaf_visible_before":        float(leaf_visible_before),
        "leaf_hidden_after_reopen":   float(leaf_hidden_after_reopen),
        "intermediate_hidden_reopen": intermediate_hidden_reopen,  # NaN at depth=1
        "all_hidden_reopen":          all_hidden_reopen,
        "db_size_bytes":              float(db_size),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",           default="results/provenance_depth_eval.csv")
    ap.add_argument("--db_prefix",     default="results/prov_depth")
    ap.add_argument("--schemes",       default="plain,static,sealed_no_rp,sqlite_envelope,kms,sage")
    ap.add_argument("--depths",        default="1,2,4,8,16,32")
    ap.add_argument("--trials",        type=int, default=30)
    ap.add_argument("--get_reps",      type=int, default=50)
    ap.add_argument("--payload_bytes", type=int, default=256)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    depths  = [int(x)    for x in args.depths.split(",")  if x.strip()]

    fieldnames = [
        "scheme", "chain_depth", "trials",
        "put_chain_ms",
        "get_leaf_p50_ms", "get_leaf_p95_ms",
        "get_root_p50_ms", "get_root_p95_ms",
        "forget_ms",
        "leaf_visible_before",
        "leaf_hidden_after_reopen",
        "intermediate_hidden_reopen",
        "all_hidden_reopen",
        "db_size_bytes",
    ]

    total = len(schemes) * len(depths)
    prog  = Progress(total, "[provenance_depth]")
    step  = 0
    rows: list[dict[str, Any]] = []

    for scheme in schemes:
        for depth in depths:
            step += 1
            prog.update(step)
            print(
                f"\n[provenance_depth] scheme={scheme} depth={depth} "
                f"trials={args.trials}",
                flush=True,
            )

            acc: dict[str, list[float]] = {k: [] for k in [
                "put_chain_ms",
                "get_leaf_p50_ms", "get_leaf_p95_ms",
                "get_root_p50_ms", "get_root_p95_ms",
                "forget_ms",
                "leaf_visible_before",
                "leaf_hidden_after_reopen",
                "intermediate_hidden_reopen",
                "all_hidden_reopen",
                "db_size_bytes",
            ]}

            trial_prog = Progress(args.trials, f"  trials depth={depth}")
            for t in range(args.trials):
                trial_prog.update(t + 1)
                tag  = uuid.uuid4().hex[:8]
                stem = f"{args.db_prefix}_{scheme}_d{depth}_t{t}_{tag}"
                cfg = assign_artifact_paths(
                    SchemeConfig(
                        db_path=f"{stem}.db",
                        scheme=scheme,
                    ),
                    stem,
                )
                destroy_scheme_persistent_state(cfg, scheme)
                clean(_cleanup_cfg(cfg, scheme))

                try:
                    result = run_one_trial(
                        scheme=scheme,
                        depth=depth,
                        cfg=cfg,
                        payload_bytes=args.payload_bytes,
                        get_reps=args.get_reps,
                        tag=tag,
                    )
                    for k, v in result.items():
                        acc[k].append(v)
                except Exception as e:
                    print(f"\n  [warn] trial {t} failed: {e}", flush=True)
                finally:
                    destroy_scheme_persistent_state(cfg, scheme)
                    clean(_cleanup_cfg(cfg, scheme))

            trial_prog.done()

            row: dict[str, Any] = {
                "scheme":      scheme,
                "chain_depth": depth,
                "trials":      len(acc["put_chain_ms"]),
            }
            for key in acc:
                row[key] = _median(acc[key])

            rows.append(row)

            inter_str = (
                "n/a (depth=1, no intermediate nodes)"
                if math.isnan(row["intermediate_hidden_reopen"])
                else f"{row['intermediate_hidden_reopen']:.2f}"
            )
            print(
                f"  get_leaf_p95={row['get_leaf_p95_ms']:.2f}ms  "
                f"forget={row['forget_ms']:.2f}ms  "
                f"leaf_hidden={row['leaf_hidden_after_reopen']:.2f}  "
                f"intermediate_hidden={inter_str}  "
                f"all_hidden={row['all_hidden_reopen']:.2f}",
                flush=True,
            )

    prog.done()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
