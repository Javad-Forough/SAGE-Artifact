"""
Restart-recovery evaluation.

This experiment measures restart latency and checks whether a deleted scope
remains inaccessible after a clean reopen and after rollback-at-restart.
The restart snapshot is taken only after the database has been closed so the
copied SQLite image is fully flushed and self-consistent.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
import uuid
from typing import Any

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


def _safe_get_recent(s: Any, scope_id: str, limit: int = 20) -> list[dict[str, Any]]:
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


def _snapshot_paths(paths: list[str], suffix: str = ".snap") -> dict[str, str]:
    """
    Copy each existing path to path+suffix.
    Returns {original_path: snapshot_path}.
    Only snapshots files that actually exist on disk.
    Also copies SQLite WAL and SHM sidecar files if present,
    to guarantee a consistent snapshot of WAL-mode databases.
    """
    snaps: dict[str, str] = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        dst = p + suffix
        shutil.copy2(p, dst)
        snaps[p] = dst
        # Copy WAL sidecars so the snapshot is self-consistent
        for sidecar in [p + "-wal", p + "-shm"]:
            if os.path.exists(sidecar):
                shutil.copy2(sidecar, sidecar + suffix)
    return snaps


def _restore_snapshots(snaps: dict[str, str], suffix: str = ".snap") -> None:
    for orig, snap in snaps.items():
        if os.path.exists(snap):
            shutil.copy2(snap, orig)
        # Restore WAL sidecars
        for sidecar_snap in [orig + "-wal" + suffix, orig + "-shm" + suffix]:
            sidecar_orig = sidecar_snap[: -len(suffix)]
            if os.path.exists(sidecar_snap):
                shutil.copy2(sidecar_snap, sidecar_orig)
            elif os.path.exists(sidecar_orig):
                # Snapshot had no WAL; remove any stale WAL so DB opens clean
                os.remove(sidecar_orig)


def _cleanup_snapshots(snaps: dict[str, str], suffix: str = ".snap") -> None:
    for snap in snaps.values():
        for f in [snap, snap.replace(suffix, "-wal" + suffix),
                  snap.replace(suffix, "-shm" + suffix)]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def run_one_trial(
    scheme: str,
    cfg: SchemeConfig,
    payload_bytes: int,
    n_items: int,
) -> dict[str, Any]:

    token_live   = f"live_{uuid.uuid4().hex[:8]}"
    token_doomed = f"doomed_{uuid.uuid4().hex[:8]}"
    scope_live   = f"restart:live:{token_live}"
    scope_doomed = f"restart:doomed:{token_doomed}"
    payload      = {"content": "x" * payload_bytes}

    snap_suffix = ".pre_forget"

    # ------------------------------------------------------------------
    # Phase 1a: populate both scopes
    # ------------------------------------------------------------------
    s = make_scheme(cfg)

    for _ in range(n_items):
        try:
            s.put(scope_live,   {**payload, "token": token_live})
        except Exception:
            pass

    for _ in range(n_items):
        try:
            s.put(scope_doomed, {**payload, "token": token_doomed})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 1b: CLOSE before snapshot to flush WAL to disk
    # ------------------------------------------------------------------
    # SQLite WAL state must be flushed before snapshotting so the copied
    # database image contains the full pre-deletion state.
    _close(s)

    # ------------------------------------------------------------------
    # Phase 1c: snapshot flushed DB state
    # ------------------------------------------------------------------
    all_paths = _cleanup_cfg(cfg, scheme)

    # Database snapshot
    db_paths_to_snap = [cfg.db_path]
    db_snap = _snapshot_paths(db_paths_to_snap, suffix=snap_suffix)

    # Snapshot rollback-restorable key state for schemes that keep it in files.
    # SAGE and KMS use trusted key state that is not restored from the storage
    # snapshot in this experiment.
    key_snaps: dict[str, str] = {}
    key_paths = rollback_restorable_aux_state_paths(cfg, scheme)
    if key_paths:
        key_snaps = _snapshot_paths(key_paths, suffix=snap_suffix)

    # ------------------------------------------------------------------
    # Phase 1d: reopen, forget doomed, write more live items, close
    # ------------------------------------------------------------------
    s = make_scheme(cfg)

    try:
        s.forget_scope(scope_doomed, delete_ciphertext_rows=True, propagate=True)
    except TypeError:
        try:
            s.forget_scope(scope_doomed, delete_ciphertext_rows=True)
        except TypeError:
            s.forget_scope(scope_doomed)

    for _ in range(5):
        try:
            s.put(scope_live, {**payload, "token": token_live})
        except Exception:
            pass

    items_before_shutdown = len(_safe_get_recent(s, scope_live, limit=500))
    _close(s)

    # ------------------------------------------------------------------
    # Phase 2: clean restart (no rollback)
    # ------------------------------------------------------------------
    t0 = _now()
    s2 = make_scheme(cfg)
    open_ms = (_now() - t0) * 1000.0

    t0 = _now()
    try:
        s2.put(scope_live, {**payload, "token": token_live})
        first_put_ms = (_now() - t0) * 1000.0
    except Exception:
        first_put_ms = float("nan")

    t0 = _now()
    items_after_restart = _safe_get_recent(s2, scope_live, limit=500)
    first_get_ms = (_now() - t0) * 1000.0

    live_accessible = _contains(items_after_restart, token_live)

    forgotten_stays_forgotten = not _contains(
        _safe_get_recent(s2, scope_doomed, limit=20), token_doomed
    )

    _close(s2)

    # ------------------------------------------------------------------
    # Phase 3: rollback-at-restart attack
    # ------------------------------------------------------------------
    # Restore the pre-forget snapshot (which contains the doomed data).
    # Baseline schemes will expose the doomed data: rollback_defeated=0.0.
    # SAGE/KMS defeat the attack:                  rollback_defeated=1.0.
    _restore_snapshots(db_snap,  suffix=snap_suffix)
    if key_snaps:
        _restore_snapshots(key_snaps, suffix=snap_suffix)

    s3 = make_scheme(cfg)

    rollback_defeated = not _contains(
        _safe_get_recent(s3, scope_doomed, limit=20), token_doomed
    )

    live_after_rollback = _contains(
        _safe_get_recent(s3, scope_live, limit=500), token_live
    )

    _close(s3)

    _cleanup_snapshots(db_snap,  suffix=snap_suffix)
    _cleanup_snapshots(key_snaps, suffix=snap_suffix)

    return {
        "open_ms":                   open_ms,
        "first_put_ms":              first_put_ms,
        "first_get_ms":              first_get_ms,
        "items_before_shutdown":     items_before_shutdown,
        "items_after_restart":       len(items_after_restart),
        "live_accessible":           float(live_accessible),
        "forgotten_stays_forgotten": float(forgotten_stays_forgotten),
        "rollback_defeated":         float(rollback_defeated),
        "live_after_rollback":       float(live_after_rollback),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",           default="results/restart_recovery_eval.csv")
    ap.add_argument("--db_prefix",     default="results/restart_recovery")
    ap.add_argument("--schemes",       default="plain,static,sealed_no_rp,sqlite_envelope,kms,sage")
    ap.add_argument("--trials",        type=int, default=50)
    ap.add_argument("--n_items",       type=int, default=200)
    ap.add_argument("--payload_bytes", type=int, default=256)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []

    total = len(schemes) * args.trials
    prog  = Progress(total, "[restart_recovery]")
    step  = 0

    for scheme in schemes:
        print(f"\n[restart_recovery] scheme={scheme} trials={args.trials}", flush=True)

        open_ms_list:             list[float] = []
        first_put_ms_list:        list[float] = []
        first_get_ms_list:        list[float] = []
        forgotten_stays_list:     list[float] = []
        rollback_defeated_list:   list[float] = []
        live_accessible_list:     list[float] = []
        live_after_rollback_list: list[float] = []

        for t in range(args.trials):
            step += 1
            prog.update(step)

            tag  = uuid.uuid4().hex[:8]
            stem = f"{args.db_prefix}_{scheme}_t{t}_{tag}"
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
                r = run_one_trial(
                    scheme=scheme,
                    cfg=cfg,
                    payload_bytes=args.payload_bytes,
                    n_items=args.n_items,
                )
                open_ms_list.append(r["open_ms"])
                first_put_ms_list.append(r["first_put_ms"])
                first_get_ms_list.append(r["first_get_ms"])
                forgotten_stays_list.append(r["forgotten_stays_forgotten"])
                rollback_defeated_list.append(r["rollback_defeated"])
                live_accessible_list.append(r["live_accessible"])
                live_after_rollback_list.append(r["live_after_rollback"])
            except Exception as e:
                print(f"\n  [warn] trial {t} failed: {e}", flush=True)
            finally:
                destroy_scheme_persistent_state(cfg, scheme)
                clean(_cleanup_cfg(cfg, scheme))
                # Clean any leftover snapshot files
                for ext in [".pre_forget", "-wal.pre_forget", "-shm.pre_forget",
                            ".snap", "-wal.snap", "-shm.snap"]:
                    for p in [f"{path}{ext}" for path in _cleanup_cfg(cfg, scheme)]:
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass

        n = len(open_ms_list)
        if n == 0:
            print(f"\n  [warn] all trials failed for scheme={scheme}", flush=True)
            continue

        rows.append({
            "scheme":                        scheme,
            "trials":                        n,
            "open_p50_ms":                   _pct(open_ms_list,       50),
            "open_p95_ms":                   _pct(open_ms_list,       95),
            "first_put_p50_ms":              _pct(first_put_ms_list,  50),
            "first_put_p95_ms":              _pct(first_put_ms_list,  95),
            "first_get_p50_ms":              _pct(first_get_ms_list,  50),
            "first_get_p95_ms":              _pct(first_get_ms_list,  95),
            "forgotten_stays_forgotten":     sum(forgotten_stays_list)     / n,
            "rollback_defeated":             sum(rollback_defeated_list)   / n,
            "live_accessible_after_restart": sum(live_accessible_list)     / n,
            "live_after_rollback":           sum(live_after_rollback_list) / n,
        })

        print(
            f"\n  open_p50={rows[-1]['open_p50_ms']:.2f}ms  "
            f"open_p95={rows[-1]['open_p95_ms']:.2f}ms  "
            f"forgotten_stays={rows[-1]['forgotten_stays_forgotten']:.2f}  "
            f"rollback_defeated={rows[-1]['rollback_defeated']:.2f}  "
            f"live_after_rollback={rows[-1]['live_after_rollback']:.2f}",
            flush=True,
        )

    prog.done()

    if not rows:
        print("[error] no results collected.", flush=True)
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
