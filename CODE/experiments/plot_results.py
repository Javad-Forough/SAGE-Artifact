from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Plot styling
# ----------------------------
def _style():
    plt.rcParams.update({
        "figure.figsize": (6.6, 4.2),
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "lines.linewidth": 2.8,
        "lines.markersize": 7.5,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "0.2",
        "legend.framealpha": 0.80,
        "legend.facecolor": "white",
    })


def _grid(ax=None):
    if ax is None:
        plt.grid(True, which="both", linestyle="--", linewidth=0.9, alpha=0.45)
    else:
        ax.grid(True, which="both", linestyle="--", linewidth=0.9, alpha=0.45)


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ----------------------------
# Helpers
# ----------------------------
def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _median_only(df: pd.DataFrame) -> pd.DataFrame:
    if "metric" in df.columns:
        med = df[df["metric"].astype(str).str.contains("median", na=False)].copy()
        if not med.empty:
            return med
    return df.copy()


def _coerce_num(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _n_unique(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    s = pd.to_numeric(df[col], errors="coerce")
    return int(s.dropna().nunique())


def _pretty(x: float) -> str:
    if x < 1:
        return f"{x:.3f}"
    if x < 10:
        return f"{x:.2f}"
    return f"{x:.1f}"


def _format_df_numbers_as_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        col = pd.to_numeric(out[c], errors="coerce")
        if col.notna().any():
            out[c] = col.map(lambda v: _pretty(float(v)) if pd.notna(v) else "")
        else:
            out[c] = out[c].astype(str)
    return out


def _parse_plot_tags(spec: str) -> Set[str]:
    tags = {x.strip() for x in spec.split(",") if x.strip()}
    if not tags:
        tags = {"all"}
    return tags


def _want(tags: Set[str], name: str) -> bool:
    return "all" in tags or name in tags


def _scheme_label(s: str) -> str:
    mapping = {
        "sealed_no_rp": "sealed-no-rp",
        "sqlite_envelope": "sqlite-envelope",
    }
    return mapping.get(str(s), str(s))


def _metric_display_name(metric: str) -> str:
    mapping = {
        "forget": "delete",
    }
    return mapping.get(str(metric), str(metric))


# Consistent ordering across figures
SCHEME_ORDER = [
    "plain",
    "static",
    "sealed_no_rp",
    "sqlite_envelope",
    "kms",
    "sage",
]


SCHEME_COLORS = {
    "plain":           "#4878CF",   # steel blue
    "static":          "#D65F00",   # burnt orange
    "sealed_no_rp":    "#B47CC7",   # muted purple
    "sqlite_envelope": "#C44E52",   # muted red
    "kms":             "#956CB4",   # medium purple
    "sage":            "#2CA02C",   # green
}

SCHEME_MARKERS = {
    "plain":           "o",
    "static":          "s",
    "sealed_no_rp":    "^",
    "sqlite_envelope": "D",
    "kms":             "v",
    "sage":            "*",
}


def _scheme_color(s: str) -> str:
    return SCHEME_COLORS.get(str(s), None)


def _scheme_marker(s: str) -> str:
    return SCHEME_MARKERS.get(str(s), "o")


def _sort_schemes(vals: list[str]) -> list[str]:
    order = {k: i for i, k in enumerate(SCHEME_ORDER)}
    return sorted(vals, key=lambda x: (order.get(str(x), 999), str(x)))


@dataclass
class PlotRule:
    experiment: str
    xcol: str
    y_label: str
    x_label: str
    x_log: bool = False
    y_log: bool = False
    outname: str = ""


def plot_line_if_dense(df: pd.DataFrame, outdir: str, rule: PlotRule, min_points: int) -> bool:
    d = df[df["experiment"] == rule.experiment].copy()
    if d.empty:
        return False

    d = _median_only(d)
    d = _coerce_num(d, rule.xcol)
    d = _coerce_num(d, "value")

    npts = _n_unique(d, rule.xcol)
    if npts < min_points:
        return False

    plt.figure()
    for scheme in _sort_schemes(d["scheme"].astype(str).unique().tolist()):
        g = d[d["scheme"].astype(str) == scheme].copy()
        g = g.dropna(subset=[rule.xcol, "value"]).sort_values(rule.xcol)
        if g.empty:
            continue
        plt.plot(
            g[rule.xcol],
            g["value"],
            marker=_scheme_marker(scheme),
            label=_scheme_label(scheme),
            color=_scheme_color(scheme),
        )

    if rule.x_log:
        plt.xscale("log")
    if rule.y_log:
        plt.yscale("log")

    plt.xlabel(rule.x_label)
    plt.ylabel(rule.y_label)
    _grid()
    plt.legend()
    savefig(os.path.join(outdir, rule.outname))
    return True


def print_compact_table(df: pd.DataFrame, experiment: str, xcol: Optional[str] = None) -> None:
    d = df[df["experiment"] == experiment].copy()
    if d.empty:
        return

    d = _median_only(d)
    d = _coerce_num(d, "value")

    if xcol is not None and xcol in d.columns:
        d = _coerce_num(d, xcol)
        d = d.dropna(subset=[xcol, "value"])
        d = d.sort_values(["scheme", xcol])
        pivot = d.pivot_table(index="scheme", columns=xcol, values="value", aggfunc="first")

        print(f"\n[table] {experiment}  (columns = {xcol})")
        print(_format_df_numbers_as_strings(pivot).to_string())
    else:
        d = d.dropna(subset=["scheme", "value"]).sort_values("scheme")
        pivot = d.pivot_table(index="scheme", values="value", aggfunc="first")
        print(f"\n[table] {experiment}")
        print(_format_df_numbers_as_strings(pivot).to_string())


# ----------------------------
# Heatmaps for kms_latency_eval
# ----------------------------
def _heatmap_ticks(vals: list[float], max_ticks: int = 8) -> list[float]:
    vals = sorted({float(v) for v in vals if pd.notna(v)})
    if len(vals) <= max_ticks:
        return vals
    idxs = np.linspace(0, len(vals) - 1, num=max_ticks).round().astype(int)
    return [vals[i] for i in sorted(set(idxs.tolist()))]


def plot_kms_latency_eval_heatmaps(
    df: pd.DataFrame,
    outdir: str,
    percentile_col: str = "p95_ms",
    max_tick_labels: int = 8,
) -> None:
    needed = {"scheme", "rtt_ms", "cache_ttl_s", "metric", percentile_col}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] kms_latency_eval missing columns {sorted(list(needed - set(df.columns)))}; skipping heatmaps.")
        return

    d = df.copy()
    for c in ["rtt_ms", "cache_ttl_s", percentile_col]:
        d = _coerce_num(d, c)
    d = d.dropna(subset=["scheme", "rtt_ms", "cache_ttl_s", "metric", percentile_col])
    if d.empty:
        print("\n[warn] kms_latency_eval has no usable rows after cleaning; skipping heatmaps.")
        return

    schemes = _sort_schemes(d["scheme"].astype(str).unique().tolist())
    metrics = sorted(d["metric"].astype(str).unique().tolist())

    for metric in metrics:
        dm = d[d["metric"].astype(str) == metric].copy()
        if dm.empty:
            continue
        metric_label = _metric_display_name(metric)

        vmin = float(dm[percentile_col].min())
        vmax = float(dm[percentile_col].max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            continue
        if vmin == vmax:
            vmax = vmin + 1e-9

        rtts_all = sorted({float(x) for x in dm["rtt_ms"].dropna().unique().tolist()})
        ttls_all = sorted({float(x) for x in dm["cache_ttl_s"].dropna().unique().tolist()})

        rtt_ticks = _heatmap_ticks(rtts_all, max_ticks=max_tick_labels)
        ttl_ticks = _heatmap_ticks(ttls_all, max_ticks=max_tick_labels)

        n = len(schemes)
        fig, axes = plt.subplots(
            1, n,
            figsize=(5.6 * n, 4.6),
            constrained_layout=True,
            squeeze=False
        )
        axes = axes[0]

        last_im = None
        for j, scheme in enumerate(schemes):
            ax = axes[j]
            ds = dm[dm["scheme"].astype(str) == scheme].copy()
            if ds.empty:
                ax.set_axis_off()
                ax.set_title(f"{scheme} (no data)")
                continue

            pivot = ds.pivot_table(
                index="cache_ttl_s",
                columns="rtt_ms",
                values=percentile_col,
                aggfunc="mean",
            ).reindex(index=ttls_all, columns=rtts_all)

            Z = pivot.to_numpy(dtype=float)

            last_im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )

            rtt_pos = [rtts_all.index(x) for x in rtt_ticks if x in rtts_all]
            ttl_pos = [ttls_all.index(x) for x in ttl_ticks if x in ttls_all]

            ax.set_xticks(rtt_pos)
            ax.set_xticklabels([str(int(x)) if float(x).is_integer() else f"{x:g}" for x in rtt_ticks])

            ax.set_yticks(ttl_pos)
            ax.set_yticklabels([f"{x:g}" for x in ttl_ticks])

            ax.set_xlabel("RTT (ms)")
            if j == 0:
                ax.set_ylabel("cache TTL (s)")
            ax.set_title(_scheme_label(scheme))

            ax.set_xticks(np.arange(-.5, len(rtts_all), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(ttls_all), 1), minor=True)
            ax.grid(which="minor", linestyle="-", linewidth=0.25, alpha=0.20)
            ax.tick_params(which="minor", bottom=False, left=False)

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.035, pad=0.02)
            cbar.set_label(f"{metric_label} {percentile_col.replace('_', ' ')}")

        outpath = os.path.join(outdir, f"kms_latency_heatmap_{metric_label}_{percentile_col}.png")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_kms_latency_eval_lines(df: pd.DataFrame, outdir: str, percentile_col: str = "p95_ms") -> None:
    needed = {"scheme", "rtt_ms", "cache_ttl_s", "metric", percentile_col}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] kms_latency_eval missing columns {sorted(list(needed - set(df.columns)))}; skipping plot.")
        return

    d = df.copy()
    for c in ["rtt_ms", "cache_ttl_s", percentile_col]:
        d = _coerce_num(d, c)

    for metric in sorted(d["metric"].astype(str).unique()):
        dm = d[d["metric"].astype(str) == metric].copy()
        if dm.empty:
            continue
        metric_label = _metric_display_name(metric)

        plt.figure()
        for (scheme, ttl), g in dm.groupby(["scheme", "cache_ttl_s"]):
            g = g.dropna(subset=["rtt_ms", percentile_col]).sort_values("rtt_ms")
            if g.empty:
                continue
            plt.plot(
                g["rtt_ms"],
                g[percentile_col],
                marker=_scheme_marker(str(scheme)),
                label=f"{_scheme_label(scheme)} ttl={ttl:g}s",
                color=_scheme_color(scheme),
            )

        plt.xlabel("RTT (ms)")
        plt.ylabel(f"{metric_label} latency ({percentile_col.replace('_', ' ')})")
        _grid()
        plt.legend()
        outpath = os.path.join(outdir, f"kms_latency_eval_{metric_label}_{percentile_col}.png")
        savefig(outpath)


# ----------------------------
# Concurrency: main paper figures only
# ----------------------------
def plot_concurrency_throughput(df: pd.DataFrame, outdir: str) -> None:
    # Accept both column name variants
    needed_a = {"scheme", "workers", "throughput_ops_per_s"}
    needed_b = {"scheme", "workers", "throughput_ops_s"}
    if not (needed_a.issubset(set(df.columns)) or
            needed_b.issubset(set(df.columns))):
        print(
            f"\n[warn] concurrency_throughput missing throughput column; "
            f"skipping plot."
        )
        return

    d = df.copy()
    # normalise column name
    if "throughput_ops_per_s" not in d.columns and             "throughput_ops_s" in d.columns:
        d = d.rename(columns={"throughput_ops_s": "throughput_ops_per_s"})

    d = _coerce_num(d, "workers")
    d = _coerce_num(d, "throughput_ops_per_s")
    d = d.dropna(subset=["scheme", "workers", "throughput_ops_per_s"])
    if d.empty:
        return

    # scale by success rate so failed ops don't inflate throughput
    if "fail_rate" in d.columns:
        d = _coerce_num(d, "fail_rate")
        d["throughput_ops_per_s"] = d["throughput_ops_per_s"] * (
            1.0 - d["fail_rate"].fillna(0.0).clip(0.0, 1.0)
        )

    # aggregate across trials (median)
    group_cols = ["scheme", "workers"]
    if "agent" in d.columns and d["agent"].notna().any():
        group_cols = ["scheme", "agent", "workers"]
    d = d.groupby(group_cols, as_index=False)["throughput_ops_per_s"].median()

    def _line_kwargs_conc(scheme: str) -> dict:
        is_sage = scheme == "sage"
        return dict(
            marker=_scheme_marker(scheme),
            linewidth=3.5 if is_sage else 1.8,
            markersize=12 if is_sage else 7,
            zorder=5 if is_sage else 2,
            color=_scheme_color(scheme),
            label=_scheme_label(scheme),
        )

    if "agent" in d.columns and d["agent"].notna().any():
        agents = sorted(d["agent"].dropna().astype(str).unique().tolist())

        for agent in agents:
            da = d[d["agent"].astype(str) == agent].copy()
            schemes = _sort_schemes(
                da["scheme"].astype(str).unique().tolist())
            workers_all = sorted(da["workers"].dropna().unique().tolist())

            plt.figure(figsize=(6.8, 4.4))

            for scheme in schemes:
                g = da[da["scheme"].astype(str) == scheme].copy()
                g = g.sort_values("workers")
                if g.empty:
                    continue
                plt.plot(
                    g["workers"],
                    g["throughput_ops_per_s"],
                    **_line_kwargs_conc(scheme),
                )

            plt.xlabel("Workers")
            plt.ylabel("Successful throughput (ops/s)")
            plt.xticks(workers_all)
            _grid()
            plt.legend(
                loc="lower right",
                framealpha=0.88,
                facecolor="white",
                edgecolor="0.2",
            )
            plt.margins(x=0.05)
            out = os.path.join(outdir, f"concurrency_throughput_{agent}.png")
            savefig(out)
            print(f"[plot] wrote {out}")
        return

    # no agent column — single figure
    workers_all = sorted(d["workers"].dropna().unique().tolist())
    plt.figure(figsize=(6.6, 4.2))
    for scheme in _sort_schemes(d["scheme"].astype(str).unique().tolist()):
        g = d[d["scheme"].astype(str) == scheme].copy()
        g = g.sort_values("workers")
        if g.empty:
            continue
        plt.plot(
            g["workers"],
            g["throughput_ops_per_s"],
            **_line_kwargs_conc(scheme),
        )

    plt.xlabel("Workers")
    plt.ylabel("Successful throughput (ops/s)")
    plt.xticks(workers_all)
    _grid()
    plt.legend(
        loc="lower right",
        framealpha=0.88,
        facecolor="white",
        edgecolor="0.2",
    )
    plt.margins(x=0.05)
    out = os.path.join(outdir, "concurrency_throughput.png")
    savefig(out)
    print(f"[plot] wrote {out}")


def print_concurrency_tables(df: pd.DataFrame) -> None:
    needed = {"scheme", "workers", "throughput_ops_per_s"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] concurrency_throughput missing columns {sorted(list(needed - set(df.columns)))}; skipping tables.")
        return

    d = df.copy()
    for c in ["workers", "throughput_ops_per_s", "lat_ms_p50", "lat_ms_p99"]:
        d = _coerce_num(d, c)

    if "agent" in d.columns and d["agent"].notna().any():
        for metric in ["throughput_ops_per_s", "lat_ms_p50", "lat_ms_p99"]:
            if metric not in d.columns:
                continue
            for agent in sorted(d["agent"].dropna().astype(str).unique()):
                da = d[d["agent"].astype(str) == agent].copy()
                pivot = da.pivot_table(index="scheme", columns="workers", values=metric, aggfunc="median")
                print(f"\n[table] concurrency ({agent}) {metric}")
                print(_format_df_numbers_as_strings(pivot).to_string())
    else:
        for metric in ["throughput_ops_per_s", "lat_ms_p50", "lat_ms_p99"]:
            if metric not in d.columns:
                continue
            pivot = d.pivot_table(index="scheme", columns="workers", values=metric, aggfunc="median")
            print(f"\n[table] concurrency {metric}")
            print(_format_df_numbers_as_strings(pivot).to_string())


# ----------------------------
# Other tables
# ----------------------------
def print_agent_correctness_table(df: pd.DataFrame) -> None:
    needed = {"scheme", "acc_before_forget", "acc_after_forget"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] agent_correctness missing columns {sorted(list(needed - set(df.columns)))}; skipping table.")
        return

    d = df.copy()
    for c in d.columns:
        if c.startswith("acc_") or c.endswith("_recovery"):
            d = _coerce_num(d, c)

    cols = [c for c in ["acc_before_forget", "acc_after_forget", "rollback_recovery", "acc_after_rollback"] if c in d.columns]
    idx_cols = ["scheme"]
    if "agent" in d.columns:
        idx_cols = ["agent", "scheme"]

    out = d[idx_cols + cols].sort_values(idx_cols).set_index(idx_cols)
    print("\n[table] agent_correctness")
    print(_format_df_numbers_as_strings(out).to_string())


def print_kms_availability_table(df: pd.DataFrame) -> None:
    needed = {"scheme", "put_ok", "put_fail", "get_ok", "get_fail", "forget_ok", "forget_fail"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] kms_availability_eval missing columns {sorted(list(needed - set(df.columns)))}; skipping table.")
        return

    d = df.copy()
    for c in ["put_ok", "put_fail", "get_ok", "get_fail", "forget_ok", "forget_fail"]:
        d = _coerce_num(d, c)

    def rate(ok: float, fail: float) -> float:
        denom = (ok or 0.0) + (fail or 0.0)
        if denom <= 0:
            return float("nan")
        return float(ok) / float(denom)

    rows = []
    for _, r in d.iterrows():
        base = {"scheme": r["scheme"]}
        if "rtt_ms" in d.columns:
            base["rtt_ms"] = r["rtt_ms"]
        rows.append({
            **base,
            "put_success": rate(r["put_ok"], r["put_fail"]),
            "get_success": rate(r["get_ok"], r["get_fail"]),
            "forget_success": rate(r["forget_ok"], r["forget_fail"]),
        })

    out = pd.DataFrame(rows)
    if "rtt_ms" in out.columns:
        out = out.sort_values(["rtt_ms", "scheme"]).set_index(["rtt_ms", "scheme"])
    else:
        out = out.sort_values("scheme").set_index("scheme")

    print("\n[table] kms_availability_eval (success rates)")
    print(_format_df_numbers_as_strings(out).to_string())


# ----------------------------
# Agent workload: main paper figure only
# ----------------------------
def plot_agent_workload_throughput(df: pd.DataFrame, outdir: str) -> None:
    needed = {"scheme", "workload", "throughput_ops_s"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] agent_workload_eval missing columns {sorted(list(needed - set(df.columns)))}; skipping throughput plot.")
        return

    d = df.copy()
    d = _coerce_num(d, "throughput_ops_s")
    d = d.dropna(subset=["scheme", "workload", "throughput_ops_s"])
    if d.empty:
        return

    agg = d.groupby(["scheme", "workload"], as_index=False)["throughput_ops_s"].median()

    workloads = list(agg["workload"].dropna().unique())
    schemes = _sort_schemes(agg["scheme"].dropna().astype(str).unique().tolist())

    x = np.arange(len(workloads))
    width = 0.8 / max(1, len(schemes))

    plt.figure(figsize=(8.2, 4.6))
    for i, scheme in enumerate(schemes):
        vals = []
        for w in workloads:
            sub = agg[(agg["scheme"].astype(str) == scheme) & (agg["workload"] == w)]
            vals.append(float(sub["throughput_ops_s"].iloc[0]) if not sub.empty else np.nan)
        plt.bar(x + i * width, vals, width=width, label=_scheme_label(scheme), color=_scheme_color(scheme))

    plt.xticks(x + width * (len(schemes) - 1) / 2, workloads)
    plt.ylabel("Throughput (ops/s)")
    plt.xlabel("Agent workload")
    _grid()
    plt.legend()
    savefig(os.path.join(outdir, "agent_workload_throughput.png"))


def print_agent_workload_tables(df: pd.DataFrame) -> None:
    needed = {"scheme", "workload", "throughput_ops_s"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] agent_workload_eval missing columns {sorted(list(needed - set(df.columns)))}; skipping tables.")
        return

    d = df.copy()
    numeric_cols = [
        "throughput_ops_s", "p50_put_ms", "p95_put_ms",
        "p50_get_ms", "p95_get_ms",
        "p50_forget_ms", "p95_forget_ms",
        "db_size_bytes",
    ]
    for c in numeric_cols:
        d = _coerce_num(d, c)

    for metric in ["throughput_ops_s", "p95_put_ms", "p95_get_ms", "p95_forget_ms", "db_size_bytes"]:
        if metric not in d.columns:
            continue
        pivot = d.groupby(["scheme", "workload"], as_index=False)[metric].median().pivot(
            index="scheme", columns="workload", values=metric
        )
        print(f"\n[table] agent_workload_eval ({metric})")
        print(_format_df_numbers_as_strings(pivot).to_string())


# ----------------------------
# Storage growth: main paper figures only
# ----------------------------
def _storage_median(df: pd.DataFrame, value_col: str, workload: str) -> pd.DataFrame:
    d = df.copy()
    d = _coerce_num(d, "event")
    d = _coerce_num(d, value_col)
    d = d[d["workload"].astype(str) == workload].copy()
    d = d.dropna(subset=["scheme", "event", value_col])
    if d.empty:
        return d
    return d.groupby(["scheme", "event"], as_index=False)[value_col].median()


def plot_storage_paper_figure(
    df: pd.DataFrame,
    outdir: str,
    workload: str,
) -> None:
    """
    Paper storage-growth figure (DB size only).

    Each workload panel auto-scales to its own data range, but all panels
    use the same unit (KB), start at 0, and apply the same tick formatting
    so the figures are visually consistent when placed side-by-side.
    """

    needed = {"scheme", "workload", "event", "db_size_bytes"}
    if not needed.issubset(set(df.columns)):
        print(f"[warn] storage_growth_eval missing columns for {workload}")
        return

    d = df.copy()
    d["event"] = pd.to_numeric(d["event"], errors="coerce")
    d["db_size_bytes"] = pd.to_numeric(d["db_size_bytes"], errors="coerce")

    d = d[d["workload"] == workload]
    d = d.dropna(subset=["scheme", "event", "db_size_bytes"])

    if d.empty:
        return

    # Convert to KB for a clean, readable unit across all panels.
    d["db_size_kb"] = d["db_size_bytes"] / 1024.0

    # median across trials
    d = (
        d.groupby(["scheme", "event"], as_index=False)["db_size_kb"]
        .median()
        .sort_values("event")
    )

    plt.figure(figsize=(6.8, 4.2))

    schemes = sorted(d["scheme"].unique())

    for scheme in schemes:
        g = d[d["scheme"] == scheme]
        is_sage = scheme == "sage"
        plt.plot(
            g["event"],
            g["db_size_kb"],
            marker=_scheme_marker(scheme),
            markersize=9 if is_sage else 5,
            linewidth=2.8 if is_sage else 1.6,
            color=_scheme_color(scheme),
            label=scheme.replace("_", "-"),
        )

    plt.xlabel("Total memory operations")
    plt.ylabel("DB size (KB)")

    # Always start at 0 so growth rates are visually comparable across panels.
    plt.ylim(bottom=0)

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend(
        loc="lower right",
        ncol=1,
        framealpha=0.8,
    )

    plt.tight_layout()

    out = os.path.join(outdir, f"storage_dbsize_{workload}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[plot] wrote {out}")


def print_storage_growth_tables(df: pd.DataFrame) -> None:
    needed = {"scheme", "event", "db_size_bytes"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] storage_growth_eval missing columns {sorted(list(needed - set(df.columns)))}; skipping tables.")
        return

    d = df.copy()
    for c in ["event", "db_size_bytes", "rows", "deletions", "active_scopes", "scopes_created_total", "table_count"]:
        d = _coerce_num(d, c)

    if "workload" in d.columns:
        final_points = d.sort_values("event").groupby(["scheme", "workload"], as_index=False).tail(1)
        for metric in ["db_size_bytes", "rows", "deletions", "active_scopes", "table_count"]:
            if metric not in final_points.columns:
                continue
            pivot = final_points.pivot(index="scheme", columns="workload", values=metric)
            print(f"\n[table] storage_growth_eval final ({metric})")
            print(_format_df_numbers_as_strings(pivot).to_string())
    else:
        final_points = d.sort_values("event").groupby("scheme", as_index=False).tail(1)
        keep_cols = [c for c in ["db_size_bytes", "rows", "deletions", "table_count"] if c in final_points.columns]
        out = final_points.set_index("scheme")[keep_cols]
        print("\n[table] storage_growth_eval final")
        print(_format_df_numbers_as_strings(out).to_string())



# ----------------------------
# Provenance chain depth
# ----------------------------
def plot_provenance_depth(df: pd.DataFrame, outdir: str) -> None:
    """
    Three figures:
      1. Leaf get p95 latency vs depth (all schemes)
      2. SAGE DB size (KB) vs depth
      3. all_hidden correctness bar chart across depths
    Accepts both the current and older column naming variants.
    """
    d = df.copy()
    d = d.rename(columns={
        "chain_depth": "depth",
        "all_hidden_reopen": "all_hidden",
        "forget_ms": "forget_p95_ms",
    })
    if "db_size_bytes" in d.columns and "db_size_kb" not in d.columns:
        d["db_size_kb"] = pd.to_numeric(d["db_size_bytes"], errors="coerce") / 1024.0

    needed = {"scheme", "depth", "all_hidden"}
    if not needed.issubset(set(d.columns)):
        print(
            f"\n[warn] provenance_depth missing columns "
            f"{sorted(list(needed - set(d.columns)))}; skipping plots."
        )
        return

    for c in ["depth", "get_leaf_p95_ms", "get_root_p95_ms",
              "forget_p95_ms", "db_size_kb", "all_hidden"]:
        d = _coerce_num(d, c)

    agg_cols = {c: "median" for c in
                ["get_leaf_p95_ms", "get_root_p95_ms", "forget_p95_ms",
                 "db_size_kb", "all_hidden"] if c in d.columns}
    agg = d.groupby(["scheme", "depth"], as_index=False).agg(agg_cols)

    schemes = _sort_schemes(agg["scheme"].astype(str).unique().tolist())
    depths = sorted(agg["depth"].dropna().unique().tolist())

    # --- Figure 1: leaf get latency vs depth ---
    if "get_leaf_p95_ms" in agg.columns:
        plt.figure(figsize=(6.6, 4.2))
        for scheme in schemes:
            g = agg[agg["scheme"].astype(str) == scheme].sort_values("depth")
            if g["get_leaf_p95_ms"].isna().all():
                continue
            lw = 3.2 if scheme == "sage" else 2.0
            ms = 9 if scheme == "sage" else 6
            plt.plot(g["depth"], g["get_leaf_p95_ms"],
                     marker=_scheme_marker(scheme), linewidth=lw, markersize=ms,
                     label=_scheme_label(scheme), color=_scheme_color(scheme))
        plt.xlabel("Provenance chain depth")
        plt.ylabel("Leaf get latency p95 (ms)")
        plt.xticks(depths)
        _grid()
        plt.legend(loc="upper left", framealpha=0.85,
                   facecolor="white", edgecolor="0.2")
        savefig(os.path.join(outdir, "provenance_depth_get_latency.png"))
        print(f"[plot] wrote {os.path.join(outdir, 'provenance_depth_get_latency.png')}")

    # --- Figure 2: SAGE DB size vs depth ---
    if "db_size_kb" in agg.columns:
        sage_d = agg[agg["scheme"].astype(str) == "sage"].sort_values("depth")
        if not sage_d.empty and not sage_d["db_size_kb"].isna().all():
            plt.figure(figsize=(6.6, 4.2))
            plt.plot(sage_d["depth"], sage_d["db_size_kb"],
                     marker=_scheme_marker("sage"), linewidth=3.0, markersize=9,
                     color=_scheme_color("sage"), label="SAGE")
            plt.xlabel("Provenance chain depth")
            plt.ylabel("DB size (KB)")
            plt.xticks(depths)
            _grid()
            plt.legend()
            savefig(os.path.join(outdir, "provenance_depth_db_size.png"))
            print(f"[plot] wrote {os.path.join(outdir, 'provenance_depth_db_size.png')}")

    # --- Figure 3: all_hidden bar chart across depths ---
    if "all_hidden" in agg.columns:
        sage_d = agg[agg["scheme"].astype(str) == "sage"].sort_values("depth")
        if not sage_d.empty:
            plt.figure(figsize=(6.6, 4.2))
            bars = plt.bar(
                sage_d["depth"].astype(str),
                sage_d["all_hidden"],
                color=_scheme_color("sage"),
                width=0.5,
                label="SAGE",
            )
            plt.ylim(0, 1.15)
            plt.xlabel("Provenance chain depth")
            plt.ylabel("Fraction all_hidden")
            plt.axhline(1.0, linestyle="--", linewidth=1.2, color="gray", alpha=0.6)
            _grid()
            plt.legend()
            savefig(os.path.join(outdir, "provenance_depth_correctness.png"))
            print(f"[plot] wrote {os.path.join(outdir, 'provenance_depth_correctness.png')}")


def print_provenance_depth_table(df: pd.DataFrame) -> None:
    d = df.copy()
    d = d.rename(columns={
        "chain_depth": "depth",
        "all_hidden_reopen": "all_hidden",
        "forget_ms": "forget_p95_ms",
    })
    if "db_size_bytes" in d.columns and "db_size_kb" not in d.columns:
        d["db_size_kb"] = pd.to_numeric(d["db_size_bytes"], errors="coerce") / 1024.0

    needed = {"scheme", "depth", "all_hidden"}
    if not needed.issubset(set(d.columns)):
        print(f"\n[warn] provenance_depth missing columns; skipping table.")
        return

    for c in ["depth", "get_leaf_p95_ms", "get_root_p95_ms",
              "forget_p95_ms", "db_size_kb", "all_hidden"]:
        d = _coerce_num(d, c)

    cols = [c for c in ["get_leaf_p95_ms", "get_root_p95_ms",
                        "forget_p95_ms", "db_size_kb", "all_hidden"]
            if c in d.columns]
    agg = d.groupby(["scheme", "depth"], as_index=False)[cols].median()
    out = agg.set_index(["scheme", "depth"])
    print("\n[table] provenance_depth")
    print(_format_df_numbers_as_strings(out).to_string())


# ----------------------------
# Restart recovery
# ----------------------------
def plot_restart_recovery(df: pd.DataFrame, outdir: str) -> None:
    """
    Two figures:
      1. Grouped bar: open_p50_ms and first_get_p50_ms per scheme
      2. Grouped bar: forgotten_stays and rollback_defeated per scheme
    Columns expected: scheme, open_p50_ms, open_p95_ms,
                      first_get_p50_ms, first_get_p95_ms,
                      forgotten_stays_forgotten, rollback_defeated
    """
    needed = {"scheme", "rollback_defeated", "forgotten_stays_forgotten"}
    if not needed.issubset(set(df.columns)):
        print(
            f"\n[warn] restart_recovery missing columns "
            f"{sorted(list(needed - set(df.columns)))}; skipping plots."
        )
        return

    d = df.copy()
    for c in ["open_p50_ms", "open_p95_ms", "first_get_p50_ms",
              "first_get_p95_ms", "forgotten_stays_forgotten", "rollback_defeated"]:
        d = _coerce_num(d, c)

    agg_cols = {c: "median" for c in
                ["open_p50_ms", "open_p95_ms", "first_get_p50_ms",
                 "first_get_p95_ms", "forgotten_stays_forgotten", "rollback_defeated"]
                if c in d.columns}
    agg = d.groupby("scheme", as_index=False).agg(agg_cols)

    schemes = _sort_schemes(agg["scheme"].astype(str).unique().tolist())
    x = np.arange(len(schemes))
    labels = [_scheme_label(s) for s in schemes]

    # --- Figure 1: latency grouped bar ---
    lat_cols = [(c, lbl) for c, lbl in [
        ("open_p50_ms", "Open p50"),
        ("first_get_p50_ms", "First-get p50"),
    ] if c in agg.columns]

    if lat_cols:
        width = 0.35
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        for i, (col, lbl) in enumerate(lat_cols):
            vals = [float(agg[agg["scheme"].astype(str) == s][col].iloc[0])
                    if not agg[agg["scheme"].astype(str) == s].empty else 0.0
                    for s in schemes]
            offset = (i - (len(lat_cols) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=lbl)
            # highlight SAGE bar
            sage_idx = schemes.index("sage") if "sage" in schemes else -1
            if sage_idx >= 0:
                bars[sage_idx].set_edgecolor("black")
                bars[sage_idx].set_linewidth(2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Latency (ms)")
        _grid(ax)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(outdir, "restart_recovery_latency.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out}")

    # --- Figure 2: correctness grouped bar ---
    corr_cols = [(c, lbl) for c, lbl in [
        ("forgotten_stays_forgotten", "Forgotten stays"),
        ("rollback_defeated", "Rollback defeated"),
    ] if c in agg.columns]

    if corr_cols:
        width = 0.30
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        for i, (col, lbl) in enumerate(corr_cols):
            vals = [float(agg[agg["scheme"].astype(str) == s][col].iloc[0])
                    if not agg[agg["scheme"].astype(str) == s].empty else 0.0
                    for s in schemes]
            offset = (i - (len(corr_cols) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=lbl)
            sage_idx = schemes.index("sage") if "sage" in schemes else -1
            if sage_idx >= 0:
                bars[sage_idx].set_edgecolor("black")
                bars[sage_idx].set_linewidth(2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Fraction of trials")
        ax.axhline(1.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.5)
        _grid(ax)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(outdir, "restart_recovery_correctness.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out}")


def print_restart_recovery_table(df: pd.DataFrame) -> None:
    needed = {"scheme", "rollback_defeated", "forgotten_stays_forgotten"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] restart_recovery missing columns; skipping table.")
        return

    d = df.copy()
    for c in ["open_p50_ms", "open_p95_ms", "first_get_p50_ms",
              "first_get_p95_ms", "forgotten_stays_forgotten", "rollback_defeated"]:
        d = _coerce_num(d, c)

    cols = [c for c in ["open_p50_ms", "open_p95_ms", "first_get_p50_ms",
                        "first_get_p95_ms", "forgotten_stays_forgotten",
                        "rollback_defeated"] if c in d.columns]
    agg = d.groupby("scheme", as_index=False)[cols].median()
    out = agg.set_index("scheme")
    print("\n[table] restart_recovery")
    print(_format_df_numbers_as_strings(out).to_string())


# ----------------------------
# Deletion pressure
# ----------------------------
def plot_deletion_pressure(df: pd.DataFrame, outdir: str) -> None:
    """
    Three deletion-pressure figures:
      1. Throughput retained (% of 0 Hz baseline) vs forget_rate_hz
      2. Put p95 latency (ms) vs forget_rate_hz
      3. Absolute throughput (ops/s) vs forget_rate_hz
    Columns expected: scheme, forget_rate_hz, throughput_ops_s, put_p95_ms
    """
    needed = {"scheme", "forget_rate_hz", "throughput_ops_s", "put_p95_ms"}
    if not needed.issubset(set(df.columns)):
        print(
            f"\n[warn] deletion_pressure missing columns "
            f"{sorted(list(needed - set(df.columns)))}; skipping plot."
        )
        return

    d = df.copy()
    for c in ["forget_rate_hz", "throughput_ops_s", "put_p95_ms"]:
        d = _coerce_num(d, c)
    d = d.dropna(subset=["scheme", "forget_rate_hz",
                          "throughput_ops_s", "put_p95_ms"])
    if d.empty:
        return

    agg = (
        d.groupby(["scheme", "forget_rate_hz"], as_index=False)
        .agg(throughput_ops_s=("throughput_ops_s", "median"),
             put_p95_ms=("put_p95_ms", "median"))
    )

    schemes = _sort_schemes(agg["scheme"].astype(str).unique().tolist())
    # Use explicit tick values and labels.
    rates = sorted(agg["forget_rate_hz"].dropna().unique().tolist())
    rate_labels = [str(int(r)) if r == int(r) else str(r) for r in rates]

    # --- compute normalised throughput (% of each scheme's 0 Hz baseline) ---
    norm_rows = []
    for scheme in schemes:
        g = agg[agg["scheme"].astype(str) == scheme].sort_values(
            "forget_rate_hz")
        base_rows = g[g["forget_rate_hz"] == 0.0]["throughput_ops_s"]
        base = float(base_rows.iloc[0]) if not base_rows.empty else float("nan")
        for _, row in g.iterrows():
            pct = (float(row["throughput_ops_s"]) / base * 100.0) \
                if base > 0 else float("nan")
            # cap at 100: values above baseline are measurement noise
            if not np.isnan(pct):
                pct = min(pct, 100.0)
            norm_rows.append({
                "scheme": scheme,
                "forget_rate_hz": float(row["forget_rate_hz"]),
                "throughput_pct": pct,
            })
    norm = pd.DataFrame(norm_rows)

    def _line_kwargs(scheme: str) -> dict:
        is_sage = scheme == "sage"
        return dict(
            marker=_scheme_marker(scheme),
            linewidth=4.0 if is_sage else 1.8,
            markersize=12 if is_sage else 7,
            linestyle="-",
            zorder=5 if is_sage else 2,
            color=_scheme_color(scheme),
            label=_scheme_label(scheme),
        )

    # --- Figure 1: normalised throughput (PRIMARY paper figure) ---
    plt.figure(figsize=(6.6, 4.2))
    for scheme in schemes:
        g = norm[norm["scheme"].astype(str) == scheme].sort_values(
            "forget_rate_hz")
        plt.plot(g["forget_rate_hz"], g["throughput_pct"],
                 **_line_kwargs(scheme))
    plt.xlabel("Delete rate (Hz)")
    plt.ylabel("Throughput retained (%)")
    plt.xticks(rates, rate_labels)
    plt.ylim(0, 110)
    plt.axhline(100, linestyle=":", linewidth=1.2, color="gray", alpha=0.6)
    _grid()
    plt.legend(loc="lower left", framealpha=0.88,
               facecolor="white", edgecolor="0.2")
    savefig(os.path.join(outdir, "deletion_pressure_throughput.png"))
    print(f"[plot] wrote {os.path.join(outdir, 'deletion_pressure_throughput.png')}")

    # --- Figure 2: put p95 latency ---
    plt.figure(figsize=(6.6, 4.2))
    for scheme in schemes:
        g = agg[agg["scheme"].astype(str) == scheme].sort_values(
            "forget_rate_hz")
        plt.plot(g["forget_rate_hz"], g["put_p95_ms"],
                 **_line_kwargs(scheme))
    plt.xlabel("Delete rate (Hz)")
    plt.ylabel("Put latency p95 (ms)")
    plt.xticks(rates, rate_labels)
    _grid()
    plt.legend(loc="upper left", framealpha=0.88,
               facecolor="white", edgecolor="0.2")
    savefig(os.path.join(outdir, "deletion_pressure_put_p95.png"))
    print(f"[plot] wrote {os.path.join(outdir, 'deletion_pressure_put_p95.png')}")

    # --- Figure 3: absolute throughput ---
    plt.figure(figsize=(6.6, 4.2))
    for scheme in schemes:
        g = agg[agg["scheme"].astype(str) == scheme].sort_values(
            "forget_rate_hz")
        plt.plot(g["forget_rate_hz"], g["throughput_ops_s"],
                 **_line_kwargs(scheme))
    plt.xlabel("Delete rate (Hz)")
    plt.ylabel("Throughput (ops/s)")
    plt.xticks(rates, rate_labels)
    _grid()
    plt.legend(loc="upper right", framealpha=0.88,
               facecolor="white", edgecolor="0.2")
    savefig(os.path.join(outdir, "deletion_pressure_throughput_abs.png"))
    print(f"[plot] wrote "
          f"{os.path.join(outdir, 'deletion_pressure_throughput_abs.png')}")


def print_deletion_pressure_table(df: pd.DataFrame) -> None:
    needed = {"scheme", "forget_rate_hz", "throughput_ops_s", "put_p95_ms"}
    if not needed.issubset(set(df.columns)):
        print(f"\n[warn] deletion_pressure missing columns; skipping table.")
        return

    d = df.copy()
    for c in ["forget_rate_hz", "throughput_ops_s", "put_p95_ms"]:
        d = _coerce_num(d, c)
    d = d.dropna(subset=["scheme", "forget_rate_hz",
                          "throughput_ops_s", "put_p95_ms"])
    if d.empty:
        return

    agg = (
        d.groupby(["scheme", "forget_rate_hz"], as_index=False)
        .agg(throughput_ops_s=("throughput_ops_s", "median"),
             put_p95_ms=("put_p95_ms", "median"))
    )

    rows = []
    for scheme in _sort_schemes(agg["scheme"].astype(str).unique().tolist()):
        g = agg[agg["scheme"].astype(str) == scheme].sort_values(
            "forget_rate_hz")
        baseline = g[g["forget_rate_hz"] == 0.0]["throughput_ops_s"]
        base_val = float(baseline.iloc[0]) if not baseline.empty \
            else float("nan")
        for _, row in g.iterrows():
            thr = float(row["throughput_ops_s"])
            deg = ((base_val - thr) / base_val * 100.0) \
                if base_val > 0 else float("nan")
            rows.append({
                "scheme": _scheme_label(scheme),
                "Hz": int(row["forget_rate_hz"]),
                "throughput (ops/s)": thr,
                "deg (%)": deg,
                "put p95 (ms)": float(row["put_p95_ms"]),
            })

    out = pd.DataFrame(rows).set_index(["scheme", "Hz"])
    print("\n[table] deletion_pressure")
    print(_format_df_numbers_as_strings(out).to_string())

def main():
    _style()

    ap = argparse.ArgumentParser()

    ap.add_argument("--bench", default="results/bench_all.csv")
    ap.add_argument("--extra", default="results/bench_extra.csv")
    ap.add_argument("--rollback", default="results/rollback_matrix.csv")

    ap.add_argument("--kms_latency_eval", default="results/kms_latency_eval.csv")
    ap.add_argument("--kms_availability_eval", default="results/kms_availability_eval.csv")
    ap.add_argument("--concurrency_throughput", default="results/concurrency_throughput.csv")
    ap.add_argument("--agent_correctness", default="results/agent_correctness.csv")
    ap.add_argument("--agent_workload_eval", default="results/agent_workload_eval.csv")
    ap.add_argument("--storage_growth_eval", default="results/storage_growth_eval.csv")
    ap.add_argument("--deletion_pressure", default="results/deletion_pressure_eval.csv")
    ap.add_argument("--provenance_depth", default="results/provenance_depth_eval.csv")
    ap.add_argument("--restart_recovery", default="results/restart_recovery_eval.csv")

    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--min_points", type=int, default=6, help="min distinct x points needed to plot")

    ap.add_argument(
        "--plot",
        default="all",
        help=(
            "Comma-separated tags: all,bench,extra,rollback,kms_latency,"
            "concurrency,agent_correctness,kms_availability,"
            "agent_workload,storage_growth,deletion_pressure,provenance_depth,restart_recovery"
        ),
    )

    ap.add_argument("--latency_heatmaps", action="store_true", help="plot RTT×TTL heatmaps for kms_latency_eval")
    ap.add_argument("--latency_lines", action="store_true", help="plot line charts for kms_latency_eval")
    ap.add_argument(
        "--latency_percentile",
        default="p95_ms",
        choices=["p50_ms", "p95_ms", "p99_ms"],
        help="which percentile column to visualize",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tags = _parse_plot_tags(args.plot)

    bench = _read_csv_if_exists(args.bench)
    extra = _read_csv_if_exists(args.extra)
    rollback = _read_csv_if_exists(args.rollback)

    kms_lat = _read_csv_if_exists(args.kms_latency_eval)
    kms_av = _read_csv_if_exists(args.kms_availability_eval)
    thr = _read_csv_if_exists(args.concurrency_throughput)
    corr = _read_csv_if_exists(args.agent_correctness)
    agent_work = _read_csv_if_exists(args.agent_workload_eval)
    storage_growth = _read_csv_if_exists(args.storage_growth_eval)
    del_pressure = _read_csv_if_exists(args.deletion_pressure)
    prov_depth = _read_csv_if_exists(args.provenance_depth)
    restart_rec = _read_csv_if_exists(args.restart_recovery)

    plotted = set()

    if _want(tags, "bench") and bench is not None and "experiment" in bench.columns:
        rules = [
            PlotRule("put_latency", "payload_bytes", "ms per put (median)", "payload bytes (log)", x_log=True, outname="put_latency.png"),
            PlotRule("get_latency", "k", "ms per get (median)", "k retrieved (log)", x_log=True, outname="get_latency.png"),
            PlotRule("forget_vs_n", "n_items", "forget latency (ms, median)", "N items in scope (log)", x_log=True, outname="forget_vs_n.png"),
        ]
        for rule in rules:
            ok = plot_line_if_dense(bench, args.outdir, rule, min_points=args.min_points)
            if ok:
                plotted.add(rule.experiment)

        for exp in sorted(set(bench["experiment"].astype(str).unique())):
            if exp in plotted:
                continue
            if exp == "forget_latency":
                print_compact_table(bench, "forget_latency", xcol=None)
            else:
                printed = False
                for candidate in ["payload_bytes", "k", "n_items"]:
                    if candidate in bench.columns and bench[bench["experiment"] == exp][candidate].notna().any():
                        print_compact_table(bench, exp, xcol=candidate)
                        printed = True
                        break
                if not printed:
                    print_compact_table(bench, exp, xcol=None)

    if _want(tags, "extra") and extra is not None and "experiment" in extra.columns:
        rule = PlotRule("storage_overhead", "n_items", "DB size (bytes, log)", "N items (log)", x_log=True, y_log=True, outname="storage_overhead.png")
        ok = plot_line_if_dense(extra, args.outdir, rule, min_points=args.min_points)
        if ok:
            plotted.add("storage_overhead")
        else:
            print_compact_table(extra, "storage_overhead", xcol="n_items")

        if (extra["experiment"] == "restart_overhead").any():
            d = extra[extra["experiment"] == "restart_overhead"].copy()
            d = _median_only(d)
            d = _coerce_num(d, "value")
            if "metric" in d.columns:
                for m in ["ms_open", "ms_first_put", "ms_first_get"]:
                    dm = d[d["metric"] == m].copy()
                    if dm.empty:
                        continue
                    dm = dm[["scheme", "value"]].dropna().sort_values("scheme")
                    print(f"\n[table] restart_overhead ({m})")
                    dm_fmt = dm.copy()
                    dm_fmt["value"] = dm_fmt["value"].map(lambda v: _pretty(float(v)) if pd.notna(v) else "")
                    print(dm_fmt.set_index("scheme")["value"].to_string())
            else:
                print_compact_table(extra, "restart_overhead", xcol=None)

    if _want(tags, "rollback") and rollback is not None:
        cols = ["scheme", "recovered_items_after_rollback"]
        if all(c in rollback.columns for c in cols):
            d = rollback[cols].copy().sort_values("scheme")
            print("\n[table] rollback_matrix (recovered_items_after_rollback)")
            print(d.set_index("scheme")["recovered_items_after_rollback"].to_string())
        else:
            print("\n[warn] rollback CSV missing expected columns; skipping print.")

    if _want(tags, "kms_latency") and kms_lat is not None:
        do_heatmaps = args.latency_heatmaps or (not args.latency_heatmaps and not args.latency_lines)
        do_lines = args.latency_lines
        if do_heatmaps:
            plot_kms_latency_eval_heatmaps(kms_lat, args.outdir, percentile_col=args.latency_percentile)
        if do_lines:
            plot_kms_latency_eval_lines(kms_lat, args.outdir, percentile_col=args.latency_percentile)

    if _want(tags, "concurrency") and thr is not None:
        plot_concurrency_throughput(thr, args.outdir)
        print_concurrency_tables(thr)

    if _want(tags, "agent_correctness") and corr is not None:
        print_agent_correctness_table(corr)

    if _want(tags, "kms_availability") and kms_av is not None:
        print_kms_availability_table(kms_av)

    if _want(tags, "agent_workload") and agent_work is not None:
        plot_agent_workload_throughput(agent_work, args.outdir)
        print_agent_workload_tables(agent_work)

    if _want(tags, "storage_growth") and storage_growth is not None:
        if "workload" in storage_growth.columns:
            for workload in sorted(storage_growth["workload"].dropna().astype(str).unique()):
                plot_storage_paper_figure(storage_growth, args.outdir, workload)
        print_storage_growth_tables(storage_growth)

    if _want(tags, "deletion_pressure") and del_pressure is not None:
        # normalise column names: deletion_pressure_eval.py uses throughput_ops_s
        dp = del_pressure.copy()
        if "throughput_ops_s" not in dp.columns and "throughput_ops_per_s" in dp.columns:
            dp = dp.rename(columns={"throughput_ops_per_s": "throughput_ops_s"})
        if "put_p95_ms" not in dp.columns and "put_p95" in dp.columns:
            dp = dp.rename(columns={"put_p95": "put_p95_ms"})
        if "forget_rate_hz" not in dp.columns and "forget_rate" in dp.columns:
            dp = dp.rename(columns={"forget_rate": "forget_rate_hz"})
        plot_deletion_pressure(dp, args.outdir)
        print_deletion_pressure_table(dp)

    if _want(tags, "provenance_depth") and prov_depth is not None:
        plot_provenance_depth(prov_depth, args.outdir)
        print_provenance_depth_table(prov_depth)

    if _want(tags, "restart_recovery") and restart_rec is not None:
        plot_restart_recovery(restart_rec, args.outdir)
        print_restart_recovery_table(restart_rec)

    print(f"\nWrote figures to: {args.outdir}")


if __name__ == "__main__":
    main()
