#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_layout

announce "Regenerating figures from available CSV outputs"
run_module experiments.plot_results \
  --bench "$MAIN_DIR/table_microbench.csv" \
  --kms_latency_eval "$MAIN_DIR/figure_kms_latency.csv" \
  --kms_availability_eval "$APPENDIX_DIR/table_kms_availability.csv" \
  --concurrency_throughput "$MAIN_DIR/figure_concurrency_throughput.csv" \
  --agent_correctness "$MAIN_DIR/table_agent_rollback.csv" \
  --storage_growth_eval "$MAIN_DIR/figure_storage_growth.csv" \
  --deletion_pressure "$MAIN_DIR/figure_deletion_pressure.csv" \
  --provenance_depth "$MAIN_DIR/table_provenance_length.csv" \
  --restart_recovery "$MAIN_DIR/table_restart_recovery.csv" \
  --outdir "$FIGURES_DIR" \
  --latency_heatmaps \
  --latency_lines

printf '\nFigures written to %s\n' "$FIGURES_DIR"
