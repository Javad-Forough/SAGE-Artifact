#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$CONFIG_DIR/paper-defaults.env"
source "$CONFIG_DIR/smoke-defaults.env"

ensure_layout

SMOKE_DIR="$RESULTS_DIR/smoke"
SMOKE_WORK="$WORK_DIR/smoke"
SMOKE_FIG="$FIGURES_DIR/smoke"
mkdir -p "$SMOKE_DIR" "$SMOKE_WORK" "$SMOKE_FIG"

announce "Smoke test: rollback matrix"
run_module experiments.rollback_matrix \
  --schemes "$SMOKE_CORE_SCHEMES" \
  --agents "$AGENT_KINDS" \
  --db "$SMOKE_WORK/rollback_matrix.db" \
  --snap "$SMOKE_WORK/rollback_matrix.db.snapshot" \
  --out "$SMOKE_DIR/rollback_matrix.csv"

announce "Smoke test: restart recovery"
run_module experiments.restart_recovery_eval \
  --schemes "$SMOKE_CORE_SCHEMES" \
  --trials "$SMOKE_RESTART_TRIALS" \
  --n_items "$SMOKE_RESTART_N_ITEMS" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --db_prefix "$SMOKE_WORK/restart" \
  --out "$SMOKE_DIR/table_restart_recovery.csv"

announce "Smoke test: provenance length"
run_module experiments.provenance_depth_eval \
  --schemes "$SMOKE_CORE_SCHEMES" \
  --depths "$SMOKE_DEPTHS" \
  --trials "$SMOKE_PROVENANCE_TRIALS" \
  --get_reps "$SMOKE_PROVENANCE_GET_REPS" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --db_prefix "$SMOKE_WORK/provenance" \
  --out "$SMOKE_DIR/table_provenance_length.csv"

announce "Smoke test: microbenchmark"
run_module experiments.bench_all \
  --schemes "$SMOKE_BENCH_SCHEMES" \
  --db "$SMOKE_WORK/bench.db" \
  --out "$SMOKE_DIR/table_microbench.csv" \
  --put_ops "$SMOKE_BENCH_PUT_OPS" \
  --get_queries "$SMOKE_BENCH_GET_QUERIES" \
  --populate "$SMOKE_BENCH_POPULATE" \
  --payloads "$SMOKE_BENCH_PAYLOADS" \
  --ks "$SMOKE_BENCH_KS" \
  --forget_ns "$SMOKE_BENCH_FORGET_NS" \
  --repeats "$SMOKE_BENCH_REPEATS" \
  --warmup "$SMOKE_BENCH_WARMUP" \
  --batch_size 0

announce "Smoke test: deletion pressure"
run_module experiments.deletion_pressure_eval \
  --schemes "$SMOKE_PERF_SCHEMES" \
  --forget_rates "$SMOKE_DELETION_RATES" \
  --workers 4 \
  --duration_s "$SMOKE_DELETION_DURATION_S" \
  --num_scopes "$SMOKE_DELETION_NUM_SCOPES" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --k "$K_VALUE" \
  --trials "$SMOKE_DELETION_TRIALS" \
  --db_prefix "$SMOKE_WORK/deletion_pressure" \
  --out "$SMOKE_DIR/figure_deletion_pressure.csv"

announce "Smoke test: concurrency throughput"
run_module experiments.concurrency_throughput_eval \
  --schemes "$SMOKE_PERF_SCHEMES" \
  --agents "$SMOKE_CONCURRENCY_AGENTS" \
  --workers "$SMOKE_CONCURRENCY_WORKERS" \
  --duration_s "$SMOKE_CONCURRENCY_DURATION_S" \
  --trials "$SMOKE_CONCURRENCY_TRIALS" \
  --db_prefix "$SMOKE_WORK/concurrency" \
  --out "$SMOKE_DIR/figure_concurrency_throughput.csv"

announce "Smoke test: scope scaling"
run_module experiments.scope_scaling_eval \
  --schemes "$SMOKE_PERF_SCHEMES" \
  --scope_sizes "$SMOKE_SCOPE_SIZES" \
  --trials "$SMOKE_SCOPE_TRIALS" \
  --measure_ops "$SMOKE_SCOPE_MEASURE_OPS" \
  --db_prefix "$SMOKE_WORK/scope_scaling" \
  --out "$SMOKE_DIR/table_scope_scaling.csv"

announce "Smoke test: storage growth"
run_module experiments.storage_growth_eval \
  --schemes "$SMOKE_PERF_SCHEMES" \
  --events "$SMOKE_STORAGE_EVENTS" \
  --trials "$SMOKE_STORAGE_TRIALS" \
  --db_prefix "$SMOKE_WORK/storage_growth" \
  --out "$SMOKE_DIR/figure_storage_growth.csv"

announce "Smoke test: KMS latency"
run_module experiments.kms_latency_eval \
  --schemes "kms,sage" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --k "$K_VALUE" \
  --ops "$SMOKE_KMS_OPS" \
  --warmup "$SMOKE_KMS_WARMUP" \
  --populate "$SMOKE_KMS_POPULATE" \
  --forget_populate "$SMOKE_KMS_FORGET_POPULATE" \
  --forget_reps "$SMOKE_KMS_FORGET_REPS" \
  --forget_batch "$SMOKE_KMS_FORGET_BATCH" \
  --rtts_ms "$SMOKE_KMS_RTTS_MS" \
  --cache_ttls "$SMOKE_KMS_CACHE_TTLS" \
  --db "$SMOKE_WORK/kms_latency.db" \
  --out "$SMOKE_DIR/figure_kms_latency.csv"

announce "Smoke test: figure generation"
run_module experiments.plot_results \
  --bench "$SMOKE_DIR/table_microbench.csv" \
  --kms_latency_eval "$SMOKE_DIR/figure_kms_latency.csv" \
  --concurrency_throughput "$SMOKE_DIR/figure_concurrency_throughput.csv" \
  --storage_growth_eval "$SMOKE_DIR/figure_storage_growth.csv" \
  --deletion_pressure "$SMOKE_DIR/figure_deletion_pressure.csv" \
  --provenance_depth "$SMOKE_DIR/table_provenance_length.csv" \
  --restart_recovery "$SMOKE_DIR/table_restart_recovery.csv" \
  --outdir "$SMOKE_FIG" \
  --min_points 2 \
  --latency_heatmaps \
  --latency_lines

printf '\nSmoke test completed. Outputs are in %s\n' "$SMOKE_DIR"
