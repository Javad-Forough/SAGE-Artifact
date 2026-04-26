#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$CONFIG_DIR/paper-defaults.env"

ensure_layout

announce "Main software evaluation: rollback matrix"
run_module experiments.rollback_matrix \
  --schemes "$SOFTWARE_SCHEMES" \
  --agents "$AGENT_KINDS" \
  --db "$WORK_DIR/rollback_matrix.db" \
  --snap "$WORK_DIR/rollback_matrix.db.snapshot" \
  --out "$MAIN_DIR/rollback_matrix.csv"

announce "Main software evaluation: restart recovery"
run_module experiments.restart_recovery_eval \
  --schemes "$SOFTWARE_SCHEMES" \
  --trials "$RESTART_TRIALS" \
  --n_items "$RESTART_N_ITEMS" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --db_prefix "$WORK_DIR/restart" \
  --out "$MAIN_DIR/table_restart_recovery.csv"

announce "Main software evaluation: provenance length"
run_module experiments.provenance_depth_eval \
  --schemes "$SOFTWARE_SCHEMES" \
  --depths "$DEPTHS" \
  --trials "$PROVENANCE_TRIALS" \
  --get_reps "$PROVENANCE_GET_REPS" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --db_prefix "$WORK_DIR/provenance" \
  --out "$MAIN_DIR/table_provenance_length.csv"

announce "Main software evaluation: microbenchmark"
run_module experiments.bench_all \
  --schemes "plain,sage" \
  --db "$WORK_DIR/bench.db" \
  --out "$MAIN_DIR/table_microbench.csv" \
  --put_ops "$BENCH_PUT_OPS" \
  --get_queries "$BENCH_GET_QUERIES" \
  --populate "$BENCH_POPULATE" \
  --payloads "$BENCH_PAYLOADS" \
  --ks "$BENCH_KS" \
  --forget_ns "$BENCH_FORGET_NS" \
  --repeats "$BENCH_REPEATS" \
  --warmup "$BENCH_WARMUP" \
  --batch_size "$BENCH_BATCH_SIZE"

announce "Main software evaluation: deletion pressure"
run_module experiments.deletion_pressure_eval \
  --schemes "$SOFTWARE_SCHEMES" \
  --forget_rates "$FORGET_RATES" \
  --workers 4 \
  --duration_s "$DELETION_DURATION_S" \
  --num_scopes "$DELETION_NUM_SCOPES" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --k "$K_VALUE" \
  --trials "$DELETION_TRIALS" \
  --db_prefix "$WORK_DIR/deletion_pressure" \
  --out "$MAIN_DIR/figure_deletion_pressure.csv"

announce "Main software evaluation: concurrency throughput"
run_module experiments.concurrency_throughput_eval \
  --schemes "$SOFTWARE_SCHEMES" \
  --agents "$CONCURRENCY_AGENTS" \
  --workers "$WORKERS" \
  --duration_s "$CONCURRENCY_DURATION_S" \
  --trials "$CONCURRENCY_TRIALS" \
  --db_prefix "$WORK_DIR/concurrency" \
  --out "$MAIN_DIR/figure_concurrency_throughput.csv"

announce "Main software evaluation: scope scaling"
run_module experiments.scope_scaling_eval \
  --schemes "kms,sage" \
  --scope_sizes "$SCOPE_SIZES" \
  --trials "$SCOPE_TRIALS" \
  --measure_ops "$SCOPE_MEASURE_OPS" \
  --db_prefix "$WORK_DIR/scope_scaling" \
  --out "$MAIN_DIR/table_scope_scaling.csv"

announce "Main software evaluation: storage growth"
run_module experiments.storage_growth_eval \
  --schemes "$SOFTWARE_SCHEMES" \
  --events "$STORAGE_EVENTS" \
  --trials "$STORAGE_TRIALS" \
  --db_prefix "$WORK_DIR/storage_growth" \
  --out "$MAIN_DIR/figure_storage_growth.csv"

announce "Main software evaluation: KMS latency"
run_module experiments.kms_latency_eval \
  --schemes "kms,sage" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --k "$K_VALUE" \
  --ops "$KMS_OPS" \
  --warmup "$KMS_WARMUP" \
  --populate "$KMS_POPULATE" \
  --forget_populate "$KMS_FORGET_POPULATE" \
  --forget_reps "$KMS_FORGET_REPS" \
  --forget_batch "$KMS_FORGET_BATCH" \
  --rtts_ms "$KMS_RTTS_MS" \
  --jitter_ms "$KMS_JITTER_MS" \
  --failure_rate "$KMS_FAILURE_RATE" \
  --cache_ttls "$KMS_CACHE_TTLS" \
  --db "$WORK_DIR/kms_latency.db" \
  --out "$MAIN_DIR/figure_kms_latency.csv"

printf '\nMain software evaluation completed. Outputs are in %s\n' "$MAIN_DIR"
