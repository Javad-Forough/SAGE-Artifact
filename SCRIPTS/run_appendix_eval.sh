#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$CONFIG_DIR/appendix-defaults.env"

ensure_layout

TMP_A="$WORK_DIR/kms_availability_rtt25.csv"
TMP_B="$WORK_DIR/kms_availability_rtt5.csv"
OUT="$APPENDIX_DIR/table_kms_availability.csv"

announce "Appendix evaluation: KMS availability at RTT 25 ms"
run_module experiments.kms_availability_eval \
  --schemes "$KMS_AVAIL_SCHEMES" \
  --ops "$KMS_AVAIL_OPS" \
  --populate "$KMS_AVAIL_POPULATE" \
  --payload_bytes "$KMS_AVAIL_PAYLOAD_BYTES" \
  --k "$KMS_AVAIL_K" \
  --rtt_ms 25 \
  --jitter_ms "$KMS_AVAIL_JITTER_MS" \
  --failure_rate "$KMS_AVAIL_FAILURE_RATE" \
  --cache_ttl_s "$KMS_AVAIL_CACHE_TTL_S" \
  --outage_start_after_s "$KMS_AVAIL_OUTAGE_START_AFTER_S" \
  --outage_duration_s "$KMS_AVAIL_OUTAGE_DURATION_S" \
  --db "$WORK_DIR/kms_availability_rtt25.db" \
  --out "$TMP_A"

announce "Appendix evaluation: KMS availability at RTT 5 ms"
run_module experiments.kms_availability_eval \
  --schemes "$KMS_AVAIL_SCHEMES" \
  --ops "$KMS_AVAIL_OPS" \
  --populate "$KMS_AVAIL_POPULATE" \
  --payload_bytes "$KMS_AVAIL_PAYLOAD_BYTES" \
  --k "$KMS_AVAIL_K" \
  --rtt_ms 5 \
  --jitter_ms "$KMS_AVAIL_JITTER_MS" \
  --failure_rate "$KMS_AVAIL_FAILURE_RATE" \
  --cache_ttl_s "$KMS_AVAIL_CACHE_TTL_S" \
  --outage_start_after_s "$KMS_AVAIL_OUTAGE_START_AFTER_S" \
  --outage_duration_s "$KMS_AVAIL_OUTAGE_DURATION_S" \
  --db "$WORK_DIR/kms_availability_rtt5.db" \
  --out "$TMP_B"

announce "Merging appendix availability outputs"
{
  head -n 1 "$TMP_A"
  tail -n +2 "$TMP_A"
  tail -n +2 "$TMP_B"
} > "$OUT"

printf '\nAppendix evaluation completed. Output is %s\n' "$OUT"
