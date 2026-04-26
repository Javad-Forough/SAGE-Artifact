#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$CONFIG_DIR/paper-defaults.env"
source "$CONFIG_DIR/tpm-defaults.env"

ensure_layout
require_command tpm2_startup
require_command tpm2_nvread
require_command tpm2_nvincrement

MODE="${1:---sim}"
STARTED_SIM=0
ENV_PATH="$SWTPM_DIR/$TPM_STATE_SUBDIR/tpm-env.sh"

case "$MODE" in
  --sim)
    require_command swtpm
    "$SCRIPT_DIR/start_swtpm.sh"
    # shellcheck disable=SC1090
    source "$ENV_PATH"
    STARTED_SIM=1
    ;;
  --real)
    :
    ;;
  *)
    printf 'Usage: %s [--sim|--real]\n' "$0" >&2
    exit 1
    ;;
esac

cleanup() {
  if [[ "$STARTED_SIM" -eq 1 ]]; then
    "$SCRIPT_DIR/stop_swtpm.sh" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

announce "TPM overhead evaluation"
run_module experiments.tpm_overhead_eval \
  --schemes "$TPM_SCHEMES" \
  --workdir "$WORK_DIR/$TPM_WORKDIR_SUBDIR" \
  --out "$MAIN_DIR/table_tpm_overhead.csv" \
  --repeats "$TPM_REPEATS" \
  --warmup "$TPM_WARMUP" \
  --put_ops "$TPM_PUT_OPS" \
  --get_queries "$TPM_GET_QUERIES" \
  --populate "$TPM_POPULATE" \
  --payload_bytes "$PAYLOAD_BYTES" \
  --k "$K_VALUE"

printf '\nTPM evaluation completed. Output is %s\n' \
  "$MAIN_DIR/table_tpm_overhead.csv"
