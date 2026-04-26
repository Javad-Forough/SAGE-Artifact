#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$CONFIG_DIR/tpm-defaults.env"

STATE_DIR="$SWTPM_DIR/$TPM_STATE_SUBDIR"
PID_PATH="$STATE_DIR/swtpm.pid"

if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  kill "$(cat "$PID_PATH")"
  rm -f "$PID_PATH"
  printf 'Stopped swtpm.\n'
else
  printf 'No running swtpm instance recorded at %s\n' "$PID_PATH"
fi
