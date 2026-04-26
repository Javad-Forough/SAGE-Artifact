#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$CONFIG_DIR/tpm-defaults.env"

require_command swtpm
require_command tpm2_startup
ensure_layout

STATE_DIR="$SWTPM_DIR/$TPM_STATE_SUBDIR"
SOCK_PATH="$STATE_DIR/swtpm.sock"
CTRL_PATH="$STATE_DIR/swtpm.ctrl"
PID_PATH="$STATE_DIR/swtpm.pid"
ENV_PATH="$STATE_DIR/tpm-env.sh"

mkdir -p "$STATE_DIR/state"

if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  printf 'swtpm is already running. Environment file: %s\n' "$ENV_PATH"
else
  rm -f "$SOCK_PATH" "$CTRL_PATH" "$PID_PATH"
  swtpm socket \
    --tpm2 \
    --server type=unixio,path="$SOCK_PATH" \
    --ctrl type=unixio,path="$CTRL_PATH" \
    --tpmstate dir="$STATE_DIR/state" \
    --flags startup-clear \
    --daemon \
    --pid file="$PID_PATH"

  export TPM2TOOLS_TCTI="swtpm:path=$SOCK_PATH"
  tpm2_startup -c >/dev/null 2>&1 || true
fi

cat > "$ENV_PATH" <<EOF
export TPM2TOOLS_TCTI="swtpm:path=$SOCK_PATH"
EOF

printf 'Started swtpm.\n'
printf 'To use it in the current shell:\n'
printf '  source %s\n' "$ENV_PATH"
