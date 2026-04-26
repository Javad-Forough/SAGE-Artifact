#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$CONFIG_DIR/paper-defaults.env"

ensure_layout
require_command curl

announce "Checking for a local Ollama server"
curl -sf http://localhost:11434/api/tags >/dev/null

announce "Agent correctness evaluation"
run_module experiments.agent_correctness_eval \
  --schemes "$AGENT_EVAL_SCHEMES" \
  --agents "$AGENT_KINDS" \
  --trials "$AGENT_TRIALS" \
  --ollama-model "$OLLAMA_MODEL" \
  --workdir "$WORK_DIR/agent_correctness" \
  --out "$MAIN_DIR/table_agent_rollback.csv"

printf '\nAgent correctness evaluation completed. Output is %s\n' \
  "$MAIN_DIR/table_agent_rollback.csv"
