#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

announce "Removing generated outputs, caches, and bytecode"

rm -rf \
  "$RESULTS_DIR/smoke" \
  "$MAIN_DIR" \
  "$APPENDIX_DIR" \
  "$FIGURES_DIR" \
  "$WORK_DIR" \
  "$LOG_DIR" \
  "$CODE_DIR/results"

rm -f \
  "$CODE_DIR"/sealed_root_key*.bin \
  "$CODE_DIR"/dev_master*.bin \
  "$CODE_DIR"/static_key.bin \
  "$CODE_DIR"/kms_state.json \
  "$CODE_DIR"/sqlite_envelope_keys.json \
  "$CODE_DIR"/epochs*.db \
  "$CODE_DIR"/epochs*.db-wal \
  "$CODE_DIR"/epochs*.db-shm \
  "$CODE_DIR"/epochs*.json

find "$ARTIFACT_ROOT" -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
find "$ARTIFACT_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.log' \) -delete

ensure_layout
printf 'Cleanup complete.\n'
