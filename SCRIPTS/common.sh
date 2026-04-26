#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_DIR="$ARTIFACT_ROOT/CODE"
CONFIG_DIR="$ARTIFACT_ROOT/CONFIGS"
RESULTS_DIR="$ARTIFACT_ROOT/RESULTS"
MAIN_DIR="$RESULTS_DIR/main"
APPENDIX_DIR="$RESULTS_DIR/appendix"
FIGURES_DIR="$RESULTS_DIR/figures"
WORK_DIR="$RESULTS_DIR/work"
LOG_DIR="$RESULTS_DIR/logs"
SWTPM_DIR="$WORK_DIR/swtpm"
MPLCONFIG_DIR="$WORK_DIR/mplconfig"

export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$MPLCONFIG_DIR"

ensure_layout() {
  mkdir -p \
    "$RESULTS_DIR" \
    "$MAIN_DIR" \
    "$APPENDIX_DIR" \
    "$FIGURES_DIR" \
    "$WORK_DIR" \
    "$LOG_DIR" \
    "$SWTPM_DIR" \
    "$MPLCONFIG_DIR"
}

announce() {
  printf '\n== %s ==\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  }
}

run_module() {
  local module="$1"
  shift
  (
    cd "$CODE_DIR"
    python -m "$module" "$@"
  )
}
