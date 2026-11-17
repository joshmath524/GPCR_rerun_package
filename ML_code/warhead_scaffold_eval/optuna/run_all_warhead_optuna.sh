#!/usr/bin/env bash
# Optuna retune RF → XGB → LightGBM on curated warhead features (100 trials each).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export DYLD_LIBRARY_PATH="$ROOT/.venv/lib/python3.13/site-packages/sklearn/.dylibs:${DYLD_LIBRARY_PATH:-}"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/ML_code/warhead_scaffold_eval/optuna"
LOG="$OUT/logs"
mkdir -p "$LOG"
N_TRIALS="${1:-100}"

run_one () {
  local name="$1"
  local script="$2"
  echo "===== START $name $(date) n_trials=$N_TRIALS =====" | tee -a "$LOG/master.log"
  "$PY" -u "$OUT/$script" --n-trials "$N_TRIALS" 2>&1 | tee "$LOG/${name}.log"
  echo "===== DONE $name $(date) =====" | tee -a "$LOG/master.log"
}

run_one rf rf_warhead_optuna.py
run_one xgb xgb_warhead_optuna.py
run_one lightgbm lightgbm_warhead_optuna.py
echo "All warhead Optuna searches finished $(date)" | tee -a "$LOG/master.log"
