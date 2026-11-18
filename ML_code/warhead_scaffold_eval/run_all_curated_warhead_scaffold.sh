#!/usr/bin/env bash
# Sequential RF → XGB → LightGBM scaffold evals with curated warhead features.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
OUT="$ROOT/ML_code/warhead_scaffold_eval"
LOG="$OUT/logs"
mkdir -p "$LOG"

run_one () {
  local name="$1"
  local script="$2"
  echo "===== START $name $(date) =====" | tee -a "$LOG/curated_master.log"
  "$PY" -u "$OUT/$script" 2>&1 | tee "$LOG/${name}_curated.log"
  echo "===== DONE $name $(date) =====" | tee -a "$LOG/curated_master.log"
}

run_one rf rf_scaffold_eval_warhead.py
run_one xgb xgb_scaffold_eval_warhead.py
run_one lightgbm lightgbm_scaffold_eval_warhead.py
echo "All curated warhead scaffold evals finished $(date)" | tee -a "$LOG/curated_master.log"
