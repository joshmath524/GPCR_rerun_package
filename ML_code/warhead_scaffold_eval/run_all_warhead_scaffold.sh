#!/usr/bin/env bash
# Run RF / XGB / LightGBM Bemis-Murcko scaffold evals with warhead features.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export DYLD_LIBRARY_PATH="$ROOT/.venv/lib/python3.13/site-packages/sklearn/.dylibs:${DYLD_LIBRARY_PATH:-}"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/ML_code/warhead_scaffold_eval"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

run_one () {
  local name="$1"
  local script="$2"
  local log="$LOGDIR/${name}.log"
  echo "===== START $name $(date) =====" | tee -a "$LOGDIR/master.log"
  "$PY" -u "$OUT/$script" 2>&1 | tee "$log"
  echo "===== DONE $name $(date) =====" | tee -a "$LOGDIR/master.log"
}

run_one rf rf_scaffold_eval_warhead.py
run_one xgb xgb_scaffold_eval_warhead.py
run_one lightgbm lightgbm_scaffold_eval_warhead.py

echo "All warhead scaffold evals finished $(date)" | tee -a "$LOGDIR/master.log"
