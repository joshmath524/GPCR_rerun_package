#!/usr/bin/env bash
# Wait for RF warhead scaffold PID (or rf.log DONE), then run XGB + LightGBM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export DYLD_LIBRARY_PATH="$ROOT/.venv/lib/python3.13/site-packages/sklearn/.dylibs:${DYLD_LIBRARY_PATH:-}"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/ML_code/warhead_scaffold_eval"
LOG="$OUT/logs"

RF_PID="${1:-}"
if [[ -n "$RF_PID" ]]; then
  echo "Waiting for RF PID $RF_PID ..."
  while kill -0 "$RF_PID" 2>/dev/null; do sleep 60; done
fi

echo "===== START xgb $(date) =====" | tee -a "$LOG/master.log"
"$PY" -u "$OUT/xgb_scaffold_eval_warhead.py" 2>&1 | tee "$LOG/xgb.log"
echo "===== DONE xgb $(date) =====" | tee -a "$LOG/master.log"

echo "===== START lightgbm $(date) =====" | tee -a "$LOG/master.log"
"$PY" -u "$OUT/lightgbm_scaffold_eval_warhead.py" 2>&1 | tee "$LOG/lgb.log"
echo "===== DONE lightgbm $(date) =====" | tee -a "$LOG/master.log"
