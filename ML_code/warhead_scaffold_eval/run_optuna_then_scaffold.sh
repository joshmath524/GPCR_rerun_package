#!/usr/bin/env bash
# 1) Optuna retune RF/XGB/LightGBM on curated warhead features (100 trials each)
# 2) Then scaffold eval for all three using the Optuna best_params JSON files
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export DYLD_LIBRARY_PATH="$ROOT/.venv/lib/python3.13/site-packages/sklearn/.dylibs:${DYLD_LIBRARY_PATH:-}"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/ML_code/warhead_scaffold_eval"
LOG="$OUT/logs"
N_TRIALS="${1:-100}"
mkdir -p "$LOG" "$OUT/optuna/logs"

echo "===== PHASE 1: OPTUNA ($N_TRIALS trials each) $(date) =====" | tee -a "$LOG/full_pipeline.log"
bash "$OUT/optuna/run_all_warhead_optuna.sh" "$N_TRIALS"

echo "===== PHASE 2: SCAFFOLD EVALS (curated warhead + Optuna params) $(date) =====" | tee -a "$LOG/full_pipeline.log"
bash "$OUT/run_all_curated_warhead_scaffold.sh"

echo "===== ALL DONE $(date) =====" | tee -a "$LOG/full_pipeline.log"
echo "Results:"
echo "  Optuna best params: $OUT/optuna/*_best_params.json"
echo "  Scaffold summaries: $OUT/*_scaffold_curated_wh_mean_std.csv"
