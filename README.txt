GPCR ligand classification - rerun package
==========================================

Warhead + ESMDynamic scaffold (compare to baseline) — 2 steps
------------------------------------------------------------

  1. pip install -r requirements.txt
  2. python run_warhead_esmd_scaffold.py

Uses Table-2 warhead_* counts + pocket-binned ESMDynamic features + warhead×ESMD
interactions, with the same fixed RF/XGB/LightGBM hyperparameters as baseline
scaffold (no Optuna retune). Summaries land in:

  ML_code/warhead_esmd_scaffold_eval/*_warhead_esmd_scaffold_mean_std.csv

RF smoke test (~2–5 hours):
  python run_warhead_esmd_scaffold.py --models rf --skip-shap

Optional:
  python run_warhead_esmd_scaffold.py --models rf
  python run_warhead_esmd_scaffold.py --rebuild-esmd-features
  python run_warhead_esmd_scaffold.py --skip-shap

Precomputed ESMDynamic receptor features are in
ML_code/warhead_esmd_scaffold_eval/cache/esmdynamic_receptor_features.csv.
Raw ESMDynamic contact maps (optional, large) live under ESMDynamic_68_receptors/.


Friend run (warhead Optuna + scaffold only) — 2 steps
-----------------------------------------------------

  1. pip install -r requirements.txt
  2. python run_warhead_pipeline.py

Optional flags:
  python run_warhead_pipeline.py --n-trials 50
  python run_warhead_pipeline.py --skip-optuna
  python run_warhead_pipeline.py --skip-scaffold

Use Python 3.11+ (3.13 works with the pinned versions).
On macOS, if LightGBM fails on OpenMP: brew install libomp.


Full package layout (keep as-is; scripts find data relative to ML_code/)
----------------------------------------------------------------------

  ML_code/shared_utilities.py
  ML_code/evaluation_scripts/              baseline / RS / scaffold / LORO
  ML_code/warhead_scaffold_eval/           curated warhead cache + Optuna/scaffold
  ML_code/warhead_esmd_scaffold_eval/      warhead + ESMDynamic scaffold
  ESMDynamic_68_receptors/                 optional raw ESMDynamic inputs/outputs
  Josh_Receptor_Features/
  MLcodes/pan_gpcr_results/splits/
  MLcodes/optuna_tuning/


Other evaluation scripts (optional)
-----------------------------------

  python ML_code/evaluation_scripts/lightgbm_baseline_eval.py
  python ML_code/evaluation_scripts/rf_baseline_eval.py
  python ML_code/evaluation_scripts/xgb_baseline_eval.py
  python ML_code/evaluation_scripts/*_random_stratified_10seeds.py
  python ML_code/evaluation_scripts/*_scaffold_eval.py
  python ML_code/evaluation_scripts/*_loro_eval.py
  python ML_code/stacking_manual_optuna_process.py
  python ML_code/evaluate_stacking_ensemble_test_set.py
