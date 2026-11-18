GPCR ligand classification - rerun package
==========================================

Friend run (curated warhead Optuna + scaffold) — 2 steps
--------------------------------------------------------

  1. pip install -r requirements.txt
  2. python run_warhead_pipeline.py

That single script retunes RF → XGBoost → LightGBM (100 Optuna trials each on the
curated warhead features), then runs the three scaffold evaluations with those
best params. Logs go under ML_code/warhead_scaffold_eval/logs/.

Optional flags:
  python run_warhead_pipeline.py --n-trials 50
  python run_warhead_pipeline.py --skip-optuna      # scaffold only
  python run_warhead_pipeline.py --skip-scaffold    # Optuna only

Use Python 3.11+ (3.13 works with the pinned versions above).
On macOS, if LightGBM fails on OpenMP, install libomp (e.g. brew install libomp).


Full package layout (keep as-is; scripts find data relative to ML_code/)
----------------------------------------------------------------------

  ML_code/shared_utilities.py
  ML_code/evaluation_scripts/            RF / XGBoost / LightGBM: baseline, RS, scaffold, LORO
  ML_code/warhead_scaffold_eval/         curated warhead Optuna + scaffold scripts + parquet cache
  ML_code/stacking_manual_optuna_process.py
  ML_code/evaluate_stacking_ensemble_test_set.py
  <receptor>/<receptor>_{agonist_enriched,antagonist_enriched,non_active_compounds}_NEW.xlsx
  Josh_Receptor_Features/<receptor>/*_pocket_residues_with_conservation.csv
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

Results land in an outputs/ folder next to each script (scaffold scripts write
next to the script). A run stops with an error if descriptors are missing/blank,
columns disagree, a receptor folder is missing, or a saved 80/20 split no longer
matches the data.
