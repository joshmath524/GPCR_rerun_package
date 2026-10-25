GPCR ligand classification - rerun package
==========================================

Folder layout (keep it exactly like this; the scripts find the data relative to ML_code/):

  ML_code/shared_utilities.py            data loading, validation, receptor features, split logic
  ML_code/evaluation_scripts/            RF / XGBoost / LightGBM: baseline, random stratified, scaffold, LORO
  ML_code/stacking_manual_optuna_process.py              stacking ensemble, global 80/20 split
  ML_code/stacking_manual_optuna_process_one_seed_shap.py stacking ensemble SHAP, global 80/20 split
  ML_code/evaluate_stacking_ensemble_test_set.py         stacking ensemble, random stratified
  <receptor>/<receptor>_{agonist_enriched,antagonist_enriched,non_active_compounds}_NEW.xlsx
                                         68 receptors, 193 files, descriptors already computed
  Josh_Receptor_Features/<receptor>/*_pocket_residues_with_conservation.csv
  MLcodes/pan_gpcr_results/splits/       the shared 80/20 split (built from this exact data)
  MLcodes/optuna_tuning/pan_gpcr_*_optuna.py  hyperparameter tuning (RF, XGBoost, LightGBM)
  MLcodes/optuna_tuning/optuna_common.py      shared tuning setup (data, split, resume, outputs)

Setup (Python 3.11+):

  pip install -r requirements.txt

Hyperparameter tuning (optional - only if re-tuning):

  python MLcodes/optuna_tuning/pan_gpcr_lightgbm_optuna.py
  python MLcodes/optuna_tuning/pan_gpcr_xgb_optuna.py
  python MLcodes/optuna_tuning/pan_gpcr_rf_optuna.py

  Each runs 500 trials on the 80% dev block only (add --n-trials N to change). If a run stops or
  crashes, run the same command again and it continues from the last finished trial. Results go to
  MLcodes/pan_gpcr_results/optuna/ (<model>_optuna_trials.csv, <model>_best_params.json) and are
  updated after every trial. The first run builds a cached feature matrix (about an hour of reading
  Excel files); later runs start in a few minutes.

Run order (from this folder):

  1. One baseline script first (it reuses the split in MLcodes/pan_gpcr_results/splits):
       python ML_code/evaluation_scripts/lightgbm_baseline_eval.py
  2. The other baseline scripts:
       python ML_code/evaluation_scripts/rf_baseline_eval.py
       python ML_code/evaluation_scripts/xgb_baseline_eval.py
  3. Any of the remaining scripts, in any order:
       ML_code/evaluation_scripts/*_random_stratified_10seeds.py
       ML_code/evaluation_scripts/*_scaffold_eval.py
       ML_code/evaluation_scripts/*_loro_eval.py
       ML_code/stacking_manual_optuna_process.py               (needs the split from step 1)
       ML_code/stacking_manual_optuna_process_one_seed_shap.py (needs the split from step 1)
       ML_code/evaluate_stacking_ensemble_test_set.py

Results are written to an outputs/ folder next to each script (scaffold scripts write next to the
script itself). Each run also writes <script>_data_manifest.csv listing every data file and
how many compounds it contributed.

Safety checks: a run stops with an error (instead of silently producing wrong results) if a
data file has missing/blank descriptors, files have different descriptor columns, a receptor
folder is missing, or a saved 80/20 split no longer matches the data. If a split error appears,
delete MLcodes/pan_gpcr_results/splits/*.json and rerun step 1.
