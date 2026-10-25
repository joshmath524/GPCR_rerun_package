#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for XGBoost (multiclass macro F1)

- Data: validated _NEW files only, same features as the evaluation scripts (see optuna_common.py)
- Uses ONLY the 80% dev block of the global split (all of it); the 20% test block is never used
- 3-fold StratifiedGroupKFold CV on the dev block (groups = canonical SMILES)
- Number of trees: early stopping (50 rounds, mlogloss) on a grouped 10% subset of each fold's
  training part, never on the fold being scored; the mean best iteration is saved as n_estimators
- Models are trained without sample weights, exactly as in the evaluation scripts
- Mean macro-F1 across folds as objective, TPESampler(seed=42), 500 trials by default
- Resumable: rerun the same command after a crash or stop
- Outputs in MLcodes/pan_gpcr_results/optuna/: xgb_optuna_trials.csv, xgb_best_params.json, xgb_optuna.db

Usage (from anywhere):
    python MLcodes/optuna_tuning/pan_gpcr_xgb_optuna.py [--n-trials 500]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optuna_common import SEED, get_dev_block, make_folds, run_study  # noqa: E402

MODEL_NAME = 'xgb'
N_FOLDS = 3
MAX_TREES = 3000
EARLY_STOPPING_ROUNDS = 50

parser = argparse.ArgumentParser()
parser.add_argument('--n-trials', type=int, default=500)
args = parser.parse_args()

X_dev, y_dev, smiles_dev, class_names, fingerprint = get_dev_block()
folds = make_folds(y_dev, smiles_dev, N_FOLDS)

FIXED_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': len(class_names),
    'tree_method': 'hist',
    'random_state': SEED,
    'n_jobs': -1,
}


def suggest_params(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
    }


def finalize_params(trial):
    """Parameters to use in the evaluation scripts (no early stopping there)."""
    params = dict(trial.params)
    params.update(FIXED_PARAMS)
    params['n_estimators'] = trial.user_attrs['n_estimators']
    return params


def objective(trial):
    params = suggest_params(trial)
    fold_scores, best_iters = [], []
    for fold in folds:
        clf = xgb.XGBClassifier(
            **params, **FIXED_PARAMS,
            n_estimators=MAX_TREES,
            eval_metric='mlogloss',
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        clf.fit(
            X_dev.iloc[fold['fit']], y_dev[fold['fit']],
            eval_set=[(X_dev.iloc[fold['early_stop']], y_dev[fold['early_stop']])],
            verbose=False,
        )
        preds = clf.predict(X_dev.iloc[fold['val']])
        fold_scores.append(f1_score(y_dev[fold['val']], preds, average='macro', zero_division=0))
        best_iters.append(clf.best_iteration + 1)

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    trial.set_user_attr('std_macro_F1', std_f1)
    trial.set_user_attr('n_estimators', int(round(np.mean(best_iters))))
    print(
        f"Trial {trial.number:03d} | F1={mean_f1:.4f}Â±{std_f1:.4f} | trees={best_iters} | "
        f"lr={params['learning_rate']:.3f}, depth={params['max_depth']}, mcw={params['min_child_weight']:.2f}, "
        f"subsample={params['subsample']:.3f}, colsample={params['colsample_bytree']:.3f}, gamma={params['gamma']:.3f}, "
        f"alpha={params['reg_alpha']:.3f}, lambda={params['reg_lambda']:.3f}",
        flush=True,
    )
    return mean_f1


run_study(MODEL_NAME, objective, fingerprint, args.n_trials, finalize_params)
print("Optuna XGBoost search complete.")
