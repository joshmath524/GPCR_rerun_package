#!/usr/bin/env python3
"""Optuna LightGBM search on curated-warhead feature matrix (dev block only)."""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warhead_optuna_common import SEED, get_dev_block, make_folds, run_study  # noqa: E402

MODEL_NAME = "lightgbm_warhead_curated"
N_FOLDS = 3
MAX_TREES = 3000
EARLY_STOPPING_ROUNDS = 50

parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=100)
args = parser.parse_args()

X_dev, y_dev, smiles_dev, class_names, fingerprint = get_dev_block()
folds = make_folds(y_dev, smiles_dev, N_FOLDS)

FIXED_PARAMS = {
    "objective": "multiclass",
    "num_class": len(class_names),
    "max_depth": -1,
    "subsample_freq": 1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def suggest_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
    }


def finalize_params(trial):
    params = dict(trial.params)
    params.update(FIXED_PARAMS)
    params["n_estimators"] = trial.user_attrs["n_estimators"]
    return params


def objective(trial):
    params = suggest_params(trial)
    fold_scores, best_iters = [], []
    for fold in folds:
        clf = lgb.LGBMClassifier(**params, **FIXED_PARAMS, n_estimators=MAX_TREES)
        clf.fit(
            X_dev.iloc[fold["fit"]],
            y_dev[fold["fit"]],
            eval_set=[(X_dev.iloc[fold["early_stop"]], y_dev[fold["early_stop"]])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        preds = clf.predict(X_dev.iloc[fold["val"]])
        fold_scores.append(f1_score(y_dev[fold["val"]], preds, average="macro", zero_division=0))
        best_iters.append(clf.best_iteration_ or MAX_TREES)
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    trial.set_user_attr("std_macro_F1", std_f1)
    trial.set_user_attr("n_estimators", int(round(np.mean(best_iters))))
    print(
        f"Trial {trial.number:03d} | F1={mean_f1:.4f}±{std_f1:.4f} | trees={best_iters} | "
        f"lr={params['learning_rate']:.3f}, leaves={params['num_leaves']}",
        flush=True,
    )
    return mean_f1


run_study(MODEL_NAME, objective, fingerprint, args.n_trials, finalize_params)
print("Warhead LightGBM Optuna complete.")
