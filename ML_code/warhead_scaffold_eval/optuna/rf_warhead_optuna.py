#!/usr/bin/env python3
"""Optuna RF search on curated-warhead feature matrix (dev block only)."""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warhead_optuna_common import SEED, get_dev_block, make_folds, run_study  # noqa: E402

MODEL_NAME = "rf_warhead_curated"
N_FOLDS = 5

parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", type=int, default=100)
args = parser.parse_args()

X_dev, y_dev, smiles_dev, class_names, fingerprint = get_dev_block()
folds = make_folds(y_dev, smiles_dev, N_FOLDS)


def suggest_params(trial):
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 25, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        "bootstrap": bootstrap,
    }
    if bootstrap:
        params["max_samples"] = trial.suggest_float("max_samples", 0.6, 1.0)
    return params


def finalize_params(trial):
    params = dict(trial.params)
    params.update({"random_state": SEED, "n_jobs": -1, "oob_score": False})
    return params


def objective(trial):
    params = suggest_params(trial)
    fold_scores = []
    for fold in folds:
        clf = RandomForestClassifier(**params, random_state=SEED, n_jobs=-1, oob_score=False)
        clf.fit(X_dev.iloc[fold["train"]], y_dev[fold["train"]])
        preds = clf.predict(X_dev.iloc[fold["val"]])
        fold_scores.append(f1_score(y_dev[fold["val"]], preds, average="macro", zero_division=0))
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    trial.set_user_attr("std_macro_F1", std_f1)
    print(
        f"Trial {trial.number:03d} | F1={mean_f1:.4f}±{std_f1:.4f} | "
        f"n_est={params['n_estimators']}, depth={params['max_depth']}, "
        f"max_feat={params['max_features']}, criterion={params['criterion']}, "
        f"class_weight={params['class_weight']}, bootstrap={params['bootstrap']}",
        flush=True,
    )
    return mean_f1


run_study(MODEL_NAME, objective, fingerprint, args.n_trials, finalize_params)
print("Warhead RF Optuna complete.")
