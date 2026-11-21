#!/usr/bin/env python3
"""
Scaffold-split evaluation with curated warhead + ESMDynamic contact features.

Uses the same fixed baseline hyperparameters as the non-warhead / warhead
scaffold scripts (no Optuna retune) for a fair comparison to baseline.

  python scaffold_eval_warhead_esmd.py --model rf
  python scaffold_eval_warhead_esmd.py --model xgb
  python scaffold_eval_warhead_esmd.py --model lgb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

HERE = Path(__file__).resolve().parent
ML = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ML))

from prepare_warhead_esmd_data import build_warhead_esmd_dataset  # noqa: E402
from shared_utilities import align_proba  # noqa: E402

SEEDS = list(range(42, 52))
N_FOLDS = 5
N_CLASSES = 3
OUTPUT_DIR = HERE

RF_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 30,
    "min_samples_split": 9,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "entropy",
    "bootstrap": False,
    "class_weight": None,
    "n_jobs": -1,
    "oob_score": False,
}

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.011729508579500059,
    "max_depth": 10,
    "min_child_weight": 1.0084867831102238,
    "subsample": 0.9804440329670632,
    "colsample_bytree": 0.7518705540721746,
    "gamma": 0.2959731631374173,
    "reg_alpha": 1.5981756177121895,
    "reg_lambda": 1.8565747985457355,
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "tree_method": "hist",
}

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "None",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.022675095411901637,
    "num_leaves": 278,
    "max_depth": -1,
    "min_child_samples": 81,
    "subsample": 0.8458649196412246,
    "colsample_bytree": 0.9888514643504509,
    "lambda_l1": 1.8080951629101958,
    "lambda_l2": 0.8806303532495801,
    "n_jobs": -1,
}


def compute_scaffold(smiles: str, fallback: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return f"NO_SCAFFOLD_{fallback}"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def compute_macro_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specs = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))


def evaluate_model(y_true, y_pred, y_proba, n_classes):
    metrics = {
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity": compute_macro_specificity(y_true, y_pred, n_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    try:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        metrics["auc"] = float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))
    except Exception:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc"] = float("nan")
    return metrics


def make_model(model_name: str, seed: int):
    if model_name == "rf":
        params = dict(RF_PARAMS)
        params["random_state"] = seed
        return RandomForestClassifier(**params)
    if model_name == "xgb":
        import xgboost as xgb

        params = dict(XGB_PARAMS)
        params["random_state"] = seed
        return xgb.XGBClassifier(**params)
    if model_name == "lgb":
        import lightgbm as lgb

        params = dict(LGB_PARAMS)
        params["random_state"] = seed
        return lgb.LGBMClassifier(**params)
    raise ValueError(model_name)


def predict_proba_safe(model, model_name, X, n_classes):
    proba = model.predict_proba(X)
    if model_name in {"xgb", "lgb"}:
        proba = align_proba(proba, getattr(model, "classes_", np.arange(n_classes)), n_classes)
    return np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb", "lgb"], required=True)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    model_name = args.model
    tag = {"rf": "rf", "xgb": "xgb", "lgb": "lightgbm"}[model_name]

    print("=" * 80)
    print(f"SCAFFOLD EVAL — warhead + ESMDynamic — model={model_name}")
    print("=" * 80)
    print("Hyperparameters: fixed baseline (no Optuna retune)", flush=True)

    print("\n[STEP 1] Building feature matrix...", flush=True)
    X, y, smiles_series, class_names = build_warhead_esmd_dataset()
    n_classes = len(class_names)
    final_feature_cols = list(X.columns)
    print(f"  samples={len(X):,} features={X.shape[1]} classes={class_names}", flush=True)

    print("\n[STEP 2] Computing Bemis–Murcko scaffolds...", flush=True)
    scaffolds = []
    total = len(smiles_series)
    for idx, smi in enumerate(smiles_series.tolist(), 1):
        if idx % 5000 == 0 or idx == total:
            print(f"  {idx:,}/{total:,}", flush=True)
        scaffolds.append(compute_scaffold(smi, str(idx)))
    scaffold_series = pd.Series(scaffolds, index=X.index)
    unique_scaffolds = scaffold_series.unique()
    print(f"  unique scaffolds={len(unique_scaffolds):,}", flush=True)

    print("\n[STEP 3] 10-seed scaffold CV + test...", flush=True)
    all_results_cv, all_results_test, all_predictions = [], [], []
    first_seed_model = first_seed_X_test = None

    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\nSeed {seed_idx}/{len(SEEDS)} (seed={seed})...", flush=True)
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(unique_scaffolds)
        train_scaffolds = set()
        train_count = 0
        train_target = int(0.8 * len(X))
        for sc in shuffled:
            n = int((scaffold_series == sc).sum())
            if train_count < train_target:
                train_scaffolds.add(sc)
                train_count += n
        train_mask = scaffold_series.isin(train_scaffolds)
        test_mask = ~train_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print(
            f"  Split: train {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%), "
            f"test {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)",
            flush=True,
        )

        groups_train = pd.Categorical(scaffold_series[train_mask]).codes
        splitter = (
            StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False)
            if StratifiedGroupKFold is not None
            else GroupKFold(n_splits=N_FOLDS)
        )
        cv_results = []
        for fold, (tr, va) in enumerate(splitter.split(X_train, y_train, groups=groups_train), 1):
            print(f"    Fold {fold}/{N_FOLDS}...", end=" ", flush=True)
            model = make_model(model_name, seed)
            model.fit(X_train.iloc[tr], np.asarray(y_train)[tr])
            y_va = np.asarray(y_train)[va]
            y_pred = model.predict(X_train.iloc[va])
            y_proba = predict_proba_safe(model, model_name, X_train.iloc[va], n_classes)
            metrics = evaluate_model(y_va, y_pred, y_proba, n_classes)
            cv_results.append(metrics)
            print(f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}", flush=True)

        cv_metrics = {
            m: float(np.nanmean([r[m] for r in cv_results]))
            for m in ["recall", "precision", "f1", "auc", "specificity", "accuracy"]
        }
        cv_metrics["seed"] = seed
        all_results_cv.append(cv_metrics)
        print(f"  CV F1={cv_metrics['f1']:.4f} AUC={cv_metrics['auc']:.4f}", flush=True)

        print(f"  Final test...", end=" ", flush=True)
        model = make_model(model_name, seed)
        model.fit(X_train, y_train)
        y_test_arr = np.asarray(y_test)
        y_pred_test = model.predict(X_test)
        y_proba_test = predict_proba_safe(model, model_name, X_test, n_classes)
        test_metrics = evaluate_model(y_test_arr, y_pred_test, y_proba_test, n_classes)
        test_metrics["seed"] = seed
        all_results_test.append(test_metrics)
        all_predictions.append(
            {
                "seed": seed,
                "y_true": y_test_arr.copy(),
                "y_pred": y_pred_test.copy(),
                "y_pred_proba": y_proba_test.copy(),
            }
        )
        if seed == SEEDS[0]:
            first_seed_model = model
            first_seed_X_test = X_test.copy()
        print(f"F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}", flush=True)

    print("\n[STEP 4] Saving summaries...", flush=True)
    cv_df = pd.DataFrame(all_results_cv)
    test_df = pd.DataFrame(all_results_test)
    metric_names = ["recall", "precision", "f1", "auc", "specificity", "accuracy"]
    summary = {}
    for metric in metric_names:
        summary[f"{metric}_scaffold_cv_mean"] = float(np.nanmean(cv_df[metric]))
        summary[f"{metric}_scaffold_cv_std"] = float(np.nanstd(cv_df[metric]))
        summary[f"{metric}_scaffold_test_mean"] = float(np.nanmean(test_df[metric]))
        summary[f"{metric}_scaffold_test_std"] = float(np.nanstd(test_df[metric]))
    summary_df = pd.DataFrame([summary])
    summary_df.index = [f"{tag}_warhead_esmd_scaffold"]

    cv_df.to_csv(OUTPUT_DIR / f"{tag}_warhead_esmd_scaffold_per_seed_cv.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / f"{tag}_warhead_esmd_scaffold_per_seed_test.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / f"{tag}_warhead_esmd_scaffold_mean_std.csv", index=True)

    test_sizes = np.array([len(p["y_true"]) for p in all_predictions])
    max_n = int(test_sizes.max())
    y_true_pad = np.full((len(all_predictions), max_n), -1, dtype=np.int32)
    y_pred_pad = np.full((len(all_predictions), max_n), -1, dtype=np.int32)
    y_proba_pad = np.full((len(all_predictions), max_n, n_classes), np.nan, dtype=np.float32)
    for i, p in enumerate(all_predictions):
        n = len(p["y_true"])
        y_true_pad[i, :n] = p["y_true"]
        y_pred_pad[i, :n] = p["y_pred"]
        y_proba_pad[i, :n, :] = p["y_pred_proba"]
    np.savez(
        OUTPUT_DIR / f"{tag}_warhead_esmd_scaffold_predictions.npz",
        seeds=np.array([p["seed"] for p in all_predictions]),
        test_sizes=test_sizes,
        y_true=y_true_pad,
        y_pred=y_pred_pad,
        y_pred_proba=y_proba_pad,
        class_names=np.array(class_names),
    )

    print("\nTest mean ± std:")
    for metric in metric_names:
        print(
            f"  {metric:12s}: {summary[f'{metric}_scaffold_test_mean']:.4f} "
            f"± {summary[f'{metric}_scaffold_test_std']:.4f}"
        )

    if not args.skip_shap and first_seed_model is not None:
        print("\n[STEP 5] SHAP (first seed)...", flush=True)
        try:
            n_shap = min(500, len(first_seed_X_test))
            sample_idx = np.random.RandomState(42).choice(len(first_seed_X_test), n_shap, replace=False)
            X_sample = first_seed_X_test.iloc[sample_idx]
            explainer = shap.TreeExplainer(first_seed_model)
            shap_vals = explainer.shap_values(X_sample)
            shap_dir = OUTPUT_DIR / "shap"
            shap_dir.mkdir(exist_ok=True)
            X_sample.to_csv(shap_dir / f"{tag}_warhead_esmd_shap_X_test_sample.csv", index=False)
            np.savez(
                shap_dir / f"{tag}_warhead_esmd_shap_values.npz",
                shap_values=np.array(shap_vals) if isinstance(shap_vals, list) else shap_vals,
                feature_names=np.array(final_feature_cols),
                class_names=np.array(class_names),
                seed_used=np.array([SEEDS[0]]),
            )
            print(f"  [OK] SHAP -> {shap_dir}", flush=True)
        except Exception as e:
            print(f"  [WARNING] SHAP failed: {e}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
