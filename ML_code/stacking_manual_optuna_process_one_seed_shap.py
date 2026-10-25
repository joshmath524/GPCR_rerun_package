#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-seed SHAP generator for manual stacking (independent ligand validation split).

This script is a lightweight copy of `stacking_manual_optuna_process.py` that:
- Loads the SAME enriched features + global independent ligand split
- Runs manual stacking for ONE seed (default: 42) with StratifiedGroupKFold by canonical SMILES
- Trains the logistic-regression meta-model
- Computes and saves SHAP values for the meta-model (LinearExplainer)

Outputs (under ML_code/outputs/shap/):
- stacking_independent_one_seed_shap_values.npz
- stacking_independent_one_seed_shap_meta_X_test_sample.csv
"""

import json
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="No further splits with positive gain")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
import shap

import sys

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from shared_utilities import (  # noqa: E402
    load_receptor_ligand_data,
    aggregate_receptor_features,
    build_interaction_terms,
    canonicalize_smiles_series,
    encode_labels,
    load_or_create_global_split,
)


N_CLASSES = 3
N_FOLDS = 5
SEED = 42  # single seed only (change if desired)
OUTPUT_DIR = _script_dir / "outputs"
SHAP_DIR = OUTPUT_DIR / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

RECEPTORS = [
    "5-HT1A",
    "5-HT1B",
    "5-HT1D",
    "5-HT1F",
    "5-HT2A",
    "5-HT2B",
    "5-HT2C",
    "5-HT4",
    "5-HT5A",
    "5-HT6",
    "5-HT7",
    "A1",
    "A2A",
    "A2B",
    "A3",
    "alpha1A",
    "alpha1B",
    "alpha2C",
    "beta1",
    "beta2",
    "beta3",
    "BLT1",
    "CB1",
    "CB2",
    "CysLT1",
    "CysLT2",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "EP2",
    "EP3",
    "EP4",
    "FFA1",
    "FFA2",
    "FFA3",
    "FFA4",
    "FP",
    "GPBA",
    "GPR119",
    "GPR139",
    "GPR174",
    "GPR35",
    "H1",
    "H2",
    "H3",
    "H4",
    "HCA2",
    "IP",
    "LPA1",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "MT1",
    "MT2",
    "P2Y1",
    "P2Y12",
    "PAF",
    "S1P1",
    "S1P2",
    "S1P3",
    "S1P5",
    "succinate",
    "TA1",
    "TP",
]


def _sanitize_feature_name(name):
    sanitized = re.sub(r"[^\w\-.]", "_", str(name))
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized if sanitized else "feature_" + str(hash(name) % 10000)


def main():
    print("=" * 80)
    print("MANUAL STACKING (INDEPENDENT SPLIT) - ONE SEED SHAP GENERATOR")
    print("Seed: %d | Folds: %d" % (SEED, N_FOLDS))
    print("Outputs:")
    print("  - %s" % (SHAP_DIR / "stacking_independent_one_seed_shap_values.npz",))
    print("  - %s" % (SHAP_DIR / "stacking_independent_one_seed_shap_meta_X_test_sample.csv",))
    print("=" * 80)

    # ---------------------------------------------------------------------
    # 1) Fixed best-trial hyperparameters (same as stacking_manual_optuna_process.py)
    # ---------------------------------------------------------------------
    lgb_params = {
        "learning_rate": 0.022675095411901637,
        "num_leaves": 278,
        "max_depth": -1,
        "min_child_samples": 81,
        "subsample": 0.8458649196412246,
        "colsample_bytree": 0.9888514643504509,
        "reg_alpha": 1.8080951629101958,
        "reg_lambda": 0.8806303532495801,
        "n_estimators": 300,
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "n_jobs": -1,
        "verbose": -1,
    }
    rf_params = {
        "bootstrap": False,
        "criterion": "entropy",
        "max_depth": 30,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 9,
        "n_estimators": 300,
        "n_jobs": -1,
    }
    xgb_params = {
        "learning_rate": 0.011729508579500059,
        "max_depth": 10,
        "min_child_weight": 1.0084867831102238,
        "subsample": 0.9804440329670632,
        "colsample_bytree": 0.7518705540721746,
        "gamma": 0.2959731631374173,
        "reg_alpha": 1.5981756177121895,
        "reg_lambda": 1.8565747985457355,
        "n_estimators": 300,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "verbosity": 0,
    }

    # ---------------------------------------------------------------------
    # 2) Load data and build enriched features (shared_utilities, same as baseline)
    # ---------------------------------------------------------------------
    workspace_root = _script_dir.parent

    print("\n[1/4] Aggregating receptor features...")
    receptor_feats = {}
    for idx, rec in enumerate(RECEPTORS, 1):
        feats = aggregate_receptor_features(rec)
        if feats:
            receptor_feats[rec] = feats
        if idx % 20 == 0 or idx == len(RECEPTORS):
            print("  Processed %d/%d receptors..." % (idx, len(RECEPTORS)))
    receptor_df = pd.DataFrame(receptor_feats).T.fillna(0)
    if len(receptor_df) == 0:
        raise ValueError("No receptor features found.")
    receptor_df.index.name = "Receptor"

    print("\n[2/4] Loading ligand data...")
    data_df, ligand_cols = load_receptor_ligand_data(list(receptor_df.index), OUTPUT_DIR)

    receptor_df_reset = receptor_df.reset_index()
    merged = data_df.merge(receptor_df_reset, on="Receptor", how="left")
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)

    merged, interaction_cols = build_interaction_terms(merged, receptor_df)
    merged, smiles_series = canonicalize_smiles_series(merged)

    feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols
    feature_cols = [c for c in feature_cols if c in merged.columns]
    X_temp = (
        merged[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    X = X_temp.select_dtypes(include=[np.number])
    X.columns = [_sanitize_feature_name(c) for c in X.columns]

    y, _class_names_array, class_names = encode_labels(merged)
    class_names = np.array(class_names)
    print("  [OK] Samples: %d | Features: %d | Classes: %s" % (len(X), X.shape[1], list(class_names)))

    # ---------------------------------------------------------------------
    # 3) Load the global independent ligand split (same as baseline scripts)
    # ---------------------------------------------------------------------
    print("\n[3/4] Loading global independent ligand validation split...")
    split_dir = workspace_root / "MLcodes" / "pan_gpcr_results" / "splits"
    global_split_file = split_dir / "final_independent_ligand_80_20_canonical_smiles.json"
    if not global_split_file.exists():
        raise IOError(
            "Global split file not found: %s. Create it by running a baseline script once "
            "(load_only=False) to generate the split." % global_split_file
        )

    train_idx, test_idx = load_or_create_global_split(
        X, y, smiles_series, split_dir, random_state=SEED, test_size=0.2, load_only=True
    )
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    X_80 = X.iloc[train_idx].reset_index(drop=True)
    y_80 = y[train_idx]
    X_test_20 = X.iloc[test_idx].reset_index(drop=True)
    y_test_20 = y[test_idx]

    dev80_smiles = set(smiles_series.iloc[train_idx].astype(str).values)
    indep20_smiles = set(smiles_series.iloc[test_idx].astype(str).values)
    overlap = dev80_smiles & indep20_smiles
    if overlap:
        raise ValueError(
            "CRITICAL: Found %d overlapping canonical SMILES between dev80 and independent20!"
            % len(overlap)
        )

    print("  [OK] dev80: %d | independent20: %d | SMILES overlap: 0" % (len(X_80), len(X_test_20)))

    # ---------------------------------------------------------------------
    # 4) Manual stacking for ONE seed, then SHAP on the meta-model
    # ---------------------------------------------------------------------
    print("\n[4/4] Running manual stacking (one seed) and computing SHAP...")

    meta_X_train = np.zeros((len(X_80), N_CLASSES * 3), dtype=float)
    meta_X_test_list = []

    groups_dev80 = np.asarray(smiles_series.iloc[train_idx].astype(str).values)
    assert len(groups_dev80) == len(X_80)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx_fold, val_idx_fold) in enumerate(sgkf.split(X_80, y_80, groups=groups_dev80), 1):
        print("  Fold %d/%d..." % (fold, N_FOLDS))

        X_train_fold = X_80.iloc[train_idx_fold].reset_index(drop=True)
        X_val_fold = X_80.iloc[val_idx_fold].reset_index(drop=True)
        y_train_fold = y_80[train_idx_fold]

        rf_params_seed = rf_params.copy()
        rf_params_seed["random_state"] = SEED
        rf_model = RandomForestClassifier(**rf_params_seed)
        rf_model.fit(X_train_fold, y_train_fold)
        rf_proba_val = rf_model.predict_proba(X_val_fold)
        rf_proba_test = rf_model.predict_proba(X_test_20)

        xgb_params_seed = xgb_params.copy()
        xgb_params_seed["random_state"] = SEED
        xgb_model = xgb.XGBClassifier(**xgb_params_seed)
        xgb_model.fit(X_train_fold, y_train_fold)
        xgb_proba_val = xgb_model.predict_proba(X_val_fold)
        xgb_proba_test = xgb_model.predict_proba(X_test_20)

        lgb_params_seed = lgb_params.copy()
        lgb_params_seed["random_state"] = SEED
        lgb_model = lgb.LGBMClassifier(**lgb_params_seed)
        lgb_model.fit(X_train_fold, y_train_fold)
        lgb_proba_val = lgb_model.predict_proba(X_val_fold)
        lgb_proba_test = lgb_model.predict_proba(X_test_20)

        meta_X_train[val_idx_fold] = np.hstack([rf_proba_val, xgb_proba_val, lgb_proba_val])
        meta_X_test_list.append(np.hstack([rf_proba_test, xgb_proba_test, lgb_proba_test]))

    meta_X_test = np.mean(meta_X_test_list, axis=0)

    meta_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )
    meta_model.fit(meta_X_train, y_80)

    # Quick sanity metric (optional, but useful to confirm run worked)
    y_proba_meta = meta_model.predict_proba(meta_X_test)
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    lb.fit(range(N_CLASSES))
    y_test_bin = lb.transform(y_test_20)
    auc_macro = roc_auc_score(y_test_bin, y_proba_meta, average="macro", multi_class="ovr")
    print("  [OK] Meta-model fit complete. independent20 macro-AUC (seed=%d): %.4f" % (SEED, auc_macro))

    # SHAP (LinearExplainer) on the meta-model
    n_shap_samples = min(500, len(meta_X_test))
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(meta_X_test), n_shap_samples, replace=False)
    meta_X_sample = meta_X_test[sample_idx]

    meta_feature_names = []
    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        for cls in class_names:
            meta_feature_names.append("%s_%s" % (model_name, cls))

    explainer = shap.LinearExplainer(meta_model, meta_X_test)
    shap_values = explainer.shap_values(meta_X_sample)
    shap_array = np.array(shap_values) if isinstance(shap_values, list) else np.asarray(shap_values)

    expected_value = explainer.expected_value
    expected_value_array = np.array(expected_value) if isinstance(expected_value, (list, np.ndarray)) else np.array([expected_value])

    pd.DataFrame(meta_X_sample, columns=meta_feature_names).to_csv(
        SHAP_DIR / "stacking_independent_one_seed_shap_meta_X_test_sample.csv", index=False
    )
    np.savez(
        SHAP_DIR / "stacking_independent_one_seed_shap_values.npz",
        shap_values=shap_array,
        expected_value=expected_value_array,
        sample_indices=sample_idx,
        feature_names=np.array(meta_feature_names),
        class_names=np.array(class_names),
        seed_used=np.array([SEED]),
        split_file=str((split_dir / "final_independent_ligand_80_20_canonical_smiles.json").as_posix()),
    )

    print("\n[SUCCESS] SHAP saved.")
    print("  - %s" % (SHAP_DIR / "stacking_independent_one_seed_shap_values.npz",))
    print("  - %s" % (SHAP_DIR / "stacking_independent_one_seed_shap_meta_X_test_sample.csv",))


if __name__ == "__main__":
    main()

