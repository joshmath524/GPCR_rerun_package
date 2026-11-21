#!/usr/bin/env python3
"""Train one Random Forest per split (seed 42 only) and save TreeSHAP for manuscript figures.

Does not run Optuna, 10-seed loops, or CV. Uses the cleaned matrix from shared_utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent))
from shared_utilities import (
    aggregate_receptor_features,
    build_interaction_terms,
    canonicalize_smiles_series,
    encode_labels,
    load_or_create_global_split,
    load_receptor_ligand_data,
)

WORKSPACE_ROOT = _script_dir.parent.parent
OUTPUT_DIR = _script_dir / "outputs"
SHAP_DIR = OUTPUT_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR = WORKSPACE_ROOT / "MLcodes" / "pan_gpcr_results" / "splits"

SEED = 42
N_SHAP = 500
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 30,
    "min_samples_split": 9,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "entropy",
    "bootstrap": False,
    "class_weight": None,
    "random_state": SEED,
    "n_jobs": -1,
    "oob_score": False,
}

RECEPTORS = [
    "5-HT1A", "5-HT1B", "5-HT1D", "5-HT1F", "5-HT2A", "5-HT2B", "5-HT2C",
    "5-HT4", "5-HT5A", "5-HT6", "5-HT7", "A1", "A2A", "A2B", "A3",
    "alpha1A", "alpha1B", "alpha2C", "beta1", "beta2", "beta3",
    "BLT1", "CB1", "CB2", "CysLT1", "CysLT2", "D1", "D2", "D3", "D4", "D5",
    "EP2", "EP3", "EP4", "FFA1", "FFA2", "FFA3", "FFA4", "FP",
    "GPBA", "GPR119", "GPR139", "GPR174", "GPR35", "H1", "H2", "H3", "H4", "HCA2", "IP",
    "LPA1", "M1", "M2", "M3", "M4", "M5", "MT1", "MT2", "P2Y1", "P2Y12", "PAF",
    "S1P1", "S1P2", "S1P3", "S1P5", "succinate", "TA1", "TP",
]


def mean_abs_per_feature(shap_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(shap_array, dtype=float)
    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SHAP ndim={arr.ndim}, shape={arr.shape}")
    # (C, N, F) vs (N, F, C)
    if arr.shape[0] <= 10 and arr.shape[0] < arr.shape[1]:
        return np.abs(arr).mean(axis=(0, 1))
    return np.abs(arr).mean(axis=(0, 2))


def save_ranked_csv(shap_array: np.ndarray, feature_names, out_path: Path) -> pd.DataFrame:
    scores = mean_abs_per_feature(shap_array)
    ranked = (
        pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": scores})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(out_path, index=False)
    return ranked


def normalize_shap_array(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.array(shap_values)
    return np.asarray(shap_values)


def print_top10(ranked: pd.DataFrame, shap_array: np.ndarray, label: str) -> None:
    print(f"\n[{label}] SHAP array shape: {np.asarray(shap_array).shape}", flush=True)
    print(f"[{label}] Top 10 features by mean |SHAP|:", flush=True)
    for _, row in ranked.head(10).iterrows():
        print(f"  {int(row['rank']):2d}. {row['feature']}: {row['mean_abs_shap']:.6f}", flush=True)


def load_matrix():
    print("Aggregating receptor features...", flush=True)
    receptor_feats = {}
    for rec in RECEPTORS:
        feats = aggregate_receptor_features(rec)
        if feats:
            receptor_feats[rec] = feats
    receptor_df = pd.DataFrame(receptor_feats).T.fillna(0)
    receptor_df.index.name = "Receptor"
    print(f"  [OK] {len(receptor_df)} receptors", flush=True)

    print("Loading cleaned ligand data...", flush=True)
    data_df, ligand_cols = load_receptor_ligand_data(list(receptor_df.index), OUTPUT_DIR)
    print(f"  [OK] {len(data_df):,} rows after cleaning", flush=True)
    if len(data_df) != 24937:
        print(f"  [WARNING] Expected 24,937 cleaned rows, got {len(data_df):,}", flush=True)

    merged = data_df.merge(receptor_df.reset_index(), on="Receptor", how="left")
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
    merged, interaction_cols = build_interaction_terms(merged, receptor_df)
    merged, smiles_series = canonicalize_smiles_series(merged)

    feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols
    feature_cols = [c for c in feature_cols if c in merged.columns]
    X = (
        merged[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .select_dtypes(include=[np.number])
    )
    y, _, class_names = encode_labels(merged)
    class_names = np.array(class_names)
    print(f"  [OK] X={X.shape}, classes={class_names.tolist()}", flush=True)
    return X, y, smiles_series, class_names


def fit_and_shap(X_train, y_train, X_test, class_names, prefix: str, label: str):
    print(f"\n=== {label} ===", flush=True)
    print(f"  Train {len(X_train):,} | Test pool {len(X_test):,}", flush=True)
    model = RandomForestClassifier(**RF_PARAMS)
    print("  Fitting Random Forest (seed 42, 300 trees)...", flush=True)
    model.fit(X_train, y_train)

    n = min(N_SHAP, len(X_test))
    sample_idx = np.random.RandomState(42).choice(len(X_test), n, replace=False)
    X_sample = X_test.iloc[sample_idx].copy()
    print(f"  Computing TreeSHAP on {n} rows...", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap_array = normalize_shap_array(shap_values)
    expected = explainer.expected_value
    expected_array = np.asarray(expected) if isinstance(expected, (list, np.ndarray)) else np.array([expected])

    X_sample.to_csv(SHAP_DIR / f"{prefix}_shap_X_test_sample.csv", index=False)
    np.savez_compressed(
        SHAP_DIR / f"{prefix}_shap_values.npz",
        shap_values=shap_array,
        expected_value=expected_array,
        sample_indices=sample_idx,
        feature_names=np.array(X_sample.columns.astype(str)),
        class_names=np.array(class_names),
        seed_used=np.array([SEED]),
    )
    ranked = save_ranked_csv(
        shap_array,
        X_sample.columns.astype(str),
        SHAP_DIR / f"{prefix}_shap_mean_abs_ranked.csv",
    )
    print(f"  [OK] Wrote {SHAP_DIR / f'{prefix}_shap_values.npz'}", flush=True)
    print(f"  [OK] Wrote {SHAP_DIR / f'{prefix}_shap_X_test_sample.csv'}", flush=True)
    print(f"  [OK] Wrote {SHAP_DIR / f'{prefix}_shap_mean_abs_ranked.csv'}", flush=True)
    print_top10(ranked, shap_array, label)


def main():
    print("=" * 80, flush=True)
    print("Missing Random Forest SHAP only (independent + random stratified)", flush=True)
    print("=" * 80, flush=True)
    X, y, smiles_series, class_names = load_matrix()

    # 1) Independent-ligand: full dev80 fit, SHAP on independent20
    print("\nLoading independent-ligand split...", flush=True)
    split_file = SPLIT_DIR / "final_independent_ligand_80_20_canonical_smiles.json"
    if not split_file.exists():
        raise FileNotFoundError(split_file)
    train_idx, test_idx = load_or_create_global_split(
        X, y, smiles_series, SPLIT_DIR, random_state=SEED, test_size=0.2, load_only=True
    )
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    X_80 = X.iloc[train_idx].reset_index(drop=True)
    y_80 = y[train_idx]
    X_20 = X.iloc[test_idx].reset_index(drop=True)
    overlap = set(smiles_series.iloc[train_idx].astype(str)) & set(smiles_series.iloc[test_idx].astype(str))
    if overlap:
        raise ValueError(f"SMILES overlap between splits: {len(overlap)}")
    print(f"  [OK] dev80={len(X_80):,} independent20={len(X_20):,} overlap=0", flush=True)
    fit_and_shap(X_80, y_80, X_20, class_names, "rf_baseline", "RF independent-ligand")

    # 2) Random stratified: one split seed 42
    print("\nBuilding random stratified split (test_size=0.2, seed=42)...", flush=True)
    indices = np.arange(len(X))
    tr, te = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
    X_tr = X.iloc[tr].reset_index(drop=True)
    y_tr = y[tr]
    X_te = X.iloc[te].reset_index(drop=True)
    print(f"  [OK] train={len(X_tr):,} test={len(X_te):,}", flush=True)
    fit_and_shap(X_tr, y_tr, X_te, class_names, "rf_random_stratified", "RF random-stratified")

    print("\n[DONE] Both Random Forest SHAP jobs finished.", flush=True)


if __name__ == "__main__":
    main()
