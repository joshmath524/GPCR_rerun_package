#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Leave-One-Receptor-Out (LORO) Evaluation.

For each receptor r: train on R \\ {r}, test on r. Repeat for SEEDS = [42, 43, 44].
Feature construction identical to baseline/scaffold: load_table (_NEW), aggregate_receptor_features,
merge, 14 interaction terms, numeric-only, inf→0, NaN→0. Label map: agonist=0, antagonist=1, inactive=2.
Uses fixed best-trial hyperparameters (Optuna trial 76), n_estimators=300.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_utilities import (
    load_receptor_ligand_data,
    aggregate_receptor_features,
    build_interaction_terms,
    should_exclude_column,
    encode_labels,
    align_proba,
)

# ============================================================================
# Configuration
# ============================================================================

SEEDS = [42, 43, 44]
N_MIN_TEST = 20
N_CLASSES = 3
MODEL_NAME = 'Random Forest'
RUN_SHAP = False  # Optional: set True to run SHAP for one receptor × one seed (after all saves)
OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECEPTORS = [
    '5-HT1A', '5-HT1B', '5-HT1D', '5-HT1F', '5-HT2A', '5-HT2B', '5-HT2C',
    '5-HT4', '5-HT5A', '5-HT6', '5-HT7', 'A1', 'A2A', 'A2B', 'A3',
    'alpha1A', 'alpha1B', 'alpha2C', 'beta1', 'beta2', 'beta3',
    'BLT1', 'CB1', 'CB2', 'CysLT1', 'CysLT2', 'D1', 'D2', 'D3', 'D4', 'D5',
    'EP2', 'EP3', 'EP4', 'FFA1', 'FFA2', 'FFA3', 'FFA4', 'FP',
    'GPBA', 'GPR119', 'GPR139', 'GPR174', 'GPR35', 'H1', 'H2', 'H3', 'H4', 'HCA2', 'IP',
    'LPA1', 'M1', 'M2', 'M3', 'M4', 'M5', 'MT1', 'MT2', 'P2Y1', 'P2Y12', 'PAF',
    'S1P1', 'S1P2', 'S1P3', 'S1P5', 'succinate', 'TA1', 'TP',
]

# Best-trial hyperparameters (Optuna trial 76); n_estimators=300
BEST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 30,
    'min_samples_split': 9,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'criterion': 'entropy',
    'bootstrap': False,
    'class_weight': None,
    'random_state': 42,
    'n_jobs': -1,
    'oob_score': False,
}


def compute_macro_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specs = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specs.append(spec)
    return float(np.mean(specs))


def evaluate_model(y_true, y_pred, y_proba, n_classes, class_names):
    """Macro and per-class metrics; AUC uses np.nan on failure."""
    metrics = {
        'accuracy_macro': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'specificity_macro': compute_macro_specificity(y_true, y_pred, n_classes),
    }
    unique = np.unique(y_true)
    if len(unique) < n_classes:
        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=range(n_classes))
            metrics['auc_macro_ovr'] = float(roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr'))
        except (ValueError, Exception):
            try:
                metrics['auc_macro_ovr'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=unique))
            except (ValueError, Exception):
                metrics['auc_macro_ovr'] = np.nan
    else:
        try:
            metrics['auc_macro_ovr'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
        except ValueError:
            metrics['auc_macro_ovr'] = np.nan

    # Per-class
    for i, name in enumerate(class_names):
        if i >= n_classes:
            break
        key = name  # agonist, antagonist, inactive
        prec = precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
        y_bin = (y_true == i).astype(int)
        y_pbin = (y_pred == i).astype(int)
        tn = ((y_bin == 0) & (y_pbin == 0)).sum()
        fp = ((y_bin == 0) & (y_pbin == 1)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f'precision_{key}'] = float(prec)
        metrics[f'recall_{key}'] = float(rec)
        metrics[f'f1_{key}'] = float(f1)
        metrics[f'specificity_{key}'] = float(spec)
        try:
            if y_proba.shape[1] > i:
                metrics[f'auc_{key}'] = float(roc_auc_score(y_bin, y_proba[:, i]))
            else:
                metrics[f'auc_{key}'] = np.nan
        except (ValueError, Exception):
            metrics[f'auc_{key}'] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            metrics[f'cm{i}{j}'] = int(cm[i, j])
    return metrics


# ============================================================================
# Load data (same as baseline/scaffold)
# ============================================================================

print("="*80)
print("RANDOM FOREST LORO EVALUATION")
print("="*80)

print("\n[1] Aggregating receptor features...")
receptor_feats = {}
for rec in RECEPTORS:
    feats = aggregate_receptor_features(rec)
    if feats:
        receptor_feats[rec] = feats
receptor_df = pd.DataFrame(receptor_feats).T.fillna(0) if receptor_feats else pd.DataFrame()
if len(receptor_df) > 0:
    receptor_df.index.name = 'Receptor'
print(f"  Receptors with features: {len(receptor_df)}")

print("\n[2] Loading ligand data (_NEW only, validated)...")
receptors_to_load = list(receptor_df.index) if len(receptor_df) > 0 else RECEPTORS
data_df, ligand_cols = load_receptor_ligand_data(receptors_to_load, OUTPUT_DIR)

if len(receptor_df) > 0:
    merged = data_df.merge(receptor_df.reset_index(), on='Receptor', how='left')
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
else:
    merged = data_df

merged, interaction_cols = build_interaction_terms(merged, receptor_df)
feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols if len(receptor_df) > 0 else ligand_cols + interaction_cols
feature_cols = [c for c in feature_cols if c in merged.columns]
X_full = merged[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
X_full = X_full.select_dtypes(include=[np.number])
final_feature_cols = X_full.columns.tolist()

y_full, class_names_array, class_names = encode_labels(merged)
expected = {'agonist': 0, 'antagonist': 1, 'inactive': 2}
actual = {class_names[i]: i for i in range(len(class_names))}
if actual != expected:
    print(f"  [WARNING] Label mapping: expected {expected}, got {actual}")

n_classes = len(class_names)
receptors_present = merged['Receptor'].unique().tolist()
print(f"  Samples: {len(merged):,}, features: {len(final_feature_cols)}, receptors: {len(receptors_present)}")

# ============================================================================
# LORO loop
# ============================================================================

print("\n[3] LORO evaluation (receptor × seed)...")
print("-"*80)

per_receptor_seed_rows = []
all_y_true = []
all_y_pred = []
all_y_proba = []
all_test_sizes = []
all_receptors_list = []
all_seeds_list = []
max_test_size = 0

for seed in SEEDS:
    for held_out in receptors_present:
        train_mask = (merged['Receptor'] != held_out)
        test_mask = (merged['Receptor'] == held_out)
        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test < N_MIN_TEST:
            print(f"  [SKIP] {held_out} seed={seed}: n_test={n_test} < {N_MIN_TEST}")
            continue
        y_train = y_full[train_mask.values]
        train_classes = set(y_train)
        if len(train_classes) < N_CLASSES:
            print(f"  [SKIP] {held_out} seed={seed}: train missing class(es) {set(range(N_CLASSES)) - train_classes}")
            continue

        X_train = X_full.loc[train_mask].reset_index(drop=True)
        X_test = X_full.loc[test_mask].reset_index(drop=True)
        y_test = y_full[test_mask.values]

        params = {**BEST_PARAMS, 'random_state': seed}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_proba = align_proba(y_proba, model.classes_, n_classes)
        if np.any(np.isnan(y_proba)) or np.any(np.isinf(y_proba)):
            y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)

        met = evaluate_model(y_test, y_pred, y_proba, n_classes, class_names)
        auc_str = f"{met['auc_macro_ovr']:.4f}" if not (isinstance(met['auc_macro_ovr'], (int, float)) and np.isnan(met['auc_macro_ovr'])) else "NaN"
        print(f"  {held_out} seed={seed} n_train={n_train} n_test={n_test} F1={met['f1_macro']:.4f} AUC={auc_str}")

        row = {
            'model': MODEL_NAME, 'receptor': held_out, 'seed': seed,
            'n_train': n_train, 'n_test': n_test,
            'accuracy_macro': met['accuracy_macro'], 'precision_macro': met['precision_macro'],
            'recall_macro': met['recall_macro'], 'f1_macro': met['f1_macro'],
            'specificity_macro': met['specificity_macro'], 'auc_macro_ovr': met['auc_macro_ovr'],
        }
        for k, v in met.items():
            if k in row:
                continue
            row[k] = v
        per_receptor_seed_rows.append(row)

        max_test_size = max(max_test_size, n_test)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_proba.append(y_proba)
        all_test_sizes.append(n_test)
        all_receptors_list.append(held_out)
        all_seeds_list.append(seed)

# ============================================================================
# DataFrames and aggregates
# ============================================================================

df_seed = pd.DataFrame(per_receptor_seed_rows)
macro_cols = ['accuracy_macro', 'precision_macro', 'recall_macro', 'f1_macro', 'specificity_macro', 'auc_macro_ovr']

per_receptor_mean_std = []
for rec in df_seed['receptor'].unique():
    sub = df_seed[df_seed['receptor'] == rec]
    row = {'receptor': rec, 'n_test_mean': sub['n_test'].mean()}
    for m in macro_cols:
        vals = sub[m].replace([np.nan], np.nan)
        row[f'{m}_mean'] = float(np.nanmean(vals))
        row[f'{m}_std'] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
    per_receptor_mean_std.append(row)
df_receptor = pd.DataFrame(per_receptor_mean_std)

overall = {'model': MODEL_NAME}
for m in macro_cols:
    vals = df_seed[m].replace([np.nan], np.nan)
    overall[f'{m}_mean'] = float(np.nanmean(vals))
    overall[f'{m}_std'] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
df_overall = pd.DataFrame([overall])

# ============================================================================
# Save CSVs
# ============================================================================

path_seed = OUTPUT_DIR / 'rf_loro_per_receptor_per_seed.csv'
path_receptor = OUTPUT_DIR / 'rf_loro_per_receptor_mean_std.csv'
path_overall = OUTPUT_DIR / 'rf_loro_overall_mean_std.csv'
df_seed.to_csv(path_seed, index=False)
df_receptor.to_csv(path_receptor, index=False)
df_overall.to_csv(path_overall, index=False)
print(f"\n[OK] {path_seed.name}")
print(f"[OK] {path_receptor.name}")
print(f"[OK] {path_overall.name}")

# ============================================================================
# Padded predictions NPZ
# ============================================================================

n_pairs = len(all_y_true)
y_true_pad = np.full((n_pairs, max_test_size), -1, dtype=np.int32)
y_pred_pad = np.full((n_pairs, max_test_size), -1, dtype=np.int32)
y_proba_pad = np.full((n_pairs, max_test_size, N_CLASSES), np.nan, dtype=np.float32)
for i in range(n_pairs):
    n = all_test_sizes[i]
    y_true_pad[i, :n] = all_y_true[i]
    y_pred_pad[i, :n] = all_y_pred[i]
    y_proba_pad[i, :n, :] = all_y_proba[i]
np.savez_compressed(
    OUTPUT_DIR / 'rf_loro_predictions.npz',
    y_true=y_true_pad, y_pred=y_pred_pad, y_proba=y_proba_pad,
    test_sizes=np.array(all_test_sizes),
    receptors=np.array(all_receptors_list, dtype=object),
    seeds=np.array(all_seeds_list),
    class_names=np.array(class_names, dtype=object),
)
print(f"[OK] rf_loro_predictions.npz")

# ============================================================================
# Overall summary print
# ============================================================================

print("\n" + "="*80)
print("OVERALL (all receptor×seed)")
print("="*80)
for m in macro_cols:
    mu, s = overall[f'{m}_mean'], overall[f'{m}_std']
    if isinstance(mu, (int, float)) and np.isnan(mu):
        print(f"  {m}: NaN")
    else:
        print(f"  {m}: {mu:.4f} ± {s:.4f}")
print("="*80)

# Optional SHAP (post-hoc, one receptor × one seed; run after saves so crashes don't lose results)
if RUN_SHAP and len(per_receptor_seed_rows) > 0:
    try:
        import shap
        rep_rec = df_seed['receptor'].iloc[0]
        rep_seed = SEEDS[0]
        train_mask = (merged['Receptor'] != rep_rec)
        test_mask = (merged['Receptor'] == rep_rec)
        if test_mask.sum() >= N_MIN_TEST and len(set(y_full[train_mask.values])) >= N_CLASSES:
            X_tr = X_full.loc[train_mask].reset_index(drop=True)
            X_te = X_full.loc[test_mask].reset_index(drop=True)
            y_tr = y_full[train_mask.values]
            rng = np.random.RandomState(42)
            n_samp = min(500, len(X_te))
            idx = rng.choice(len(X_te), size=n_samp, replace=False)
            X_te_samp = X_te.iloc[idx]
            model_shap = RandomForestClassifier(**{**BEST_PARAMS, 'random_state': rep_seed})
            model_shap.fit(X_tr, y_tr)
            explainer = shap.TreeExplainer(model_shap)
            shap_vals = explainer.shap_values(X_te_samp)
            np.savez_compressed(
                OUTPUT_DIR / 'rf_loro_shap_values.npz',
                shap_values=np.array(shap_vals, dtype=object) if isinstance(shap_vals, list) else shap_vals,
                feature_names=np.array(final_feature_cols, dtype=object),
                class_names=np.array(class_names, dtype=object),
            )
            print("\n[OK] Optional SHAP saved: rf_loro_shap_values.npz")
    except Exception as e:
        print(f"\n[WARNING] Optional SHAP failed: {e}")
