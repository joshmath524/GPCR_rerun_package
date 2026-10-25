#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Random Stratified Split Evaluation (10 seeds)

- Uses fixed best-trial hyperparameters (Optuna trial 76, F1≈0.8357); n_estimators=300.
- Random stratified split: 80% training / 20% test (allows SMILES overlap).
- For each seed: stratified 80/20 split, 5-fold StratifiedKFold CV on 80% training portion;
  train on full 80% training set, evaluate on 20% test set (10 seeds: 42-51).
- Saves per-seed and summary metrics (CSVs), and test-set predictions (.npz).
- Saves SHAP values post-hoc (after all metrics are saved) using the first-seed model on a sample
  of the test set; writes rf_random_stratified_shap_values.npz and rf_random_stratified_shap_X_test_sample.csv.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.ensemble import RandomForestClassifier
import shap

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_utilities import (
    align_proba,
    load_receptor_ligand_data,
    aggregate_receptor_features,
    build_interaction_terms,
    should_exclude_column,
    canonicalize_smiles_series,
    encode_labels
)

print("="*80)
print("RANDOM FOREST RANDOM STRATIFIED SPLIT EVALUATION (10 SEEDS)")
print("="*80)
print("\nRandom Stratified Split: 80% training / 20% test (allows SMILES overlap)")
print("For each seed: 5-fold StratifiedKFold CV on training portion + test set evaluation")
print("Using fixed best-trial hyperparameters (trial 76), n_estimators=300")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_FOLDS = 5
N_CLASSES = 3
OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECEPTORS = [
    '5-HT1A','5-HT1B','5-HT1D','5-HT1F','5-HT2A','5-HT2B','5-HT2C',
    '5-HT4','5-HT5A','5-HT6','5-HT7','A1','A2A','A2B','A3',
    'alpha1A','alpha1B','alpha2C','beta1','beta2','beta3',
    'BLT1','CB1','CB2','CysLT1','CysLT2','D1','D2','D3','D4','D5',
    'EP2','EP3','EP4','FFA1','FFA2','FFA3','FFA4','FP',
    'GPBA','GPR119','GPR139','GPR174','GPR35','H1','H2','H3','H4','HCA2','IP',
    'LPA1','M1','M2','M3','M4','M5','MT1','MT2','P2Y1','P2Y12','PAF',
    'S1P1','S1P2','S1P3','S1P5','succinate','TA1','TP'
]

# Best-trial hyperparameters (Optuna trial 76, F1≈0.8357); n_estimators=300
print("\n[STEP 0] Using fixed best-trial hyperparameters (trial 76)...")
print("-"*80)
best_params = {
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
print(f"  [OK] Hyperparameters set (n_estimators=300)")

# ============================================================================
# Step 1: Load Data
# ============================================================================

print("\n[STEP 1] Loading data...")
print("-"*80)

# Aggregate receptor features
print("Aggregating receptor features...")
receptor_feats = {}
for rec in RECEPTORS:
    feats = aggregate_receptor_features(rec)
    if feats:
        receptor_feats[rec] = feats

receptor_df = pd.DataFrame(receptor_feats).T.fillna(0) if receptor_feats else pd.DataFrame()
if len(receptor_df) > 0:
    receptor_df.index.name = 'Receptor'
    print(f"  [OK] {len(receptor_df)} receptors with features")

# Load ligand data
print("\nLoading ligand data from individual receptor files (_NEW only, validated)...")
receptors_to_load = list(receptor_df.index) if len(receptor_df) > 0 else RECEPTORS
data_df, ligand_cols = load_receptor_ligand_data(receptors_to_load, OUTPUT_DIR)
print(f"  [OK] {len(data_df):,} total samples")

# Merge with receptor features (receptor_df has Receptor as index -> reset for merge)
if len(receptor_df) > 0:
    receptor_df_reset = receptor_df.reset_index()
    merged = data_df.merge(receptor_df_reset, on='Receptor', how='left')
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
else:
    merged = data_df

# Build interaction terms
merged, interaction_cols = build_interaction_terms(merged, receptor_df)
print(f"  [OK] {len(interaction_cols)} interaction terms")

# Canonicalize SMILES (for consistency, but we allow overlap in random split)
merged, smiles_series = canonicalize_smiles_series(merged)

# Build feature matrix
feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols if len(receptor_df) > 0 else ligand_cols + interaction_cols
feature_cols = [c for c in feature_cols if c in merged.columns]

y, class_names_array, class_names = encode_labels(merged)
X_temp = merged[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
X = X_temp.select_dtypes(include=[np.number])
final_feature_cols = X.columns.tolist()

print(f"  [OK] Total features (numeric only): {len(final_feature_cols)}")
print(f"  [OK] Feature matrix: {X.shape}, Classes: {list(class_names)}")

# ============================================================================
# Step 2: Metrics Functions
# ============================================================================

def compute_macro_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    specs = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
    return float(np.mean(specs))

def evaluate_model(y_true, y_pred, y_proba, n_classes, class_names=None):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'specificity_macro': compute_macro_specificity(y_true, y_pred, n_classes),
    }
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
    except ValueError:
        metrics['auc'] = np.nan
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )
    for i, class_name in enumerate(class_names if class_names else range(n_classes)):
        metrics[f'{class_name}_precision'] = float(precisions[i])
        metrics[f'{class_name}_recall'] = float(recalls[i])
        metrics[f'{class_name}_f1'] = float(f1s[i])
    return metrics

# ============================================================================
# Step A: Evaluation Loop (10 Seeds × 5-Fold CV + Test Set)
# ============================================================================

print("\n[STEP A] Evaluation Loop (10 seeds × 5-fold StratifiedKFold CV + test set)...")
print("-"*80)

all_results_cv = []
all_results_test = []
all_predictions = []
first_seed_model = None
first_seed_X_test = None
first_seed_y_test = None

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\nSeed {seed_idx}/{len(SEEDS)} (seed={seed})...", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    test_indices = X_test.index.values
    print(f"  Split: Train={len(X_train):,}, Test={len(X_test):,}", flush=True)

    print(f"  Starting {N_FOLDS}-fold CV on training set...", flush=True)
    cv_results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"    Fold {fold}/{N_FOLDS}...", end=' ', flush=True)
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        model_params = best_params.copy()
        model_params['random_state'] = seed
        model_params['oob_score'] = False
        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train_fold, y_train_fold)

        y_pred = clf.predict(X_val_fold)
        y_proba = clf.predict_proba(X_val_fold)
        y_proba = align_proba(y_proba, clf.classes_, N_CLASSES)
        metrics = evaluate_model(y_val_fold, y_pred, y_proba, N_CLASSES, class_names)
        cv_results.append(metrics)
        auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else "NaN"
        print(f"F1={metrics['f1_macro']:.4f}, AUC={auc_str}", flush=True)

    cv_metrics = {'seed': seed}
    for key in cv_results[0].keys():
        values = [r[key] for r in cv_results]
        cv_metrics[f'cv_{key}_mean'] = float(np.mean(values))
        cv_metrics[f'cv_{key}_std'] = float(np.std(values, ddof=1))
    all_results_cv.append(cv_metrics)
    auc_mean_str = f"{cv_metrics['cv_auc_mean']:.4f}" if not np.isnan(cv_metrics['cv_auc_mean']) else "NaN"
    auc_std_str = f"{cv_metrics['cv_auc_std']:.4f}" if not np.isnan(cv_metrics['cv_auc_std']) else "NaN"
    print(f"  [CV RESULTS] F1={cv_metrics['cv_f1_macro_mean']:.4f}±{cv_metrics['cv_f1_macro_std']:.4f}, AUC={auc_mean_str}±{auc_std_str}", flush=True)

    print(f"  Final test evaluation: Train on ALL training set, test on test set...", end=' ', flush=True)
    model_params_final = best_params.copy()
    model_params_final['random_state'] = seed
    model_params_final['oob_score'] = False
    clf_final = RandomForestClassifier(**model_params_final)
    clf_final.fit(X_train, y_train)

    if seed == SEEDS[0]:
        first_seed_model = clf_final
        first_seed_X_test = X_test.copy()
        first_seed_y_test = y_test.copy()

    y_pred_test = clf_final.predict(X_test)
    y_proba_test = clf_final.predict_proba(X_test)
    y_proba_test = align_proba(y_proba_test, clf_final.classes_, N_CLASSES)
    test_metrics = evaluate_model(y_test, y_pred_test, y_proba_test, N_CLASSES, class_names)
    test_metrics['seed'] = seed
    test_metrics_prefixed = {f'test_{k}' if k != 'seed' else k: v for k, v in test_metrics.items()}
    all_results_test.append(test_metrics_prefixed)

    all_predictions.append({
        'seed': seed,
        'split_type': 'random_stratified_80_20',
        'y_true': y_test,
        'y_pred': y_pred_test,
        'y_pred_proba': y_proba_test,
        'test_indices': test_indices
    })
    auc_str = f"{test_metrics['auc']:.4f}" if not np.isnan(test_metrics['auc']) else "NaN"
    print(f"F1={test_metrics['f1_macro']:.4f}, AUC={auc_str}", flush=True)

# ============================================================================
# Step B: Save Metrics CSVs
# ============================================================================

print("\n" + "="*80)
print("[STEP B] Saving metrics to CSV files...")
print("="*80)

per_seed_data = []
for i, seed in enumerate(SEEDS):
    combined_row = {**all_results_cv[i], **all_results_test[i]}
    per_seed_data.append(combined_row)
per_seed_df = pd.DataFrame(per_seed_data)
per_seed_file = OUTPUT_DIR / 'rf_random_stratified_per_seed.csv'
per_seed_df.to_csv(per_seed_file, index=False)
print(f"  [OK] Saved per-seed results: {per_seed_file}")

summary_dict = {}
cv_metric_names = [k.replace('cv_', '').replace('_mean', '') for k in all_results_cv[0].keys() if k.startswith('cv_') and k.endswith('_mean')]
for metric_name in cv_metric_names:
    if metric_name == 'seed':
        continue
    mean_key = f'cv_{metric_name}_mean'
    std_key = f'cv_{metric_name}_std'
    if mean_key in all_results_cv[0]:
        values = [r[mean_key] for r in all_results_cv]
        if metric_name == 'auc':
            summary_dict[f'cv_{metric_name}_mean'] = float(np.nanmean(values))
            summary_dict[f'cv_{metric_name}_std'] = float(np.nanstd(values, ddof=1))
        else:
            summary_dict[f'cv_{metric_name}_mean'] = float(np.mean(values))
            summary_dict[f'cv_{metric_name}_std'] = float(np.std(values, ddof=1))

test_metric_names = [k.replace('test_', '') for k in all_results_test[0].keys() if k.startswith('test_')]
for metric_name in test_metric_names:
    key = f'test_{metric_name}'
    values = [r[key] for r in all_results_test]
    if metric_name == 'auc':
        summary_dict[f'test_{metric_name}_mean'] = float(np.nanmean(values))
        summary_dict[f'test_{metric_name}_std'] = float(np.nanstd(values, ddof=1))
    else:
        summary_dict[f'test_{metric_name}_mean'] = float(np.mean(values))
        summary_dict[f'test_{metric_name}_std'] = float(np.std(values, ddof=1))

summary_df = pd.DataFrame([summary_dict])
summary_df.index = ['RandomForest_random_stratified']
summary_file = OUTPUT_DIR / 'rf_random_stratified_mean_std.csv'
summary_df.to_csv(summary_file, index=True)
print(f"  [OK] Saved summary: {summary_file}")

per_class_test_data = []
for seed in SEEDS:
    test_row = next(r for r in all_results_test if r['seed'] == seed)
    row = {'seed': seed}
    for class_name in class_names:
        for metric_type in ['recall', 'precision', 'f1']:
            row[f'test_{class_name}_{metric_type}'] = test_row.get(f'test_{class_name}_{metric_type}', np.nan)
    per_class_test_data.append(row)
per_class_test_df = pd.DataFrame(per_class_test_data)
per_class_test_file = OUTPUT_DIR / 'rf_random_stratified_per_class_test.csv'
per_class_test_df.to_csv(per_class_test_file, index=False)

per_class_test_summary_dict = {}
for class_name in class_names:
    for metric_type in ['recall', 'precision', 'f1']:
        key = f'test_{class_name}_{metric_type}'
        mean = float(np.nanmean(per_class_test_df[key])) if per_class_test_df[key].notna().any() else np.nan
        std = float(np.nanstd(per_class_test_df[key])) if per_class_test_df[key].notna().any() else np.nan
        per_class_test_summary_dict[f'{key}_mean'] = mean
        per_class_test_summary_dict[f'{key}_std'] = std
per_class_test_summary_df = pd.DataFrame([per_class_test_summary_dict])
per_class_test_summary_file = OUTPUT_DIR / 'rf_random_stratified_per_class_test_mean_std.csv'
per_class_test_summary_df.to_csv(per_class_test_summary_file, index=False)
print(f"  [OK] Saved per-class test results: {per_class_test_file}")
print(f"  [OK] Saved per-class test summary: {per_class_test_summary_file}")

# Save predictions
predictions_file = OUTPUT_DIR / 'rf_random_stratified_predictions.npz'
test_seed_list = np.array([p['seed'] for p in all_predictions])
test_sizes = [len(p['y_true']) for p in all_predictions]
max_test_size = max(test_sizes) if test_sizes else 0
test_y_true_all = np.full((len(all_predictions), max_test_size), -1, dtype=np.int32)
test_y_pred_all = np.full((len(all_predictions), max_test_size), -1, dtype=np.int32)
test_y_pred_proba_all = np.full((len(all_predictions), max_test_size, N_CLASSES), np.nan, dtype=np.float32)
test_indices_all = np.full((len(all_predictions), max_test_size), -1, dtype=np.int32)
for i, pred in enumerate(all_predictions):
    n_test = len(pred['y_true'])
    test_y_true_all[i, :n_test] = np.asarray(pred['y_true'])
    test_y_pred_all[i, :n_test] = np.asarray(pred['y_pred'])
    test_y_pred_proba_all[i, :n_test, :] = np.asarray(pred['y_pred_proba'])
    test_indices_all[i, :n_test] = np.asarray(pred['test_indices'])

np.savez_compressed(
    predictions_file,
    seed_list=test_seed_list,
    split_types=np.array([p.get('split_type', 'random_stratified_80_20') for p in all_predictions]),
    test_sizes=np.array(test_sizes),
    y_true=test_y_true_all,
    y_pred=test_y_pred_all,
    y_pred_proba=test_y_pred_proba_all,
    test_indices=test_indices_all,
    class_names=np.array(class_names)
)
print(f"  [OK] Saved test set predictions: {predictions_file}")
print(f"       Shape: y_true {test_y_true_all.shape}, y_pred_proba {test_y_pred_proba_all.shape}")
print(f"       Test sizes per seed: {test_sizes}")

# Print summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print("\n5-Fold CV Results (Mean ± Std across seeds):")
print(f"  Accuracy:  {summary_dict['cv_accuracy_mean']:.4f} ± {summary_dict['cv_accuracy_std']:.4f}")
print(f"  Precision: {summary_dict['cv_precision_macro_mean']:.4f} ± {summary_dict['cv_precision_macro_std']:.4f}")
print(f"  Recall:    {summary_dict['cv_recall_macro_mean']:.4f} ± {summary_dict['cv_recall_macro_std']:.4f}")
print(f"  F1-score:  {summary_dict['cv_f1_macro_mean']:.4f} ± {summary_dict['cv_f1_macro_std']:.4f}")
print(f"  Specificity: {summary_dict['cv_specificity_macro_mean']:.4f} ± {summary_dict['cv_specificity_macro_std']:.4f}")
auc_mean_str = f"{summary_dict['cv_auc_mean']:.4f}" if not np.isnan(summary_dict['cv_auc_mean']) else "NaN"
auc_std_str = f"{summary_dict['cv_auc_std']:.4f}" if not np.isnan(summary_dict['cv_auc_std']) else "NaN"
print(f"  AUC:       {auc_mean_str} ± {auc_std_str}")
print("\nTest Set Results (Mean ± Std across seeds):")
print(f"  Accuracy:  {summary_dict['test_accuracy_mean']:.4f} ± {summary_dict['test_accuracy_std']:.4f}")
print(f"  Precision: {summary_dict['test_precision_macro_mean']:.4f} ± {summary_dict['test_precision_macro_std']:.4f}")
print(f"  Recall:    {summary_dict['test_recall_macro_mean']:.4f} ± {summary_dict['test_recall_macro_std']:.4f}")
print(f"  F1-score:  {summary_dict['test_f1_macro_mean']:.4f} ± {summary_dict['test_f1_macro_std']:.4f}")
print(f"  Specificity: {summary_dict['test_specificity_macro_mean']:.4f} ± {summary_dict['test_specificity_macro_std']:.4f}")
auc_mean_str = f"{summary_dict['test_auc_mean']:.4f}" if not np.isnan(summary_dict['test_auc_mean']) else "NaN"
auc_std_str = f"{summary_dict['test_auc_std']:.4f}" if not np.isnan(summary_dict['test_auc_std']) else "NaN"
print(f"  AUC:       {auc_mean_str} ± {auc_std_str}")

# ============================================================================
# Step C: Compute and Save SHAP (POST-HOC)
# ============================================================================

print("\n" + "="*80)
print("[STEP C] Computing SHAP values (POST-HOC interpretability only)...")
print("-"*80)

if first_seed_model is None:
    print("  [WARNING] No model from first seed available for SHAP computation", flush=True)
else:
    print(f"  Using model from first seed (seed={SEEDS[0]})...", flush=True)
    try:
        n_shap_samples = min(500, len(first_seed_X_test))
        sample_idx = np.random.RandomState(42).choice(len(first_seed_X_test), n_shap_samples, replace=False)
        X_test_shap = first_seed_X_test.iloc[sample_idx]
        print(f"  Computing SHAP values on sample of {n_shap_samples} test samples...", flush=True)
        explainer = shap.TreeExplainer(first_seed_model)
        shap_values = explainer.shap_values(X_test_shap)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value_array = np.array(expected_value)
        else:
            expected_value_array = np.array([expected_value])
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values)
        elif isinstance(shap_values, np.ndarray):
            shap_array = shap_values
        else:
            raise TypeError(f"Unexpected SHAP values type: {type(shap_values).__name__}")
        shap_dir = OUTPUT_DIR / 'shap'
        shap_dir.mkdir(parents=True, exist_ok=True)
        X_test_shap.to_csv(shap_dir / 'rf_random_stratified_shap_X_test_sample.csv', index=False)
        np.savez(
            shap_dir / 'rf_random_stratified_shap_values.npz',
            shap_values=shap_array,
            expected_value=expected_value_array,
            sample_indices=sample_idx,
            feature_names=np.array(final_feature_cols),
            class_names=np.array(class_names),
            seed_used=np.array([SEEDS[0]])
        )
        print(f"  [OK] SHAP values saved to {shap_dir / 'rf_random_stratified_shap_values.npz'}", flush=True)
    except Exception as e:
        print(f"  [WARNING] SHAP computation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
