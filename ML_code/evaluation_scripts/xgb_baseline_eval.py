#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Baseline Evaluation with Best Optuna Hyperparameters
Unified design with global split shared across all scripts.

Evaluates on:
1. 5-fold StratifiedKFold CV on 80% dev block
2. Final evaluation on 20% independent test set (train on ALL dev80, test on independent20)
   - This is the ONLY true test set evaluation
   - Independent test (20%) is HELD OUT and never used for CV or tuning
   - SHAP values computed on test set for post-hoc interpretability only (not used for model selection)

Uses global split: splits/final_independent_ligand_80_20_canonical_smiles.json
Shared across baseline, Optuna, and scaffold scripts.

Loads data from individual receptor files (only _NEW files via load_table)
Uses shared utilities for consistent feature construction.
Uses fixed best-trial hyperparameters (Optuna trial 111, ligand-grouped CV F1≈0.7192); n_estimators=1843.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_utilities import (
    WORKSPACE_ROOT,
    align_proba,
    load_receptor_ligand_data,
    aggregate_receptor_features,
    build_interaction_terms,
    should_exclude_column,
    canonicalize_smiles_series,
    encode_labels,
    load_or_create_global_split
)

print("="*80)
print("XGBOOST BASELINE EVALUATION - BEST OPTUNA HYPERPARAMETERS")
print("UNIFIED DESIGN - GLOBAL SPLIT")
print("="*80)
print("\n5-fold StratifiedKFold CV on 80% dev block")
print("\nFinal evaluation: Train on ALL dev80, test on independent20 (held-out test set)")
print("10 seeds × 5-fold CV + final independent test evaluation")
print("\nNote: SHAP values computed on test set for post-hoc interpretability only (not used for model selection)")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

SEEDS = list(range(42, 52))  # 10 seeds: 42-51
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

# ============================================================================
# Step 1: Best-Trial Hyperparameters (Optuna trial 111, ligand-grouped CV F1≈0.7192); n_estimators=1843
# ============================================================================

print("\n[STEP 1] Using fixed best-trial hyperparameters (trial 111)...")
print("-"*80)
best_params = {
    'n_estimators': 1843,
    'learning_rate': 0.01252429957342966,
    'max_depth': 8,
    'min_child_weight': 4.871195746919201,
    'subsample': 0.9821311054801911,
    'colsample_bytree': 0.8825716321923549,
    'gamma': 0.00014044502676304957,
    'reg_alpha': 1.7667662377574005,
    'reg_lambda': 0.7558033347639419,
    'objective': 'multi:softprob',
    'num_class': N_CLASSES,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
}
print(f"  [OK] Hyperparameters set (n_estimators=1843)")

# ============================================================================
# Step 2: Load Data and Prepare Features
# ============================================================================

print("\n[STEP 2] Loading data and preparing features...")
print("-"*80)

# Note: load_table, aggregate_receptor_features, build_interaction_terms,
# canonicalize_smiles_series, and encode_labels are imported from shared_utilities

# Aggregate receptor features
print("\nAggregating receptor features...")
receptor_feats = {}
for idx, rec in enumerate(RECEPTORS, 1):
    print(f"  Processing receptor {idx}/{len(RECEPTORS)}: {rec}...", end=' ', flush=True)
    feats = aggregate_receptor_features(rec)
    if feats:
        receptor_feats[rec] = feats
        print("[OK]", flush=True)
    else:
        print("[SKIP - no features found]", flush=True)

receptor_df = pd.DataFrame(receptor_feats).T.fillna(0) if receptor_feats else pd.DataFrame()
if len(receptor_df) > 0:
    receptor_df.index.name = 'Receptor'
    print(f"  [OK] {len(receptor_df)} receptors with features")
else:
    print("  [WARNING] No receptor features found")

# Load ligand data from individual receptor files
print("\nLoading ligand data from individual receptor files (_NEW only, validated)...")
receptors_to_load = list(receptor_df.index) if len(receptor_df) > 0 else RECEPTORS
data_df, ligand_cols = load_receptor_ligand_data(receptors_to_load, OUTPUT_DIR)
print(f"  [OK] {len(data_df):,} total samples")

# Merge with receptor features
if len(receptor_df) > 0:
    receptor_df_reset = receptor_df.reset_index()  # makes "Receptor" a real column
    merged = data_df.merge(receptor_df_reset, on='Receptor', how='left')
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
else:
    merged = data_df

print(f"  [OK] {len(ligand_cols)} ligand descriptor columns")
print(f"  [OK] {len(receptor_df.columns) if len(receptor_df) > 0 else 0} receptor feature columns")

# Build 14 interaction terms (using shared utility)
merged, interaction_cols = build_interaction_terms(merged, receptor_df)
print(f"  [OK] {len(interaction_cols)} interaction terms")

# Canonicalize SMILES for grouping (BEFORE building X to ensure alignment)
# This prevents leakage from different SMILES representing same molecule
merged, smiles_series = canonicalize_smiles_series(merged)

# NOW build X from merged (after canonicalization and filtering)
feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols if len(receptor_df) > 0 else ligand_cols + interaction_cols
# Filter to only columns that exist in merged (prevents KeyError if any receptor file is missing descriptor columns)
feature_cols = [c for c in feature_cols if c in merged.columns]
X_temp = merged[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
X = X_temp.select_dtypes(include=[np.number])
final_feature_cols = X.columns.tolist()

print(f"  [OK] Total features (numeric only): {len(final_feature_cols)}")

# Encode labels (using shared utility for consistent mapping)
y, class_names_array, class_names = encode_labels(merged)
n_classes = len(class_names)

print(f"  [OK] Classes: {class_names} (n={n_classes})")
print(f"  [OK] Total samples: {len(X):,}")

# ============================================================================
# Step 3: Create Splits
# ============================================================================

print("\n[STEP 3] Loading global independent ligand validation split...")
print("-"*80)

# Global split lives in MLcodes/pan_gpcr_results/splits; it is created on first run and
# must match the current dataset exactly on later runs (load_or_create_global_split raises otherwise)
SPLIT_DIR = WORKSPACE_ROOT / 'MLcodes' / 'pan_gpcr_results' / 'splits'

train_idx, test_idx = load_or_create_global_split(
    X, y, smiles_series, SPLIT_DIR, random_state=42, test_size=0.2, load_only=False
)
train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

# Create training and test sets from global split
# Dev80 block: used for CV (two modes) and final training
# Independent20 block: HELD OUT, only used for final test evaluation
X_dev80 = X.iloc[train_idx].reset_index(drop=True)
X_independent20 = X.iloc[test_idx].reset_index(drop=True)
y_dev80 = y[train_idx]
y_independent20 = y[test_idx]
smiles_dev80 = smiles_series.iloc[train_idx].reset_index(drop=True) if hasattr(smiles_series, 'iloc') else smiles_series[train_idx]
smiles_independent20 = smiles_series.iloc[test_idx].reset_index(drop=True) if hasattr(smiles_series, 'iloc') else smiles_series[test_idx]

# Verify no SMILES overlap
dev80_smiles_set = set(smiles_dev80)
independent20_smiles_set = set(smiles_independent20)
overlap = dev80_smiles_set & independent20_smiles_set
if overlap:
    print(f"  [ERROR] Found {len(overlap)} overlapping SMILES in global split!", flush=True)
    raise ValueError("Global split has SMILES overlap! This should not happen with GroupShuffleSplit by canonical SMILES.")
else:
    print(f"  [OK] Global split verified (no SMILES overlap):", flush=True)
    print(f"    Dev80 block (80%): {len(X_dev80):,} samples ({len(dev80_smiles_set):,} unique canonical SMILES)", flush=True)
    print(f"    Independent20 block (20%): {len(X_independent20):,} samples ({len(independent20_smiles_set):,} unique canonical SMILES)", flush=True)
    print(f"    [NOTE] Independent20 block is HELD OUT and only used for final test evaluation", flush=True)

# Check class distribution in dev80/independent20 split
print(f"  [INFO] Class distribution check (GroupShuffleSplit is not stratified):", flush=True)
dev80_class_dist = pd.Series(y_dev80).value_counts().sort_index()
independent20_class_dist = pd.Series(y_independent20).value_counts().sort_index()
for i, class_name in enumerate(class_names):
    dev80_count = dev80_class_dist.get(i, 0)
    indep20_count = independent20_class_dist.get(i, 0)
    dev80_pct = (dev80_count / len(y_dev80) * 100) if len(y_dev80) > 0 else 0
    indep20_pct = (indep20_count / len(y_independent20) * 100) if len(y_independent20) > 0 else 0
    print(f"    {class_name}: Dev80 {dev80_count} ({dev80_pct:.1f}%), Independent20 {indep20_count} ({indep20_pct:.1f}%)", flush=True)

# Verify all classes are present in independent20 test set
# Note: load_or_create_global_split() now tries multiple seeds to ensure all classes are present
# This check is a final verification
missing_classes = []
for i, class_name in enumerate(class_names):
    if independent20_class_dist.get(i, 0) == 0:
        missing_classes.append(class_name)

if missing_classes:
    print(f"  [ERROR] Independent20 test set is missing classes: {missing_classes}", flush=True)
    print(f"  [ERROR] This should not happen - load_or_create_global_split() tries multiple seeds to avoid this", flush=True)
    raise ValueError(
        f"CRITICAL: Independent20 test set is missing classes: {missing_classes}. "
        f"This will cause misleading AUC and per-class metrics. "
        f"This error indicates the robust split creation in load_or_create_global_split() failed. "
        f"Please check the data or adjust test_size."
    )
print(f"  [OK] All classes present in independent20 test set (verified)", flush=True)

# Warn if class distribution is very imbalanced between dev80/independent20 (>5% absolute difference)
for i, class_name in enumerate(class_names):
    dev80_count = dev80_class_dist.get(i, 0)
    indep20_count = independent20_class_dist.get(i, 0)
    dev80_pct = (dev80_count / len(y_dev80) * 100) if len(y_dev80) > 0 else 0
    indep20_pct = (indep20_count / len(y_independent20) * 100) if len(y_independent20) > 0 else 0
    diff_pct = abs(dev80_pct - indep20_pct)
    if diff_pct > 5.0:
        print(f"  [WARNING] {class_name} distribution imbalance: Dev80 {dev80_pct:.1f}% vs Independent20 {indep20_pct:.1f}% (diff={diff_pct:.1f}%)", flush=True)

# ============================================================================
# Step 4: Define Metrics Computation Functions
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

def compute_per_class_metrics(y_true, y_pred, y_proba, class_names):
    """Compute per-class metrics (recall, precision, f1, specificity, AUC)"""
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    # Per-class precision, recall, f1
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )
    
    # Per-class specificity
    specificities = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    
    # Per-class AUC (one-vs-rest)
    per_class_auc = []
    try:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        for i in range(n_classes):
            try:
                auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                per_class_auc.append(auc)
            except ValueError:
                per_class_auc.append(np.nan)
    except:
        per_class_auc = [np.nan] * n_classes
    
    # Build per-class dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[f'{class_name}_recall'] = float(recalls[i])
        per_class_metrics[f'{class_name}_precision'] = float(precisions[i])
        per_class_metrics[f'{class_name}_f1'] = float(f1s[i])
        per_class_metrics[f'{class_name}_specificity'] = float(specificities[i])
        per_class_metrics[f'{class_name}_auc'] = float(per_class_auc[i])
    
    return per_class_metrics

def evaluate_model(y_true, y_pred, y_proba, n_classes, class_names=None):
    """Evaluate model and return all metrics (macro-averaged and per-class)"""
    # Use precision_recall_fscore_support for better sklearn version compatibility
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    metrics = {
        'recall': float(rec),
        'precision': float(prec),
        'f1': float(f1),
        'specificity': compute_macro_specificity(y_true, y_pred, n_classes),
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }
    
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
    except ValueError:
        metrics['auc'] = np.nan
    
    # Add per-class metrics if class_names provided
    if class_names is not None:
        per_class = compute_per_class_metrics(y_true, y_pred, y_proba, class_names)
        metrics.update(per_class)
    
    return metrics

# ============================================================================
# Step 5: Evaluation Loop (10 Seeds × 5-Fold CV + Independent Test)
# ============================================================================

print("\n[STEP 5] Evaluation Loop (10 seeds × 5-fold StratifiedKFold CV + independent test set)...")
print("-"*80)

all_results_cv = []  # 5-fold StratifiedKFold CV results
all_results_final_test = []  # Final evaluation on independent20 (held-out test set)
all_predictions = []  # Store predictions for later confusion matrix generation
all_cv_oof_predictions = []  # Store out-of-fold CV predictions
first_seed_model = None  # Store model from first seed for SHAP computation (after Step 6)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\nSeed {seed_idx}/{len(SEEDS)} (seed={seed})...", flush=True)
    
    # ========================================================================
    # 5-fold StratifiedKFold CV on dev80 block
    # ========================================================================
    print(f"  5-fold StratifiedKFold CV on dev80...", flush=True)
    cv_results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    # Initialize arrays for out-of-fold predictions aligned to Dev80 order
    n_dev80 = len(X_dev80)
    oof_y_true = np.full(n_dev80, -1, dtype=np.int32)  # Initialize with -1 to detect missing
    oof_y_pred_proba = np.full((n_dev80, n_classes), np.nan, dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev80, y_dev80), 1):
        print(f"    Fold {fold}/{N_FOLDS}...", end=' ', flush=True)
        
        X_train_fold = X_dev80.iloc[train_idx]
        X_val_fold = X_dev80.iloc[val_idx]
        y_train_fold = y_dev80[train_idx]
        y_val_fold = y_dev80[val_idx]
        
        # Check if all classes are present in validation fold
        fold_classes = set(y_val_fold)
        if len(fold_classes) < n_classes:
            missing_fold_classes = [class_names[i] for i in range(n_classes) if i not in fold_classes]
            print(f"[WARNING: Missing classes {missing_fold_classes} in fold {fold}]", end=' ', flush=True)
        
        # XGBoost: No scaling needed (tree-based model)
        model_params = best_params.copy()
        model_params['random_state'] = seed
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        y_pred = model.predict(X_val_fold)
        y_proba = model.predict_proba(X_val_fold)
        y_proba = align_proba(y_proba, model.classes_, n_classes)  # Use shared utility
        
        # Store out-of-fold predictions in correct positions (aligned to Dev80 order)
        # val_idx contains the positions in the Dev80 DataFrame
        oof_y_true[val_idx] = y_val_fold
        oof_y_pred_proba[val_idx, :] = y_proba
        
        metrics = evaluate_model(y_val_fold, y_pred, y_proba, n_classes, class_names)
        cv_results.append(metrics)
        
        auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else "NaN"
        print(f"F1={metrics['f1']:.4f}, AUC={auc_str}", flush=True)
    
    # Verify that every Dev80 sample got exactly one OOF prediction
    missing_oof = np.sum(oof_y_true == -1)
    if missing_oof > 0:
        raise ValueError(f"CRITICAL: {missing_oof} Dev80 samples did not receive OOF predictions!")
    
    nan_proba_count = np.isnan(oof_y_pred_proba).any(axis=1).sum()
    if nan_proba_count > 0:
        raise ValueError(f"CRITICAL: {nan_proba_count} Dev80 samples have NaN probabilities!")
    
    print(f"  [OK] OOF predictions collected for all {n_dev80} Dev80 samples", flush=True)
    
    # Store OOF predictions for this seed
    all_cv_oof_predictions.append({
        'seed': seed,
        'oof_y_true': oof_y_true.copy(),
        'oof_y_pred_proba': oof_y_pred_proba.copy()
    })
    
    # Average CV results across folds
    cv_metrics = {
        'recall': np.mean([r['recall'] for r in cv_results]),
        'precision': np.mean([r['precision'] for r in cv_results]),
        'f1': np.mean([r['f1'] for r in cv_results]),
        'auc': float(np.nanmean([r['auc'] for r in cv_results])),
        'specificity': np.mean([r['specificity'] for r in cv_results]),
        'accuracy': np.mean([r['accuracy'] for r in cv_results]),
        'seed': seed
    }
    # Add per-class metrics
    for class_name in class_names:
        for metric_type in ['recall', 'precision', 'f1', 'specificity', 'auc']:
            key = f'{class_name}_{metric_type}'
            values = [r.get(key, np.nan) for r in cv_results]
            cv_metrics[key] = float(np.nanmean(values)) if not all(np.isnan(v) for v in values) else np.nan
    all_results_cv.append(cv_metrics)
    auc_str = f"{cv_metrics['auc']:.4f}" if not np.isnan(cv_metrics['auc']) else "NaN"
    print(f"  CV Results - F1: {cv_metrics['f1']:.4f}, AUC: {auc_str}", flush=True)
    
    # ========================================================================
    # FINAL TEST EVALUATION: Train on ALL dev80, test on independent20 (held-out)
    # This is the ONLY true test set evaluation
    # ========================================================================
    print(f"  Final test evaluation: Train on ALL dev80, test on independent20 (held-out)...", end=' ', flush=True)
    
    # XGBoost: No scaling needed (tree-based model)
    model_params = best_params.copy()
    model_params['random_state'] = seed
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_dev80, y_dev80)  # Train on ALL dev80
    
    # Evaluate on independent20 (held-out test set)
    y_pred_final = model.predict(X_independent20)
    y_proba_final = model.predict_proba(X_independent20)
    y_proba_final = align_proba(y_proba_final, model.classes_, n_classes)  # Use shared utility
    
    final_test_metrics = evaluate_model(y_independent20, y_pred_final, y_proba_final, n_classes, class_names)
    final_test_metrics['seed'] = seed
    final_test_metrics['split_type'] = 'independent20_final_test'
    all_results_final_test.append(final_test_metrics)
    
    # Store predictions for later confusion matrix regeneration (independent20 test set)
    all_predictions.append({
        'seed': seed,
        'split_type': 'independent20_final_test',
        'y_true': y_independent20,
        'y_pred': y_pred_final,
        'y_pred_proba': y_proba_final
    })
    if seed == SEEDS[0]:
        first_seed_model = model
    
    auc_str = f"{final_test_metrics['auc']:.4f}" if not np.isnan(final_test_metrics['auc']) else "NaN"
    print(f"F1={final_test_metrics['f1']:.4f}, AUC={auc_str}", flush=True)

# ============================================================================
# Step 6: Aggregate Results and Save
# ============================================================================

print("\n[STEP 6] Aggregating results and saving...")
print("-"*80)

# Convert to DataFrames
cv_df = pd.DataFrame(all_results_cv)
final_test_df = pd.DataFrame(all_results_final_test)

# Compute mean ± std across seeds (using nanmean for AUC which may have NaN values)
metric_names = ['recall', 'precision', 'f1', 'auc', 'specificity', 'accuracy']

summary_dict = {}
for metric in metric_names:
    if metric == 'auc':
        cv_mean = float(np.nanmean(cv_df[metric]))
        cv_std = float(np.nanstd(cv_df[metric]))
        final_test_mean = float(np.nanmean(final_test_df[metric]))
        final_test_std = float(np.nanstd(final_test_df[metric]))
        cv_auc_frac = float(cv_df[metric].notna().mean())
        final_test_auc_frac = float(final_test_df[metric].notna().mean())
    else:
        cv_mean = cv_df[metric].mean()
        cv_std = cv_df[metric].std()
        final_test_mean = final_test_df[metric].mean()
        final_test_std = final_test_df[metric].std()
    
    summary_dict[f'{metric}_cv_mean'] = cv_mean
    summary_dict[f'{metric}_cv_std'] = cv_std
    summary_dict[f'{metric}_final_test_mean'] = final_test_mean
    summary_dict[f'{metric}_final_test_std'] = final_test_std

# Report AUC coverage
if 'auc' in metric_names:
    print(f"\n  [INFO] AUC defined for {cv_auc_frac*100:.1f}% of CV, {final_test_auc_frac*100:.1f}% of final test evaluations", flush=True)

# Create summary DataFrame
summary_df = pd.DataFrame([summary_dict])
summary_df.index = ['XGBoost_baseline']

# Save files
per_seed_file = OUTPUT_DIR / 'xgb_baseline_per_seed.csv'
summary_file = OUTPUT_DIR / 'xgb_baseline_mean_std.csv'

# Combine CV and final test results per seed
per_seed_data = []
for seed in SEEDS:
    cv_row = cv_df[cv_df['seed'] == seed].iloc[0]
    final_test_row = final_test_df[final_test_df['seed'] == seed].iloc[0]
    row = {'seed': seed}
    for metric in metric_names:
        row[f'cv_{metric}'] = cv_row[metric]
        row[f'final_test_{metric}'] = final_test_row[metric]
    per_seed_data.append(row)

per_seed_df = pd.DataFrame(per_seed_data)
per_seed_df.to_csv(per_seed_file, index=False)
summary_df.to_csv(summary_file, index=True)

print(f"  [OK] Saved per-seed results: {per_seed_file}")
print(f"  [OK] Saved summary: {summary_file}")

# Save per-class metrics (final test only - independent20)
per_class_final_test_data = []
for seed in SEEDS:
    final_test_row = final_test_df[final_test_df['seed'] == seed].iloc[0]
    row = {'seed': seed}
    for class_name in class_names:
        for metric_type in ['recall', 'precision', 'f1', 'specificity', 'auc']:
            key = f'{class_name}_{metric_type}'
            row[key] = final_test_row.get(key, np.nan)
    per_class_final_test_data.append(row)

per_class_final_test_df = pd.DataFrame(per_class_final_test_data)
per_class_final_test_file = OUTPUT_DIR / 'xgb_baseline_per_class_final_test.csv'
per_class_final_test_df.to_csv(per_class_final_test_file, index=False)

# Compute mean ± std for per-class final test metrics (using nanmean/nanstd to handle missing values)
per_class_final_test_summary_dict = {}
for class_name in class_names:
    for metric_type in ['recall', 'precision', 'f1', 'specificity', 'auc']:
        key = f'{class_name}_{metric_type}'
        mean = float(np.nanmean(per_class_final_test_df[key])) if per_class_final_test_df[key].notna().any() else np.nan
        std = float(np.nanstd(per_class_final_test_df[key])) if per_class_final_test_df[key].notna().any() else np.nan
        per_class_final_test_summary_dict[f'{key}_mean'] = mean
        per_class_final_test_summary_dict[f'{key}_std'] = std

per_class_final_test_summary_df = pd.DataFrame([per_class_final_test_summary_dict])
per_class_final_test_summary_file = OUTPUT_DIR / 'xgb_baseline_per_class_final_test_mean_std.csv'
per_class_final_test_summary_df.to_csv(per_class_final_test_summary_file, index=False)

print(f"  [OK] Saved per-class final test results: {per_class_final_test_file}")
print(f"  [OK] Saved per-class final test summary: {per_class_final_test_summary_file}")

# Save out-of-fold CV predictions (5-fold StratifiedKFold on Dev80)
cv_oof_file = OUTPUT_DIR / 'xgb_cv_oof_predictions.npz'
# Prepare arrays for saving
seed_list = np.array([p['seed'] for p in all_cv_oof_predictions])
# Stack OOF predictions: shape [n_seeds, n_dev80] for y_true and [n_seeds, n_dev80, n_classes] for y_pred_proba
oof_y_true_all = np.stack([p['oof_y_true'] for p in all_cv_oof_predictions], axis=0)
oof_y_pred_proba_all = np.stack([p['oof_y_pred_proba'] for p in all_cv_oof_predictions], axis=0)

np.savez_compressed(
    cv_oof_file,
    seed_list=seed_list,
    oof_y_true=oof_y_true_all,
    oof_y_pred_proba=oof_y_pred_proba_all,
    class_names=np.array(class_names)
)
print(f"  [OK] Saved CV out-of-fold predictions: {cv_oof_file}")
print(f"       Shape: oof_y_true [{oof_y_true_all.shape}], oof_y_pred_proba [{oof_y_pred_proba_all.shape}]")

# Save predictions for later confusion matrix generation (independent20 test set only)
predictions_file = OUTPUT_DIR / 'xgb_baseline_predictions.npz'
# Prepare arrays - independent20 predictions with probabilities
# Note: independent20 size is fixed across all seeds (same split), so we can stack them
test_seed_list = np.array([p['seed'] for p in all_predictions])
n_independent20 = len(all_predictions[0]['y_true'])  # Get size from first prediction
test_y_true_all = np.stack([np.asarray(p['y_true']) for p in all_predictions], axis=0)  # Shape: [n_seeds, n_independent20]
test_y_pred_all = np.stack([np.asarray(p['y_pred']) for p in all_predictions], axis=0)  # Shape: [n_seeds, n_independent20]
test_y_pred_proba_all = np.stack([np.asarray(p['y_pred_proba']) for p in all_predictions], axis=0)  # Shape: [n_seeds, n_independent20, n_classes]

np.savez_compressed(
    predictions_file,
    seed_list=test_seed_list,
    split_types=np.array([p.get('split_type', 'independent20_final_test') for p in all_predictions]),
    y_true=test_y_true_all,
    y_pred=test_y_pred_all,
    y_pred_proba=test_y_pred_proba_all,
    class_names=np.array(class_names)
)
print(f"  [OK] Saved independent20 test set predictions: {predictions_file}")
print(f"       Shape: y_true [{test_y_true_all.shape}], y_pred_proba [{test_y_pred_proba_all.shape}]")
print(f"       (Run generate_baseline_confusion_matrix.py to generate confusion matrices)")

# Print summary
print("\n" + "="*80)
print("RESULTS - MEAN ± STANDARD DEVIATION (across 10 seeds)")
print("="*80)

print("\n5-fold StratifiedKFold CV (on dev80 block):")
print("-"*80)
for metric in metric_names:
    mean = summary_dict[f'{metric}_cv_mean']
    std = summary_dict[f'{metric}_cv_std']
    print(f"  {metric.capitalize():15s}: {mean:.4f} ± {std:.4f}")

print("\nFinal Test Evaluation (Independent20 - Held-Out Test Set):")
print("-"*80)
print("  Note: Train on ALL dev80, test on independent20 (held-out, never used for CV or tuning)")
for metric in metric_names:
    mean = summary_dict[f'{metric}_final_test_mean']
    std = summary_dict[f'{metric}_final_test_std']
    print(f"  {metric.capitalize():15s}: {mean:.4f} ± {std:.4f}")

print("\n" + "="*80)
print("Evaluation complete!")
print("="*80)

# ============================================================================
# Step 7: Compute SHAP Values (POST-HOC INTERPRETABILITY ONLY)
# ============================================================================

print("\n[STEP 7] Computing SHAP values (POST-HOC interpretability only)...")
print("-"*80)
print("Note: SHAP values are computed on the test set for interpretability purposes only.")
print("They are NOT used for model selection, tuning, or any feedback into training.")
print("-"*80)

if first_seed_model is None:
    print("  [WARNING] No model from first seed available for SHAP computation", flush=True)
else:
    print(f"  Using model from first seed (seed={SEEDS[0]})...", flush=True)
    try:
        n_shap_samples = min(500, len(X_independent20))
        sample_idx = np.random.RandomState(42).choice(len(X_independent20), n_shap_samples, replace=False)
        X_test_shap = X_independent20.iloc[sample_idx]
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
            print(f"  [INFO] SHAP values type: list -> converted to array, shape: {shap_array.shape}", flush=True)
        elif isinstance(shap_values, np.ndarray):
            shap_array = shap_values
            print(f"  [INFO] SHAP values type: ndarray, shape: {shap_array.shape}", flush=True)
        else:
            raise TypeError(f"Unexpected SHAP values type: {type(shap_values).__name__}")
        shap_dir = OUTPUT_DIR / 'shap'
        shap_dir.mkdir(parents=True, exist_ok=True)
        X_test_shap.to_csv(shap_dir / 'xgb_baseline_shap_X_test_sample.csv', index=False)
        print(f"  [OK] X_test_sample saved to {shap_dir / 'xgb_baseline_shap_X_test_sample.csv'}", flush=True)
        np.savez(
            shap_dir / 'xgb_baseline_shap_values.npz',
            shap_values=shap_array,
            expected_value=expected_value_array,
            sample_indices=sample_idx,
            feature_names=np.array(final_feature_cols),
            class_names=np.array(class_names),
            seed_used=np.array([SEEDS[0]])
        )
        print(f"  [OK] SHAP values saved to {shap_dir / 'xgb_baseline_shap_values.npz'}", flush=True)
        print(f"       Format: shap_values {shap_array.shape}, expected_value {expected_value_array.shape}, seed_used: {SEEDS[0]}", flush=True)
    except Exception as e:
        print(f"  [WARNING] SHAP computation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("SHAP computation complete!")
print("="*80)

