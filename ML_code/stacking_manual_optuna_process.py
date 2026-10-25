#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Stacking Ensemble - Exact Optuna Process
Implements the exact process described:
1. Fixed best-trial hyperparameters (trial 282 LGB, 76 RF, 111 XGB; same trees as the cleaned-data single-model runs)
2. Load data and prepare features
3. Load global 80/20 split (final_independent_ligand_80_20_canonical_smiles.json, same as baseline)
4. 5-fold StratifiedKFold manual stacking with out-of-fold predictions
5. Average test predictions across folds
6. Train meta-model on meta-features
7. Evaluate on independent 20% test set
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='No further splits with positive gain')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb
import shap

# Import shared utilities (same as baseline scripts)
import sys
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
from shared_utilities import (
    load_receptor_ligand_data,
    aggregate_receptor_features,
    build_interaction_terms,
    canonicalize_smiles_series,
    encode_labels,
    load_or_create_global_split,
)

print("="*80, flush=True)
print("MANUAL STACKING ENSEMBLE - EXACT OPTUNA PROCESS", flush=True)
print("="*80, flush=True)
print("\nBase-model hyperparameters match the cleaned-data single-model runs:", flush=True)
print("  LightGBM trial 282 (614 trees), Random Forest trial 76 (300 trees),", flush=True)
print("  XGBoost trial 111 (1843 trees).", flush=True)
# TODO: Update header text — we no longer scale per fold (scaling removed for tree models).
# Remove the "Per-fold scaling" bullet to avoid methods/code mismatch.
print("\nKey differences from sklearn StackingClassifier:", flush=True)
print("  - Manual averaging of test predictions across 5 folds", flush=True)
print("  - Explicit out-of-fold prediction handling", flush=True)
print("="*80, flush=True)

# ============================================================================
# Constants
# ============================================================================

SEED = 42
N_FOLDS = 5
N_CLASSES = 3
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # 10 seeds for robustness
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
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
# Step 1: Fixed best-trial hyperparameters (same as baseline scripts)
# ============================================================================
# LightGBM: trial 282 (saved model in lightgbm_trial282_results), n_estimators=614
# RF: trial 76, n_estimators=300
# XGBoost: trial 111, n_estimators=1843

print("\n[1/8] Using fixed best-trial hyperparameters (trial 282 LGB, 76 RF, 111 XGB)...", flush=True)

lgb_params = {
    'learning_rate': 0.011249562072804727,
    'num_leaves': 293,
    'max_depth': -1,
    'min_child_samples': 34,
    'subsample': 0.9327986680901219,
    'subsample_freq': 1,
    'colsample_bytree': 0.9998957099075128,
    'reg_alpha': 0.29417314311904885,
    'reg_lambda': 0.15439671666779758,
    'n_estimators': 614,
    'objective': 'multiclass',
    'num_class': N_CLASSES,
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
}

rf_params = {
    'bootstrap': False,
    'criterion': 'entropy',
    'max_depth': 30,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 9,
    'n_estimators': 300,
    'random_state': SEED,
    'n_jobs': -1
}

xgb_params = {
    'learning_rate': 0.01252429957342966,
    'max_depth': 8,
    'min_child_weight': 4.871195746919201,
    'subsample': 0.9821311054801911,
    'colsample_bytree': 0.8825716321923549,
    'gamma': 0.00014044502676304957,
    'reg_alpha': 1.7667662377574005,
    'reg_lambda': 0.7558033347639419,
    'n_estimators': 1843,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0
}

print(f"  [OK] Hyperparameters set (LGB trial 282, RF trial 76, XGB trial 111)", flush=True)

# ============================================================================
# Step 2: Load Data and Prepare Features (same as baseline: shared_utilities)
# ============================================================================

print("\n[2/8] Loading data and preparing features (shared_utilities, same as baseline)...", flush=True)

workspace_root = _script_dir.parent

# Aggregate receptor features (shared_utilities)
receptor_feats = {}
for idx, rec in enumerate(RECEPTORS, 1):
    feats = aggregate_receptor_features(rec)
    if feats:
        receptor_feats[rec] = feats
    if idx % 20 == 0 or idx == len(RECEPTORS):
        print(f"  Processed {idx}/{len(RECEPTORS)} receptors...", flush=True)

receptor_df = pd.DataFrame(receptor_feats).T.fillna(0) if receptor_feats else pd.DataFrame()
if len(receptor_df) > 0:
    receptor_df.index.name = 'Receptor'
    print(f"  [OK] {len(receptor_df)} receptors with features", flush=True)
else:
    print("  [WARNING] No receptor features found", flush=True)

# Load ligand data (same folder logic as baseline)
receptors_to_load = list(receptor_df.index) if len(receptor_df) > 0 else RECEPTORS
data_df, ligand_cols = load_receptor_ligand_data(receptors_to_load, OUTPUT_DIR)
print(f"  [OK] {len(data_df):,} total samples", flush=True)

# Merge with receptor features
if len(receptor_df) > 0:
    receptor_df_reset = receptor_df.reset_index()
    merged = data_df.merge(receptor_df_reset, on='Receptor', how='left')
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
else:
    merged = data_df

# Build interaction terms (shared_utilities)
merged, interaction_cols = build_interaction_terms(merged, receptor_df)
print(f"  [OK] {len(interaction_cols)} interaction terms", flush=True)

# Canonicalize SMILES (same as baseline)
merged, smiles_series = canonicalize_smiles_series(merged)

# Build X (numeric only)
feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols if len(receptor_df) > 0 else ligand_cols + interaction_cols
feature_cols = [c for c in feature_cols if c in merged.columns]
X_temp = merged[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
X = X_temp.select_dtypes(include=[np.number])

# Sanitize feature names for LightGBM compatibility (remove special JSON characters)
def sanitize_feature_name(name):
    """Replace special JSON characters with underscores (for LightGBM JSON dumping)."""
    sanitized = re.sub(r"[^\w\-.]", "_", str(name))
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized if sanitized else "feature_" + str(hash(name) % 10000)

X.columns = [sanitize_feature_name(col) for col in X.columns]

# Encode labels (shared_utilities)
y, class_names_array, class_names = encode_labels(merged)
class_names = np.array(class_names)  # ensure safe indexing with numpy arrays
print(f"  [OK] Features: {X.shape[1]}, Classes: {class_names.tolist()}, Samples: {len(X):,}", flush=True)

# ============================================================================
# Step 3: Load Global Independent Ligand Validation Split (same as baseline)
# ============================================================================

print("\n[3/8] Loading global independent ligand validation split...", flush=True)

SPLIT_DIR = workspace_root / 'MLcodes' / 'pan_gpcr_results' / 'splits'
global_split_file = SPLIT_DIR / 'final_independent_ligand_80_20_canonical_smiles.json'
if not global_split_file.exists():
    raise FileNotFoundError(
        f"Global split file not found: {global_split_file}. "
        "Create it by running a baseline script (e.g. lightgbm_baseline_eval) with load_only=False once."
    )

train_idx, test_idx = load_or_create_global_split(
    X, y, smiles_series, SPLIT_DIR, random_state=SEED, test_size=0.2, load_only=True
)
train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

# Create 80% train and 20% independent test sets
X_80 = X.iloc[train_idx].reset_index(drop=True)
y_80 = y[train_idx]
X_test_20 = X.iloc[test_idx].reset_index(drop=True)
y_test_20 = y[test_idx]

print(f"  [OK] Training (80%): {len(X_80):,} samples", flush=True)
print(f"  [OK] Test (20%): {len(X_test_20):,} samples", flush=True)

# TODO: Make SMILES overlap check STRICT (match baseline behavior).
# If any canonical SMILES appear in both dev80 and independent20, raise ValueError.
# Do NOT allow a warning-only behavior.
# This must behave identically to baseline scripts.
dev80_smiles = set(smiles_series.iloc[train_idx] if hasattr(smiles_series, 'iloc') else smiles_series[train_idx])
indep20_smiles = set(smiles_series.iloc[test_idx] if hasattr(smiles_series, 'iloc') else smiles_series[test_idx])
overlap = dev80_smiles & indep20_smiles
if overlap:
    raise ValueError(
        f"CRITICAL: Found {len(overlap)} overlapping canonical SMILES between dev80 and independent20!"
    )
else:
    print("  [OK] No SMILES overlap (canonical)", flush=True)

# ============================================================================
# Step 4: Define Metrics Computation Functions
# ============================================================================

print("\n[4/8] Defining metrics computation functions...", flush=True)

def compute_macro_specificity(y_true, y_pred, n_classes):
    """Build confusion matrix and compute macro-averaged specificity"""
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

def calculate_specificity(y_true, y_pred, class_label, all_classes):
    """Calculate specificity for a specific class (True Negative Rate)"""
    y_true_binary = (y_true == class_label).astype(int)
    y_pred_binary = (y_pred == class_label).astype(int)
    
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

# TODO: Fix AUC calculation.
# Use integer labels (range(N_CLASSES)) for binarization.
# Must align with evaluate_model() implementation.
def calculate_multiclass_auc(y_true, y_pred_proba, classes):
    """Calculate macro-averaged AUC and per-class AUC (integer labels for binarization)."""
    from sklearn.preprocessing import label_binarize
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    try:
        macro_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
        per_class_auc = {}
        for i, class_name in enumerate(classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                try:
                    auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                    per_class_auc[class_name] = auc
                except Exception:
                    per_class_auc[class_name] = np.nan
            else:
                per_class_auc[class_name] = np.nan
        return macro_auc, per_class_auc
    except Exception:
        return np.nan, {class_name: np.nan for class_name in classes}

def evaluate_model(y_true, y_pred, y_proba, n_classes):
    """Compute all metrics (macro-averaged)"""
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    specificity = compute_macro_specificity(y_true, y_pred, n_classes)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute AUC (multi-class ROC-AUC, OVR, macro-averaged)
    try:
        lb = LabelBinarizer()
        lb.fit(range(n_classes))
        y_true_bin = lb.transform(y_true)
        auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'accuracy': accuracy
    }

print("  [OK] Metrics functions defined", flush=True)

# ============================================================================
# Step 5: 5-fold StratifiedKFold Stacking (Multiple Seeds)
# ============================================================================

print("\n" + "="*80, flush=True)
print("[5/8] PERFORMING 5-FOLD STRATIFIEDKFOLD STACKING (10 SEEDS)", flush=True)
print("="*80, flush=True)
print("This process implements manual stacking with:", flush=True)
print("  - 5-fold CV for out-of-fold meta-features", flush=True)
print("  - Averaging test predictions across folds", flush=True)
print("  - Independent Ligand Validation (no SMILES overlap)", flush=True)
print("  - Base models: LightGBM trial 282, Random Forest trial 76, XGBoost trial 111", flush=True)
print("="*80, flush=True)

all_seed_metrics = []
all_per_class_metrics = {class_name: [] for class_name in class_names}
all_predictions = []  # Store predictions for confusion matrix generation
first_seed_meta_model = None  # For SHAP computation
first_seed_meta_X_test = None  # Meta-features for first seed test set

for seed_idx, seed in enumerate(SEEDS, 1):
    print("\n" + "-"*80, flush=True)
    print(f"SEED {seed_idx}/10 (seed={seed})", flush=True)
    print("-"*80, flush=True)
    
    # Initialize meta-feature arrays
    meta_X_train = np.zeros((len(X_80), N_CLASSES * 3))  # 3 models × 3 classes = 9 meta-features
    meta_X_test_list = []
    
    # TODO: Replace StratifiedKFold with StratifiedGroupKFold.
    # Groups must be canonical SMILES (from smiles_series restricted to dev80).
    # This prevents the same ligand (same canonical SMILES) appearing in both
    # train and validation folds during stacking.
    # Must match manuscript statement about grouping by canonical SMILES.
    groups_dev80 = np.asarray(smiles_series.iloc[train_idx] if hasattr(smiles_series, 'iloc') else smiles_series[train_idx])
    assert len(groups_dev80) == len(X_80), "groups_dev80 must align with X_80 row order"
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (train_idx_fold, val_idx_fold) in enumerate(
            sgkf.split(X_80, y_80, groups=groups_dev80), 1):
        print(f"\n  [SEED {seed_idx}/10] FOLD {fold}/{N_FOLDS}...", flush=True)
        
        # Split data
        X_train_fold = X_80.iloc[train_idx_fold].reset_index(drop=True)
        X_val_fold = X_80.iloc[val_idx_fold].reset_index(drop=True)
        y_train_fold = y_80[train_idx_fold]
        y_val_fold = y_80[val_idx_fold]
        
        print(f"    Train fold: {len(X_train_fold):,} samples, Val fold: {len(X_val_fold):,} samples", flush=True)
        
        # TODO: Remove StandardScaler for XGBoost and LightGBM.
        # Tree-based models do not require scaling.
        # Baseline scripts do not scale features.
        # Keep preprocessing identical across baseline and stacking.
        # Train base models on fold-train (raw features for all models)
        print(f"    Training base models (RF, XGB, LGB)...", flush=True)
        
        # Random Forest (no scaling needed)
        print(f"      Training Random Forest...", flush=True)
        rf_model = RandomForestClassifier(**{**rf_params, 'random_state': seed})
        rf_model.fit(X_train_fold, y_train_fold)
        rf_proba_val = rf_model.predict_proba(X_val_fold)
        rf_proba_test = rf_model.predict_proba(X_test_20)
        print(f"      [OK] Random Forest trained", flush=True)
        
        # XGBoost (raw features, no scaling)
        print(f"      Training XGBoost...", flush=True)
        xgb_model = xgb.XGBClassifier(**{**xgb_params, 'random_state': seed})
        xgb_model.fit(X_train_fold, y_train_fold)
        xgb_proba_val = xgb_model.predict_proba(X_val_fold)
        xgb_proba_test = xgb_model.predict_proba(X_test_20)
        print(f"      [OK] XGBoost trained", flush=True)
        
        # LightGBM (raw features, no scaling)
        print(f"      Training LightGBM...", flush=True)
        lgb_model = lgb.LGBMClassifier(**{**lgb_params, 'random_state': seed})
        lgb_model.fit(X_train_fold, y_train_fold)
        lgb_proba_val = lgb_model.predict_proba(X_val_fold)
        lgb_proba_test = lgb_model.predict_proba(X_test_20)
        print(f"      [OK] LightGBM trained", flush=True)
        
        # Store out-of-fold predictions (for meta-training)
        meta_features_fold = np.hstack([rf_proba_val, xgb_proba_val, lgb_proba_val])  # Shape: (n_val, 9)
        meta_X_train[val_idx_fold] = meta_features_fold
        
        # Store test predictions (will average later)
        fold_test_predictions = np.hstack([rf_proba_test, xgb_proba_test, lgb_proba_test])  # Shape: (n_test, 9)
        meta_X_test_list.append(fold_test_predictions)
        
        print(f"    [OK] Fold {fold}/{N_FOLDS} complete", flush=True)
    
    print(f"\n  [SEED {seed_idx}/10] Meta-features created: train shape {meta_X_train.shape}, test list length {len(meta_X_test_list)}", flush=True)
    
    # ============================================================================
    # Step 6: Build Meta-Features for Test Set
    # ============================================================================
    
    print(f"  [SEED {seed_idx}/10] Averaging test predictions across {N_FOLDS} folds...", flush=True)
    
    # Average test predictions across folds
    meta_X_test = np.mean(meta_X_test_list, axis=0)  # Shape: (n_test_20, 9)
    
    # ============================================================================
    # Step 7: Train Meta-Model
    # ============================================================================
    
    print(f"  [SEED {seed_idx}/10] Training meta-model (Logistic Regression)...", flush=True)
    
    # Create meta-model
    meta_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=seed
    )
    
    # Train on meta-features
    meta_model.fit(meta_X_train, y_80)
    print(f"  [SEED {seed_idx}/10] [OK] Meta-model trained", flush=True)
    
    # ============================================================================
    # Step 8: Evaluate on Independent 20% Test Set
    # ============================================================================
    
    print(f"  [SEED {seed_idx}/10] Evaluating on independent test set...", flush=True)
    
    # Predict with meta-model
    y_pred_meta = meta_model.predict(meta_X_test)
    y_proba_meta = meta_model.predict_proba(meta_X_test)
    
    # Store predictions for confusion matrix generation
    all_predictions.append({
        'seed': seed,
        'split_type': 'independent20_final_test',
        'y_true': y_test_20.copy(),
        'y_pred': y_pred_meta.copy(),
        'y_pred_proba': y_proba_meta.copy()
    })
    
    # Store first-seed meta-model and meta-features for SHAP
    if seed == SEEDS[0]:
        first_seed_meta_model = meta_model
        first_seed_meta_X_test = meta_X_test.copy()
    
    # Compute overall metrics
    print(f"  [SEED {seed_idx}/10] Computing metrics...", flush=True)
    metrics = evaluate_model(y_test_20, y_pred_meta, y_proba_meta, N_CLASSES)
    metrics['seed'] = seed
    all_seed_metrics.append(metrics)
    
    # Compute per-class metrics
    precision_per_class = precision_score(y_test_20, y_pred_meta, labels=range(len(class_names)), average=None, zero_division=0)
    recall_per_class = recall_score(y_test_20, y_pred_meta, labels=range(len(class_names)), average=None, zero_division=0)
    f1_per_class = f1_score(y_test_20, y_pred_meta, labels=range(len(class_names)), average=None, zero_division=0)
    
    # Calculate per-class AUC
    macro_auc, per_class_auc = calculate_multiclass_auc(y_test_20, y_proba_meta, class_names)
    
    # Convert to class names for specificity calculation
    y_test_names = class_names[y_test_20]
    y_pred_names = class_names[y_pred_meta]
    
    for i, class_name in enumerate(class_names):
        specificity = calculate_specificity(y_test_names, y_pred_names, class_name, class_names)
        all_per_class_metrics[class_name].append({
            'seed': seed,
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i],
            'specificity': specificity,
            'auc': per_class_auc.get(class_name, np.nan)
        })
    
    print(f"\n  [SEED {seed_idx}/10] [OK] COMPLETE!", flush=True)
    print(f"      Overall Metrics:", flush=True)
    print(f"        Accuracy:  {metrics['accuracy']:.6f}", flush=True)
    print(f"        Precision: {metrics['precision']:.6f}", flush=True)
    print(f"        Recall:    {metrics['recall']:.6f}", flush=True)
    print(f"        F1-Score:  {metrics['f1']:.6f}", flush=True)
    print(f"        AUC:       {metrics['auc']:.6f}", flush=True)
    print(f"        Specificity: {metrics['specificity']:.6f}", flush=True)
    print(f"      Per-Class Metrics:", flush=True)
    for i, class_name in enumerate(class_names):
        print(f"        {class_name}:", flush=True)
        print(f"          Precision: {precision_per_class[i]:.6f}", flush=True)
        print(f"          Recall:    {recall_per_class[i]:.6f}", flush=True)
        print(f"          F1:        {f1_per_class[i]:.6f}", flush=True)
        print(f"          Specificity: {calculate_specificity(y_test_names, y_pred_names, class_name, class_names):.6f}", flush=True)
        auc_val = per_class_auc.get(class_name, np.nan)
        if np.isnan(auc_val):
            print(f"          AUC:       NaN (only one class present in test set)", flush=True)
        else:
            print(f"          AUC:       {auc_val:.6f}", flush=True)

print("\n" + "="*80, flush=True)
print("[OK] ALL 10 SEEDS COMPLETED!", flush=True)
print("="*80, flush=True)

# ============================================================================
# Step 9: Compute Statistics Across Seeds
# ============================================================================

print("\n[8/8] Computing statistics across 10 seeds...", flush=True)

# Convert to DataFrame
metrics_df = pd.DataFrame(all_seed_metrics)

# Compute mean and std
mean_accuracy = metrics_df['accuracy'].mean()
mean_precision = metrics_df['precision'].mean()
mean_recall = metrics_df['recall'].mean()
mean_f1 = metrics_df['f1'].mean()
mean_auc = metrics_df['auc'].mean()
mean_specificity = metrics_df['specificity'].mean()

std_accuracy = metrics_df['accuracy'].std()
std_precision = metrics_df['precision'].std()
std_recall = metrics_df['recall'].std()
std_f1 = metrics_df['f1'].std()
std_auc = metrics_df['auc'].std()
std_specificity = metrics_df['specificity'].std()

print("\n" + "="*80, flush=True)
print("RESULTS - MANUAL STACKING ENSEMBLE (10 Seeds)", flush=True)
print("="*80, flush=True)
print(f"\nOverall Metrics (Mean ± Std, Macro-Averaged):", flush=True)
print(f"  Accuracy:  {mean_accuracy:.6f} ± {std_accuracy:.6f}", flush=True)
print(f"  Precision: {mean_precision:.6f} ± {std_precision:.6f}", flush=True)
print(f"  Recall:    {mean_recall:.6f} ± {std_recall:.6f}", flush=True)
print(f"  F1-Score:  {mean_f1:.6f} ± {std_f1:.6f}", flush=True)
print(f"  AUC:       {mean_auc:.6f} ± {std_auc:.6f}", flush=True)
print(f"  Specificity: {mean_specificity:.6f} ± {std_specificity:.6f}", flush=True)

# Compute per-class statistics
print(f"\nPer-Class Metrics (Mean ± Std):", flush=True)
print(f"{'Class':<15} {'Precision':<20} {'Recall':<20} {'F1-Score':<20} {'Specificity':<20} {'AUC':<20}", flush=True)
print('-'*115, flush=True)

per_class_summary = {}
for class_name in class_names:
    class_df = pd.DataFrame(all_per_class_metrics[class_name])
    per_class_summary[class_name] = {
        'precision_mean': class_df['precision'].mean(),
        'precision_std': class_df['precision'].std(),
        'recall_mean': class_df['recall'].mean(),
        'recall_std': class_df['recall'].std(),
        'f1_mean': class_df['f1'].mean(),
        'f1_std': class_df['f1'].std(),
        'specificity_mean': class_df['specificity'].mean(),
        'specificity_std': class_df['specificity'].std(),
        'auc_mean': class_df['auc'].mean(skipna=True),  # Skip NaN values
        'auc_std': class_df['auc'].std(skipna=True)  # Skip NaN values
    }
    summary = per_class_summary[class_name]
    # Format AUC with NaN handling
    if pd.isna(summary['auc_mean']) or pd.isna(summary['auc_std']):
        auc_str = "NaN"
    else:
        auc_str = f"{summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}"
    
    print(f"{class_name:<15} "
          f"{summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}     "
          f"{summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}     "
          f"{summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}     "
          f"{summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}     "
          f"{auc_str:<20}", flush=True)

# Save results
outputs_dir = OUTPUT_DIR
outputs_dir.mkdir(parents=True, exist_ok=True)

results = {
    'model': 'Manual Stacking Ensemble',
    'process': 'Exact Optuna Process with Independent Ligand Validation',
    'n_seeds': len(SEEDS),
    'seeds': SEEDS,
    'n_folds': N_FOLDS,
    'test_set': 'Independent Ligand Validation (GroupShuffleSplit, no SMILES overlap)',
    'split_file': 'final_independent_ligand_80_20_canonical_smiles.json (global, same as baseline)',
    'overall': {
        'accuracy_mean': float(mean_accuracy),
        'accuracy_std': float(std_accuracy),
        'precision_mean': float(mean_precision),
        'precision_std': float(std_precision),
        'recall_mean': float(mean_recall),
        'recall_std': float(std_recall),
        'f1_mean': float(mean_f1),
        'f1_std': float(std_f1),
        'auc_mean': float(mean_auc),
        'auc_std': float(std_auc),
        'specificity_mean': float(mean_specificity),
        'specificity_std': float(std_specificity),
    },
    'per_seed': all_seed_metrics,
    'per_class': {}
}

# Add per-class summary to results
for class_name in class_names:
    summary = per_class_summary[class_name]
    # Handle NaN values for JSON serialization
    auc_mean_val = summary['auc_mean']
    auc_std_val = summary['auc_std']
    results['per_class'][class_name] = {
        'precision_mean': float(summary['precision_mean']),
        'precision_std': float(summary['precision_std']),
        'recall_mean': float(summary['recall_mean']),
        'recall_std': float(summary['recall_std']),
        'f1_mean': float(summary['f1_mean']),
        'f1_std': float(summary['f1_std']),
        'specificity_mean': float(summary['specificity_mean']),
        'specificity_std': float(summary['specificity_std']),
        'auc_mean': None if pd.isna(auc_mean_val) else float(auc_mean_val),
        'auc_std': None if pd.isna(auc_std_val) else float(auc_std_val)
    }

with open(outputs_dir / 'stacking_manual_optuna_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save per-seed CSV (overall metrics)
metrics_df.to_csv(outputs_dir / 'stacking_manual_optuna_per_seed.csv', index=False)

# Save per-class metrics per seed
per_class_per_seed_data = []
for class_name in class_names:
    for seed_metrics in all_per_class_metrics[class_name]:
        per_class_per_seed_data.append({
            'seed': seed_metrics['seed'],
            'class': class_name,
            'precision': seed_metrics['precision'],
            'recall': seed_metrics['recall'],
            'f1_score': seed_metrics['f1'],
            'specificity': seed_metrics['specificity'],
            'auc': seed_metrics['auc']
        })
per_class_per_seed_df = pd.DataFrame(per_class_per_seed_data)
per_class_per_seed_df.to_csv(outputs_dir / 'stacking_manual_optuna_per_class_per_seed.csv', index=False)

# Save per-class summary (mean ± std)
per_class_summary_data = []
for class_name in class_names:
    summary = per_class_summary[class_name]
    per_class_summary_data.append({
        'class': class_name,
        'precision_mean': summary['precision_mean'],
        'precision_std': summary['precision_std'],
        'recall_mean': summary['recall_mean'],
        'recall_std': summary['recall_std'],
        'f1_mean': summary['f1_mean'],
        'f1_std': summary['f1_std'],
        'specificity_mean': summary['specificity_mean'],
        'specificity_std': summary['specificity_std'],
        'auc_mean': summary['auc_mean'],
        'auc_std': summary['auc_std']
    })
per_class_summary_df = pd.DataFrame(per_class_summary_data)
per_class_summary_df.to_csv(outputs_dir / 'stacking_manual_optuna_per_class_mean_std.csv', index=False)

# Save predictions for confusion matrix generation (same format as baseline)
predictions_file = outputs_dir / 'stacking_manual_optuna_predictions.npz'
seed_list = np.array([p['seed'] for p in all_predictions])
n_test = len(all_predictions[0]['y_true'])
test_y_true_all = np.stack([np.asarray(p['y_true']) for p in all_predictions], axis=0)
test_y_pred_all = np.stack([np.asarray(p['y_pred']) for p in all_predictions], axis=0)
test_y_pred_proba_all = np.stack([np.asarray(p['y_pred_proba']) for p in all_predictions], axis=0)
np.savez_compressed(
    predictions_file,
    seed_list=seed_list,
    split_types=np.array([p['split_type'] for p in all_predictions]),
    y_true=test_y_true_all,
    y_pred=test_y_pred_all,
    y_pred_proba=test_y_pred_proba_all,
    class_names=np.array(class_names)
)
print(f"  [OK] Saved predictions: {predictions_file}", flush=True)
print(f"       Shape: y_true {test_y_true_all.shape}, y_pred_proba {test_y_pred_proba_all.shape}", flush=True)

print(f"\n[OK] Results saved:", flush=True)
print(f"  - outputs/stacking_manual_optuna_results.json", flush=True)
print(f"  - outputs/stacking_manual_optuna_per_seed.csv (overall metrics per seed)", flush=True)
print(f"  - outputs/stacking_manual_optuna_per_class_per_seed.csv (per-class metrics per seed)", flush=True)
print(f"  - outputs/stacking_manual_optuna_per_class_mean_std.csv (per-class mean ± std)", flush=True)
print(f"  - outputs/stacking_manual_optuna_predictions.npz", flush=True)

# ============================================================================
# Step 10: Compute SHAP Values (POST-HOC INTERPRETABILITY)
# ============================================================================
# TODO: Fix indentation of SHAP block.
# The 'else:' is incorrectly indented and will cause a syntax error.
# Structure should be:
# if no model:
#     print warning
# else:
#     try:
#         ...
#     except:
#         ...

print("\n" + "="*80, flush=True)
print("[10/10] Computing SHAP values on meta-model (POST-HOC interpretability)...", flush=True)
print("="*80, flush=True)

if first_seed_meta_model is None or first_seed_meta_X_test is None:
    print("  [WARNING] No first-seed meta-model available for SHAP", flush=True)
else:
    try:
        n_shap_samples = min(500, len(first_seed_meta_X_test))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(first_seed_meta_X_test), n_shap_samples, replace=False)
        meta_X_sample = first_seed_meta_X_test[sample_idx]
        
        meta_feature_names = []
        for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            for cls in class_names:
                meta_feature_names.append(f'{model_name}_{cls}')
        
        print(f"  Using LinearExplainer on Logistic Regression meta-model...", flush=True)
        print(f"  Sample size: {n_shap_samples} test samples, 9 meta-features", flush=True)
        
        explainer = shap.LinearExplainer(first_seed_meta_model, first_seed_meta_X_test)
        shap_values = explainer.shap_values(meta_X_sample)
        
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values)
        else:
            shap_array = np.asarray(shap_values)
        
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value_array = np.array(expected_value)
        else:
            expected_value_array = np.array([expected_value])
        
        shap_dir = outputs_dir / 'shap'
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame(meta_X_sample, columns=meta_feature_names).to_csv(
            shap_dir / 'stacking_shap_meta_X_test_sample.csv', index=False
        )
        np.savez(
            shap_dir / 'stacking_shap_values.npz',
            shap_values=shap_array,
            expected_value=expected_value_array,
            sample_indices=sample_idx,
            feature_names=np.array(meta_feature_names),
            class_names=np.array(class_names),
            seed_used=np.array([SEEDS[0]])
        )
        print(f"  [OK] SHAP values saved to {shap_dir / 'stacking_shap_values.npz'}", flush=True)
        print(f"  [OK] Meta-features sample saved to {shap_dir / 'stacking_shap_meta_X_test_sample.csv'}", flush=True)
    except Exception as e:
        print(f"  [WARNING] SHAP computation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\n" + "="*80, flush=True)
print("COMPLETE!", flush=True)
print("="*80, flush=True)
