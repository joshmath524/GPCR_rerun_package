#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking Ensemble Evaluation on Random Stratified Test Set

- Uses shared_utilities (same data/features as independent ligand script): 31 receptor features,
  build_interaction_terms, canonicalize_smiles_series, encode_labels.
- Random stratified split (allows SMILES overlap) for testing set.
- StackingClassifier(cv=5) uses StratifiedKFold for meta-features only — NOT grouped by canonical
  SMILES. Do not describe this script as "grouped by canonical SMILES"; only the manual
  independent-ligand stacking script (stacking_manual_optuna_process.py) uses StratifiedGroupKFold
  by SMILES.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import re
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='No further splits with positive gain')

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, make_scorer
)
import lightgbm as lgb
import xgboost as xgb
import shap

# Import shared utilities (same as independent ligand / stacking_manual_optuna_process)
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
from shared_utilities import (
    load_receptor_ligand_data,
    aggregate_receptor_features,
    build_interaction_terms,
    canonicalize_smiles_series,
    encode_labels,
)

N_CLASSES = 3
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Explicit AUC scorer for multiclass (macro-OVR), matches test-set AUC definition
auc_scorer = make_scorer(
    roc_auc_score, needs_proba=True, multi_class="ovr", average="macro"
)

OUTPUT_DIR = _script_dir / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STACKING ENSEMBLE EVALUATION - TESTING SET (Random Stratified)")
print("Uses shared_utilities: 31 receptor features, interaction terms, canonical SMILES")
print("Allows SMILES overlap between train and test")
print("="*80)

# ============================================================================
# Utility Functions (metrics only; data/features from shared_utilities)
# ============================================================================

def calculate_specificity(y_true, y_pred, class_label, all_classes):
    """Calculate specificity for a specific class (True Negative Rate)"""
    y_true_binary = (y_true == class_label).astype(int)
    y_pred_binary = (y_pred == class_label).astype(int)
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

# AUC with integer labels (align with independent script)
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
                    per_class_auc[class_name] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                except Exception:
                    per_class_auc[class_name] = np.nan
            else:
                per_class_auc[class_name] = np.nan
        return macro_auc, per_class_auc
    except Exception:
        return np.nan, {c: np.nan for c in classes}

# ============================================================================
# 1. LOAD DATA AND PREPARE ENRICHED FEATURES (shared_utilities, same as independent)
# ============================================================================

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

workspace_root = _script_dir.parent

print("\n[1/5] Aggregating receptor features (shared_utilities, 31 features)...")
receptor_feats = {}
for idx, rec in enumerate(RECEPTORS, 1):
    feats = aggregate_receptor_features(rec)
    if feats:
        receptor_feats[rec] = feats
    if idx % 20 == 0 or idx == len(RECEPTORS):
        print(f"  Processed {idx}/{len(RECEPTORS)} receptors...")
receptor_df = pd.DataFrame(receptor_feats).T.fillna(0) if receptor_feats else pd.DataFrame()
if len(receptor_df) > 0:
    receptor_df.index.name = 'Receptor'
    print(f"  [OK] {len(receptor_df)} receptors with 31 features")
else:
    raise ValueError("No receptor features found.")

print("\n[2/5] Loading ligand data (_NEW only, validated)...")
data_df, ligand_cols = load_receptor_ligand_data(list(receptor_df.index), OUTPUT_DIR)
print(f"  [OK] {len(data_df):,} total samples")

if len(receptor_df) > 0:
    receptor_df_reset = receptor_df.reset_index()
    merged = data_df.merge(receptor_df_reset, on='Receptor', how='left')
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
else:
    merged = data_df

merged, interaction_cols = build_interaction_terms(merged, receptor_df)
print(f"  [OK] {len(interaction_cols)} interaction terms")

merged, smiles_series = canonicalize_smiles_series(merged)

feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols if len(receptor_df) > 0 else ligand_cols + interaction_cols
feature_cols = [c for c in feature_cols if c in merged.columns]
X = merged[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
X = X.select_dtypes(include=[np.number])

# Sanitize feature names for LightGBM compatibility (remove special JSON characters)
def sanitize_feature_name(name):
    """Replace special JSON characters with underscores (for LightGBM JSON dumping)."""
    sanitized = re.sub(r"[^\w\-.]", "_", str(name))
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized if sanitized else "feature_" + str(hash(name) % 10000)

X.columns = [sanitize_feature_name(col) for col in X.columns]

y, class_names_array, class_names = encode_labels(merged)
class_names = np.array(class_names)

print(f"  Features: {len(feature_cols):,} (ligand: {len(ligand_cols)}, receptor: {len(receptor_df.columns)}, interactions: {len(interaction_cols)})")
print(f"  Classes: {class_names.tolist()}")

# ============================================================================
# 2. SPLIT PER SEED (one random stratified split per seed for consistency with other scripts)
# ============================================================================
# train_test_split is done inside the seed loop with random_state=seed so each seed
# gets a different train/test split (same as "10 seeds = 10 random splits" elsewhere).

# ============================================================================
# 3. DEFINE BASE MODELS (trial 282 LGB, 76 RF, 111 XGB - same as the cleaned-data single-model runs)
# ============================================================================
print("\n[5/5] Setting up base models (trial 282 LGB, 76 RF, 111 XGB)...")

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
    'n_jobs': -1,
    'verbose': -1,
}
rf_params = {
    'bootstrap': False,
    'criterion': 'entropy',
    'max_depth': 30,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 9,
    'n_estimators': 300,
    'n_jobs': -1,
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
    'n_jobs': -1,
    'verbosity': 0,
}

# ============================================================================
# 4. TRAIN AND EVALUATE STACKING ENSEMBLE (MULTIPLE SEEDS)
# ============================================================================
# Note: StackingClassifier(cv=5) uses plain StratifiedKFold, not grouped by SMILES.
# For full consistency with manuscript (group by canonical SMILES), use manual
# stacking with StratifiedGroupKFold (as in stacking_manual_optuna_process.py).
print("\n[6/6] Training stacking ensemble (10 seeds, one split per seed, raw features)...")
print("  Base models: Random Forest, XGBoost, LightGBM (trial 76/111/282)")
print("  Meta-model: Logistic Regression, cv=5 for meta-features")

all_metrics = []
all_cv_metrics = []  # 5-fold CV on train (mirrors RF script: CV block + test block)
all_per_class_metrics = {class_name: [] for class_name in class_names}
all_predictions = []
overlap_per_seed = []
seed42_stacking_clf = None
seed42_X_test = None
n_train_actual = None
n_test_actual = None

indices = np.arange(len(X))
N_FOLDS_CV = 5

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n  Seed {seed_idx}/10 (seed={seed})...")

    # One random stratified split per seed (10 seeds = 10 different splits)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Overlap report (allowed for random stratified; report for manuscript)
    train_smiles = set(smiles_series.iloc[train_idx].values)
    test_smiles_set = set(smiles_series.iloc[test_idx].values)
    n_overlap = len(train_smiles & test_smiles_set)
    overlap_per_seed.append({'seed': seed, 'n_overlap': n_overlap})
    print(f"    Train: {len(X_train):,}, Test: {len(X_test):,} | SMILES overlap (train ∩ test): {n_overlap} canonical SMILES")

    base_models = [
        ('rf', RandomForestClassifier(**{**rf_params, 'random_state': seed})),
        ('xgb', xgb.XGBClassifier(**{**xgb_params, 'random_state': seed, 'verbosity': 0})),
        ('lgbm', lgb.LGBMClassifier(**{**lgb_params, 'random_state': seed})),
    ]
    meta_model = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)

    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    # 5-fold CV on training portion (mirrors RF script: report CV + test).
    # Note: Nested CV — outer 5 folds here, and StackingClassifier(cv=5) runs internal CV
    # for meta-features in each fold, so runtime is heavy but correct.
    skf = StratifiedKFold(n_splits=N_FOLDS_CV, shuffle=True, random_state=seed)
    cv_scores = cross_validate(
        stacking_clf, X_train, y_train, cv=skf,
        scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'auc_ovr': auc_scorer},
        n_jobs=-1, return_train_score=False
    )
    all_cv_metrics.append({
        'seed': seed,
        'cv_accuracy_mean': float(np.mean(cv_scores['test_accuracy'])),
        'cv_accuracy_std': float(np.std(cv_scores['test_accuracy'], ddof=1)),
        'cv_f1_mean': float(np.mean(cv_scores['test_f1_macro'])),
        'cv_f1_std': float(np.std(cv_scores['test_f1_macro'], ddof=1)),
        'cv_auc_mean': float(np.mean(cv_scores['test_auc_ovr'])),
        'cv_auc_std': float(np.std(cv_scores['test_auc_ovr'], ddof=1)),
    })
    # Std here is across the 5 CV folds for this seed; reported "Mean ± Std" later is across seeds
    print(f"    CV (5-fold on train): F1={all_cv_metrics[-1]['cv_f1_mean']:.4f}±{all_cv_metrics[-1]['cv_f1_std']:.4f}, AUC={all_cv_metrics[-1]['cv_auc_mean']:.4f}±{all_cv_metrics[-1]['cv_auc_std']:.4f}")

    # Final fit on full train, evaluate on test
    stacking_clf.fit(X_train, y_train)
    assert np.array_equal(stacking_clf.classes_, np.arange(N_CLASSES)), (
        "stacking_clf.classes_ must match 0..N_CLASSES-1 for proba alignment"
    )

    if seed == 42:
        seed42_stacking_clf = stacking_clf
        seed42_X_test = X_test.copy()
        n_train_actual = len(X_train)
        n_test_actual = len(X_test)

    y_pred_encoded = stacking_clf.predict(X_test)
    y_pred_proba = stacking_clf.predict_proba(X_test)

    test_smiles_arr = np.asarray(smiles_series.iloc[test_idx].values)
    all_predictions.append({
        'seed': seed,
        'split_type': 'random_stratified_test',
        'y_true': y_test.copy(),
        'y_pred': y_pred_encoded.copy(),
        'y_pred_proba': y_pred_proba.copy(),
        'test_idx': np.asarray(test_idx),
        'test_smiles': test_smiles_arr,
    })

    accuracy = accuracy_score(y_test, y_pred_encoded)
    precision_macro = precision_score(y_test, y_pred_encoded, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred_encoded, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred_encoded, average='macro', zero_division=0)
    macro_auc, per_class_auc = calculate_multiclass_auc(y_test, y_pred_proba, class_names)

    all_metrics.append({
        'seed': seed,
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'auc': macro_auc
    })

    precision_per_class = precision_score(y_test, y_pred_encoded, labels=range(len(class_names)), average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred_encoded, labels=range(len(class_names)), average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred_encoded, labels=range(len(class_names)), average=None, zero_division=0)
    y_test_names = class_names[y_test]
    y_pred_names = class_names[y_pred_encoded]

    for i, class_name in enumerate(class_names):
        specificity = calculate_specificity(y_test_names, y_pred_names, class_name, class_names)
        auc_val = per_class_auc.get(class_name, np.nan)
        all_per_class_metrics[class_name].append({
            'seed': seed,
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i],
            'specificity': specificity,
            'auc': auc_val
        })

print("\n  [OK] All seeds completed!")

# ============================================================================
# 5. COMPUTE STATISTICS ACROSS SEEDS
# ============================================================================
print("\n[7/7] Computing statistics across seeds...")

# Overall metrics (skipna=True so NaN AUC does not tank the mean; warn if any NaN)
# Keys match independent script: precision, recall, f1, auc
metrics_df = pd.DataFrame(all_metrics)
mean_accuracy = metrics_df['accuracy'].mean(skipna=True)
mean_precision = metrics_df['precision'].mean(skipna=True)
mean_recall = metrics_df['recall'].mean(skipna=True)
mean_f1 = metrics_df['f1'].mean(skipna=True)
mean_auc = metrics_df['auc'].mean(skipna=True)
std_accuracy = metrics_df['accuracy'].std(skipna=True)
std_precision = metrics_df['precision'].std(skipna=True)
std_recall = metrics_df['recall'].std(skipna=True)
std_f1 = metrics_df['f1'].std(skipna=True)
std_auc = metrics_df['auc'].std(skipna=True)
if metrics_df['auc'].isna().any():
    print("  [WARNING] Some seeds have NaN macro AUC (check per-seed metrics).")

# CV metrics (5-fold on train, mean ± std across seeds)
cv_df = pd.DataFrame(all_cv_metrics)
cv_accuracy_mean = cv_df['cv_accuracy_mean'].mean()
cv_accuracy_std = cv_df['cv_accuracy_mean'].std(ddof=1)
cv_f1_mean = cv_df['cv_f1_mean'].mean()
cv_f1_std = cv_df['cv_f1_mean'].std(ddof=1)
cv_auc_mean = cv_df['cv_auc_mean'].mean()
cv_auc_std = cv_df['cv_auc_mean'].std(ddof=1)

# Per-class metrics (mean and std, skipna=True)
per_class_summary = {}
for class_name in class_names:
    class_df = pd.DataFrame(all_per_class_metrics[class_name])
    per_class_summary[class_name] = {
        'precision_mean': class_df['precision'].mean(skipna=True),
        'precision_std': class_df['precision'].std(skipna=True),
        'recall_mean': class_df['recall'].mean(skipna=True),
        'recall_std': class_df['recall'].std(skipna=True),
        'f1_mean': class_df['f1'].mean(skipna=True),
        'f1_std': class_df['f1'].std(skipna=True),
        'specificity_mean': class_df['specificity'].mean(skipna=True),
        'specificity_std': class_df['specificity'].std(skipna=True),
        'auc_mean': class_df['auc'].mean(skipna=True),
        'auc_std': class_df['auc'].std(skipna=True)
    }
    if class_df['auc'].isna().any():
        print(f"  [WARNING] Some seeds have NaN AUC for class '{class_name}'.")

# Confusion matrix: representative seed=42 (deterministic, not "last seed")
seed42_pred = next((p for p in all_predictions if p['seed'] == 42), None)
if seed42_pred is not None:
    cm = confusion_matrix(seed42_pred['y_true'], seed42_pred['y_pred'], labels=range(len(class_names)))
else:
    cm = np.zeros((len(class_names), len(class_names)))

# ============================================================================
# 6. DISPLAY RESULTS
# ============================================================================
print("\n" + "="*80)
print("STACKING ENSEMBLE RESULTS (10 Seeds)")
print("="*80)

print(f"\n5-Fold CV on Train (Mean ± Std across seeds):")
print(f"  Accuracy: {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")
print(f"  F1-Score: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
print(f"  AUC:      {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

print(f"\nTest Set Metrics (Mean ± Std):")
print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"  Macro Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"  Macro Recall: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"  Macro F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Macro AUC: {mean_auc:.4f} ± {std_auc:.4f}")

print(f"\nPer-Class Metrics (Mean ± Std):")
print(f"{'Class':<15} {'Precision':<20} {'Recall':<20} {'F1-Score':<20} {'Specificity':<20} {'AUC':<20}")
print('-'*115)
for class_name in class_names:
    summary = per_class_summary[class_name]
    print(f"{class_name:<15} "
          f"{summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}     "
          f"{summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}     "
          f"{summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}     "
          f"{summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}     "
          f"{summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")

print(f"\nConfusion Matrix (representative seed=42):")
print(f"  {'':<15}", end="")
for cls in class_names:
    print(f"{cls[:10]:<12}", end="")
print()
for i, cls in enumerate(class_names):
    print(f"  {cls[:14]:<15}", end="")
    for j, cls2 in enumerate(class_names):
        print(f"{cm[i, j]:<12}", end="")
    print()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n[8/8] Saving results...")

outputs_dir = OUTPUT_DIR
outputs_dir.mkdir(parents=True, exist_ok=True)

# Save overall metrics (with mean and std)
# Use actual sizes from representative seed 42 (matches sklearn split rounding)
n_train = n_train_actual if n_train_actual is not None else int(0.8 * len(X))
n_test = n_test_actual if n_test_actual is not None else int(0.2 * len(X))

results = {
    'model': 'Stacking Ensemble',
    'test_set': 'random_stratified',
    'n_seeds': len(SEEDS),
    'n_train': n_train,
    'n_test': n_test,
    'n_features': len(feature_cols),
    'features_type': 'enriched (ligand + receptor + interactions)',
    'cv_accuracy_mean': float(cv_accuracy_mean),
    'cv_accuracy_std': float(cv_accuracy_std),
    'cv_f1_mean': float(cv_f1_mean),
    'cv_f1_std': float(cv_f1_std),
    'cv_auc_mean': float(cv_auc_mean),
    'cv_auc_std': float(cv_auc_std),
    'test_accuracy_mean': float(mean_accuracy),
    'test_accuracy_std': float(std_accuracy),
    'test_precision_mean': float(mean_precision),
    'test_precision_std': float(std_precision),
    'test_recall_mean': float(mean_recall),
    'test_recall_std': float(std_recall),
    'test_f1_mean': float(mean_f1),
    'test_f1_std': float(std_f1),
    'test_auc_mean': float(mean_auc),
    'test_auc_std': float(std_auc),
    'per_class': {}
}

for class_name in class_names:
    summary = per_class_summary[class_name]
    results['per_class'][class_name] = {
        'precision_mean': float(summary['precision_mean']),
        'precision_std': float(summary['precision_std']),
        'recall_mean': float(summary['recall_mean']),
        'recall_std': float(summary['recall_std']),
        'f1_mean': float(summary['f1_mean']),
        'f1_std': float(summary['f1_std']),
        'specificity_mean': float(summary['specificity_mean']),
        'specificity_std': float(summary['specificity_std']),
        'auc_mean': float(summary['auc_mean']),
        'auc_std': float(summary['auc_std'])
    }

with open(outputs_dir / 'stacking_ensemble_test_set_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save per-seed results (CV + test, like RF script)
per_seed_data = []
for i, m in enumerate(all_metrics):
    row = {**all_cv_metrics[i], **m}
    per_seed_data.append(row)
per_seed_df = pd.DataFrame(per_seed_data)
per_seed_df.to_csv(outputs_dir / 'stacking_ensemble_test_set_per_seed.csv', index=False)

# Save CV summary (mean ± std across seeds)
cv_summary_df = pd.DataFrame([{
    'cv_accuracy_mean': cv_accuracy_mean, 'cv_accuracy_std': cv_accuracy_std,
    'cv_f1_mean': cv_f1_mean, 'cv_f1_std': cv_f1_std,
    'cv_auc_mean': cv_auc_mean, 'cv_auc_std': cv_auc_std,
}])
cv_summary_df.to_csv(outputs_dir / 'stacking_ensemble_test_set_cv_summary.csv', index=False)

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
per_class_per_seed_df.to_csv(outputs_dir / 'stacking_ensemble_per_class_test_set_per_seed.csv', index=False)

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
per_class_summary_df.to_csv(outputs_dir / 'stacking_ensemble_per_class_test_set_mean_std.csv', index=False)

# Save predictions with padding (robust to variable test size across seeds, like RF script)
predictions_file = outputs_dir / 'stacking_ensemble_test_set_predictions.npz'
seed_list = np.array([p['seed'] for p in all_predictions])
test_sizes = np.array([len(p['y_true']) for p in all_predictions])
max_test_size = int(np.max(test_sizes))
n_seeds = len(all_predictions)
test_y_true_all = np.full((n_seeds, max_test_size), -1, dtype=np.int32)
test_y_pred_all = np.full((n_seeds, max_test_size), -1, dtype=np.int32)
test_y_pred_proba_all = np.full((n_seeds, max_test_size, N_CLASSES), np.nan, dtype=np.float32)
test_indices_all = np.full((n_seeds, max_test_size), -1, dtype=np.int64)
# test_smiles: pad with empty string (object array for variable-length strings)
test_smiles_all = np.empty((n_seeds, max_test_size), dtype=object)
test_smiles_all.fill('')
for i, p in enumerate(all_predictions):
    n = len(p['y_true'])
    test_y_true_all[i, :n] = np.asarray(p['y_true'])
    test_y_pred_all[i, :n] = np.asarray(p['y_pred'])
    test_y_pred_proba_all[i, :n, :] = np.asarray(p['y_pred_proba'])
    test_indices_all[i, :n] = np.asarray(p['test_idx'])
    test_smiles_all[i, :n] = np.asarray(p['test_smiles'])
np.savez_compressed(
    predictions_file,
    seed_list=seed_list,
    test_sizes=test_sizes,
    split_types=np.array([p['split_type'] for p in all_predictions]),
    y_true=test_y_true_all,
    y_pred=test_y_pred_all,
    y_pred_proba=test_y_pred_proba_all,
    test_indices_all=test_indices_all,
    test_smiles_all=test_smiles_all,
    class_names=np.array(class_names)
)
print(f"  [OK] Predictions saved: {predictions_file} (test_sizes per seed: {test_sizes.tolist()})")

# Save SMILES overlap per seed (report only; overlap is allowed for random stratified)
overlap_df = pd.DataFrame(overlap_per_seed)
overlap_df.to_csv(outputs_dir / 'stacking_ensemble_test_set_smiles_overlap_per_seed.csv', index=False)
print(f"  [OK] SMILES overlap per seed: {outputs_dir / 'stacking_ensemble_test_set_smiles_overlap_per_seed.csv'}")

# Save summary
summary_data = [{
    'Model': 'Stacking Ensemble',
    'Test_Set': 'Random Stratified',
    'N_Seeds': len(SEEDS),
    'Features': 'Enriched (ligand + receptor + interactions)',
    'Accuracy': f"{mean_accuracy:.4f} ± {std_accuracy:.4f}",
    'Precision': f"{mean_precision:.4f} ± {std_precision:.4f}",
    'Recall': f"{mean_recall:.4f} ± {std_recall:.4f}",
    'F1': f"{mean_f1:.4f} ± {std_f1:.4f}",
    'AUC': f"{mean_auc:.4f} ± {std_auc:.4f}"
}]
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(outputs_dir / 'stacking_ensemble_test_set_summary.csv', index=False)

print(f"\n[OK] Results saved:")
print(f"  - outputs/stacking_ensemble_test_set_results.json")
print(f"  - outputs/stacking_ensemble_test_set_per_seed.csv (CV + test per seed)")
print(f"  - outputs/stacking_ensemble_test_set_cv_summary.csv")
print(f"  - outputs/stacking_ensemble_per_class_test_set_per_seed.csv")
print(f"  - outputs/stacking_ensemble_per_class_test_set_mean_std.csv")
print(f"  - outputs/stacking_ensemble_test_set_summary.csv")
print(f"  - outputs/stacking_ensemble_test_set_predictions.npz")
print(f"  - outputs/stacking_ensemble_test_set_smiles_overlap_per_seed.csv")

# ============================================================================
# 8. SHAP (post-hoc interpretability on meta-model, representative seed=42 only)
# ============================================================================
print("\n[9/9] Computing SHAP values on meta-model (test set, seed=42 only)...")
shap_dir = outputs_dir / 'shap'
shap_dir.mkdir(parents=True, exist_ok=True)
if seed42_stacking_clf is None or seed42_X_test is None:
    print("  [WARNING] No seed=42 stacking model available for SHAP")
else:
    try:
        meta_X_test = seed42_stacking_clf.transform(seed42_X_test)
        meta_model = seed42_stacking_clf.final_estimator_
        n_shap_samples = min(500, len(meta_X_test))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(meta_X_test), n_shap_samples, replace=False)
        meta_X_sample = meta_X_test[sample_idx]
        meta_feature_names = []
        for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
            for cls in class_names:
                meta_feature_names.append(f'{model_name}_{cls}')
        explainer = shap.LinearExplainer(meta_model, meta_X_test)
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
        pd.DataFrame(meta_X_sample, columns=meta_feature_names).to_csv(
            shap_dir / 'stacking_test_set_shap_meta_X_sample.csv', index=False
        )
        np.savez(
            shap_dir / 'stacking_test_set_shap_values.npz',
            shap_values=shap_array,
            expected_value=expected_value_array,
            sample_indices=sample_idx,
            feature_names=np.array(meta_feature_names),
            class_names=np.array(class_names),
        )
        print(f"  [OK] SHAP saved to {shap_dir / 'stacking_test_set_shap_values.npz'}")
    except Exception as e:
        print(f"  [WARNING] SHAP failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("[SUCCESS] STACKING ENSEMBLE EVALUATION COMPLETE!")
print("="*80)














