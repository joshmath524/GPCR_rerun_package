#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Scaffold Split Evaluation with Best Optuna Hyperparameters

Uses fixed best-trial hyperparameters (Optuna trial 111, n_estimators=1843), same as baseline and random stratified — hardcoded in code, no CSV.

Scaffold-based train/test split (no scaffold overlap), seed-dependent split.
10 seeds × 5-fold CV on training portion (scaffold-leak-free) + test set evaluation.

- Scaffolds computed once; per seed, unique scaffolds shuffled with RandomState(seed), 80/20 by sample count.
- CV uses StratifiedGroupKFold (or GroupKFold) so same scaffold never in train and val.
- No feature scaling (tree models).
- Outputs: per-seed CV/test CSVs, mean±std summary, padded predictions NPZ, SHAP (first seed only).
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb
import shap
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_utilities import align_proba, load_receptor_ligand_data

print("="*80)
print("XGBOOST SCAFFOLD SPLIT EVALUATION")
print("="*80)
print("\nScaffold-based split (no scaffold overlap between train and test)")
print("10 seeds × 5-fold CV on training portion + test set evaluation")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

SEEDS = list(range(42, 52))  # 10 seeds: 42-51
N_FOLDS = 5
N_CLASSES = 3
OUTPUT_DIR = Path(__file__).parent  # Save to same folder as script
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
# Same as baseline and random stratified — hardcoded, no CSV.
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
# Step 2: Load Data and Prepare Features (Same as RF)
# ============================================================================

print("\n[STEP 2] Loading data and preparing features...")
print("-"*80)

def aggregate_receptor_features(receptor_name: str):
    """Extract all 31 receptor features from pocket residues CSV file."""
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent  # Go up two levels to workspace root
    possible_paths = [
        workspace_root / 'Josh_Receptor_Features' / receptor_name,  # Workspace root
        script_dir.parent / 'Josh_Receptor_Features' / receptor_name,  # ML_code folder
        Path(r'c:\Users\Piano\Downloads\Josh_Receptor_Features') / receptor_name,  # Absolute path
    ]
    
    pocket = None
    for p in possible_paths:
        if p.exists():
            pocket = p
            break
    
    if pocket is None:
        return None
    
    files = list(pocket.glob('*_pocket_residues_with_conservation.csv'))
    if not files:
        return None
    
    df = pd.read_csv(files[0])
    total = len(df)
    if total == 0:
        return None
    
    def count_category(column: str, value: str) -> int:
        if column not in df.columns:
            return 0
        series = df[column].astype(str).str.lower()
        return int((series == value).sum())
    
    def sum_boolean(column: str) -> int:
        if column not in df.columns:
            return 0
        series = df[column]
        if series.dtype == bool:
            return int(series.sum())
        return int(pd.to_numeric(series, errors='coerce').fillna(0).sum())
    
    feats = {
        'num_residues': total,
        'num_aromatic': int(df['aromatic'].sum()) if 'aromatic' in df.columns else 0,
        'num_acidic': int(df['acidic'].sum()) if 'acidic' in df.columns else 0,
        'num_basic': int(df['basic'].sum()) if 'basic' in df.columns else 0,
        'num_charge_positive': count_category('charge', 'positive'),
        'num_charge_negative': count_category('charge', 'negative'),
        'num_charge_neutral': count_category('charge', 'neutral'),
        'num_polar': count_category('polarity', 'polar'),
        'num_nonpolar': count_category('polarity', 'nonpolar'),
        'num_size_small': count_category('size', 'small'),
        'num_size_medium': count_category('size', 'medium'),
        'num_size_large': count_category('size', 'large'),
        'num_sulfur': sum_boolean('sulfur'),
        'num_hydroxyl': sum_boolean('hydroxyl'),
        'num_amide': sum_boolean('amide'),
    }
    
    nr = max(feats['num_residues'], 1)
    
    feats['aromatic_ratio'] = feats['num_aromatic'] / nr
    feats['basic_ratio'] = feats['num_basic'] / nr
    feats['acidic_ratio'] = feats['num_acidic'] / nr
    feats['charge_positive_ratio'] = feats['num_charge_positive'] / nr
    feats['charge_negative_ratio'] = feats['num_charge_negative'] / nr
    feats['charge_neutral_ratio'] = feats['num_charge_neutral'] / nr
    feats['polar_ratio'] = feats['num_polar'] / nr
    feats['nonpolar_ratio'] = feats['num_nonpolar'] / nr
    feats['size_small_ratio'] = feats['num_size_small'] / nr
    feats['size_medium_ratio'] = feats['num_size_medium'] / nr
    feats['size_large_ratio'] = feats['num_size_large'] / nr
    feats['sulfur_ratio'] = feats['num_sulfur'] / nr
    feats['hydroxyl_ratio'] = feats['num_hydroxyl'] / nr
    feats['amide_ratio'] = feats['num_amide'] / nr
    
    feats['avg_distance'] = float(df['distance_to_ligand'].mean()) if 'distance_to_ligand' in df.columns else 0.0
    feats['avg_conservation'] = float(df['conservation_score'].mean()) if 'conservation_score' in df.columns else 0.0
    
    return feats

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

print("\nLoading ligand data from individual receptor files (_NEW only, validated)...")
receptors_to_load = list(receptor_df.index) if len(receptor_df) > 0 else RECEPTORS
data_df, ligand_cols = load_receptor_ligand_data(receptors_to_load, OUTPUT_DIR)
print(f"  [OK] {len(data_df):,} total samples")

if len(receptor_df) > 0:
    merged = data_df.merge(receptor_df, on='Receptor', how='left')
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
else:
    merged = data_df

print(f"  [OK] {len(ligand_cols)} ligand descriptor columns")
print(f"  [OK] {len(receptor_df.columns) if len(receptor_df) > 0 else 0} receptor feature columns")

interaction_pairs = [
    ('LogP', 'aromatic_ratio'), ('LogP', 'num_basic'), ('TPSA', 'acidic_ratio'),
    ('HBD', 'num_hydroxyl'), ('HBA', 'num_basic'), ('HBA', 'num_amide'),
    ('AromaticRings', 'aromatic_ratio'), ('AromaticRings', 'num_aromatic'),
    ('FormalCharge', 'num_charge_positive'), ('FormalCharge', 'num_charge_negative'),
    ('MolWt', 'num_residues'), ('RotatableBonds', 'num_residues'),
    ('LogP', 'avg_conservation'), ('TPSA', 'avg_conservation'),
]

interaction_cols = []
if len(receptor_df) > 0:
    for lig, rec in interaction_pairs:
        if lig in merged.columns and rec in merged.columns:
            name = f'INT_{lig}_X_{rec}'
            merged[name] = merged[lig] * merged[rec]
            interaction_cols.append(name)

print(f"  [OK] {len(interaction_cols)} interaction terms")

feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols if len(receptor_df) > 0 else ligand_cols + interaction_cols
X_temp = merged[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
X = X_temp.select_dtypes(include=[np.number])
final_feature_cols = X.columns.tolist()

print(f"  [OK] Total features (numeric only): {len(final_feature_cols)}")

encoder = LabelEncoder()
y = encoder.fit_transform(merged['Label_Type'])
class_names = encoder.classes_
n_classes = len(class_names)

print(f"  [OK] Classes: {list(class_names)} (n={n_classes})")

expected_mapping = {"agonist": 0, "antagonist": 1, "inactive": 2}
actual_mapping = {name: int(encoder.transform([name])[0]) for name in expected_mapping.keys()}
if actual_mapping != expected_mapping:
    print(f"  [WARNING] Label encoding mismatch! Expected: {expected_mapping}, Got: {actual_mapping}")

if 'SMILES' in merged.columns:
    smiles_series = merged['SMILES'].astype(str)
else:
    raise ValueError("SMILES column not found! Required for scaffold splitting.")

print(f"  [OK] Total samples: {len(X):,}")

# ============================================================================
# Step 3: Compute Bemis–Murcko scaffolds once (outside seed loop)
# ============================================================================

print("\n[STEP 3] Computing Bemis–Murcko scaffolds (once for all samples)...")
print("-"*80)

def compute_scaffold(smiles: str, fallback: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return f'NO_SCAFFOLD_{fallback}'
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

scaffolds = []
total_smiles = len(smiles_series)
for idx, smi in enumerate(smiles_series.tolist(), 1):
    if idx % 5000 == 0 or idx == total_smiles:
        print(f"  Progress: {idx:,}/{total_smiles:,} ({idx/total_smiles*100:.1f}%)", flush=True)
    scaffolds.append(compute_scaffold(smi, str(idx)))

scaffold_series = pd.Series(scaffolds, index=X.index)
unique_scaffolds = scaffold_series.unique()
print(f"  [OK] Unique scaffolds: {len(unique_scaffolds):,} (split will be seed-dependent, 80/20 by samples)")

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

def evaluate_model(y_true, y_pred, y_proba, n_classes):
    """Evaluate model and return all metrics. AUC: handle missing classes via label_binarize; use np.nan if computation fails."""
    metrics = {
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'specificity': compute_macro_specificity(y_true, y_pred, n_classes),
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }
    unique_classes = np.unique(y_true)
    if len(unique_classes) < n_classes:
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            metrics['auc'] = float(roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr'))
        except (ValueError, Exception):
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=unique_classes))
            except (ValueError, Exception):
                metrics['auc'] = np.nan
    else:
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
        except ValueError as e:
            print(f"\n      [WARNING] AUC computation failed: {e}", flush=True)
            metrics['auc'] = np.nan
    return metrics

# ============================================================================
# Step 5: Evaluation Loop (10 Seeds × scaffold-leak-free CV + Test Set)
# ============================================================================

print("\n[STEP 5] Evaluation Loop (10 seeds × scaffold-leak-free CV + test set)...")
print("-"*80)

all_results_cv = []
all_results_test = []
all_predictions = []
first_seed_model = None
first_seed_X_test = None

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\nSeed {seed_idx}/{len(SEEDS)} (seed={seed})...", flush=True)
    rng = np.random.RandomState(seed)

    # --- Seed-dependent 80/20 scaffold split ---
    shuffled_scaffolds = rng.permutation(unique_scaffolds)
    train_scaffolds = set()
    train_count = 0
    train_target = int(0.8 * len(X))
    for sc in shuffled_scaffolds:
        n = (scaffold_series == sc).sum()
        if train_count < train_target:
            train_scaffolds.add(sc)
            train_count += n
    train_mask = scaffold_series.isin(train_scaffolds)
    test_mask = ~train_mask
    assert not (train_mask & test_mask).any(), "Train/test mask overlap"
    overlap = set(scaffold_series[train_mask].unique()) & set(scaffold_series[test_mask].unique())
    assert len(overlap) == 0, f"Scaffold overlap train/test: {len(overlap)}"

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    print(f"  Split: train {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%), test {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    # --- Scaffold-leak-free CV ---
    scaffold_train = scaffold_series[train_mask]
    groups_train = pd.Categorical(scaffold_train).codes
    if StratifiedGroupKFold is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False)
    else:
        cv_splitter = GroupKFold(n_splits=N_FOLDS)
    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups=groups_train), 1):
        print(f"    Fold {fold}/{N_FOLDS} - train {len(train_idx):,}, val {len(val_idx):,}...", end=' ', flush=True)
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = np.asarray(y_train)[train_idx]
        y_val_fold = np.asarray(y_train)[val_idx]
        model_params = best_params.copy()
        model_params['random_state'] = seed
        model_fold = xgb.XGBClassifier(**model_params)
        model_fold.fit(X_train_fold, y_train_fold)
        y_pred = model_fold.predict(X_val_fold)
        y_proba = model_fold.predict_proba(X_val_fold)
        y_proba = align_proba(y_proba, model_fold.classes_, n_classes)
        metrics = evaluate_model(y_val_fold, y_pred, y_proba, n_classes)
        cv_results.append(metrics)
        auc_f = metrics['auc']; auc_str = f"{auc_f:.4f}" if not (isinstance(auc_f, (int, float)) and np.isnan(auc_f)) else "NaN"
        print(f"F1={metrics['f1']:.4f}, AUC={auc_str}", flush=True)

    cv_metrics = {
        'recall': np.mean([r['recall'] for r in cv_results]),
        'precision': np.mean([r['precision'] for r in cv_results]),
        'f1': np.mean([r['f1'] for r in cv_results]),
        'auc': float(np.nanmean([r['auc'] for r in cv_results])),
        'specificity': np.mean([r['specificity'] for r in cv_results]),
        'accuracy': np.mean([r['accuracy'] for r in cv_results]),
        'seed': seed
    }
    all_results_cv.append(cv_metrics)
    cv_auc = cv_metrics['auc']; cv_auc_str = f"{cv_auc:.4f}" if not (isinstance(cv_auc, (int, float)) and np.isnan(cv_auc)) else "NaN"
    print(f"  CV Results - F1: {cv_metrics['f1']:.4f}, AUC: {cv_auc_str}", flush=True)

    # --- Train on full training set, evaluate on test (no scaling) ---
    print(f"  Final test - train {len(X_train):,}, test {len(X_test):,}...", end=' ', flush=True)
    model_params = best_params.copy()
    model_params['random_state'] = seed
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)
    y_proba_test = align_proba(y_proba_test, model.classes_, n_classes)
    y_test_arr = np.asarray(y_test)
    test_metrics = evaluate_model(y_test_arr, y_pred_test, y_proba_test, n_classes)
    test_metrics['seed'] = seed
    all_results_test.append(test_metrics)
    all_predictions.append({
        'seed': seed,
        'y_true': y_test_arr.copy(),
        'y_pred': y_pred_test.copy(),
        'y_pred_proba': y_proba_test.copy(),
    })
    if seed == SEEDS[0]:
        first_seed_model = model
        first_seed_X_test = X_test.copy()
    auc_str = f"{test_metrics['auc']:.4f}" if not (isinstance(test_metrics['auc'], (int, float)) and np.isnan(test_metrics['auc'])) else "NaN"
    print(f"F1={test_metrics['f1']:.4f}, AUC={auc_str}", flush=True)

# ============================================================================
# Step 6: Aggregate Results and Save
# ============================================================================

print("\n[STEP 6] Aggregating results and saving...")
print("-"*80)

cv_df = pd.DataFrame(all_results_cv)
test_df = pd.DataFrame(all_results_test)

metric_names = ['recall', 'precision', 'f1', 'auc', 'specificity', 'accuracy']

summary_dict = {}
for metric in metric_names:
    use_nan = (metric == 'auc')
    cv_mean = float(np.nanmean(cv_df[metric])) if use_nan else cv_df[metric].mean()
    cv_std = float(np.nanstd(cv_df[metric])) if use_nan else cv_df[metric].std()
    test_mean = float(np.nanmean(test_df[metric])) if use_nan else test_df[metric].mean()
    test_std = float(np.nanstd(test_df[metric])) if use_nan else test_df[metric].std()
    
    summary_dict[f'{metric}_scaffold_cv_mean'] = cv_mean
    summary_dict[f'{metric}_scaffold_cv_std'] = cv_std
    summary_dict[f'{metric}_scaffold_test_mean'] = test_mean
    summary_dict[f'{metric}_scaffold_test_std'] = test_std

summary_df = pd.DataFrame([summary_dict])
summary_df.index = ['XGBoost_scaffold']

cv_file = OUTPUT_DIR / 'xgb_scaffold_per_seed_cv.csv'
test_file = OUTPUT_DIR / 'xgb_scaffold_per_seed_test.csv'
summary_file = OUTPUT_DIR / 'xgb_scaffold_mean_std.csv'

cv_df.to_csv(cv_file, index=False)
test_df.to_csv(test_file, index=False)
summary_df.to_csv(summary_file, index=True)

print(f"  [OK] Saved CV results: {cv_file}")
print(f"  [OK] Saved test results: {test_file}")
print(f"  [OK] Saved summary: {summary_file}")

predictions_file = OUTPUT_DIR / 'xgb_scaffold_predictions.npz'
test_sizes = np.array([len(p['y_true']) for p in all_predictions])
max_test_size = int(test_sizes.max())
n_seeds = len(all_predictions)
test_y_true = np.full((n_seeds, max_test_size), -1, dtype=np.int32)
test_y_pred = np.full((n_seeds, max_test_size), -1, dtype=np.int32)
test_y_pred_proba = np.full((n_seeds, max_test_size, N_CLASSES), np.nan, dtype=np.float32)
for i, p in enumerate(all_predictions):
    n = len(p['y_true'])
    test_y_true[i, :n] = p['y_true']
    test_y_pred[i, :n] = p['y_pred']
    test_y_pred_proba[i, :n, :] = p['y_pred_proba']
np.savez(
    predictions_file,
    seeds=np.array([p['seed'] for p in all_predictions]),
    test_sizes=test_sizes,
    y_true=test_y_true,
    y_pred=test_y_pred,
    y_pred_proba=test_y_pred_proba,
    split_types=np.array(['scaffold_80_20'] * len(all_predictions)),
    class_names=np.array(class_names),
)
print(f"  [OK] Saved test set predictions: {predictions_file} (test_sizes per seed: {test_sizes.tolist()})")

print("\n" + "="*80)
print("RESULTS - MEAN ± STANDARD DEVIATION (across 10 seeds)")
print("="*80)

print("\n5-Fold CV on Training Portion:")
print("-"*80)
for metric in metric_names:
    mean = summary_dict[f'{metric}_scaffold_cv_mean']
    std = summary_dict[f'{metric}_scaffold_cv_std']
    print(f"  {metric.capitalize():15s}: {mean:.4f} ± {std:.4f}")

print("\nTest Set (Unseen Scaffolds):")
print("-"*80)
for metric in metric_names:
    mean = summary_dict[f'{metric}_scaffold_test_mean']
    std = summary_dict[f'{metric}_scaffold_test_std']
    print(f"  {metric.capitalize():15s}: {mean:.4f} ± {std:.4f}")

print("\n" + "="*80)
print("Evaluation complete!")
print("="*80)

# ============================================================================
# Step 7: SHAP (POST-HOC, first seed only)
# ============================================================================
print("\n[STEP 7] Computing SHAP values (post-hoc, first seed)...")
print("-"*80)
if first_seed_model is None:
    print("  [WARNING] No first-seed model available for SHAP")
else:
    try:
        n_shap = min(500, len(first_seed_X_test))
        sample_idx = np.random.RandomState(42).choice(len(first_seed_X_test), n_shap, replace=False)
        X_test_sample = first_seed_X_test.iloc[sample_idx]
        explainer = shap.TreeExplainer(first_seed_model)
        shap_vals = explainer.shap_values(X_test_sample)
        ev = explainer.expected_value
        ev_arr = np.array(ev) if np.ndim(ev) else np.array([ev])
        if isinstance(shap_vals, list):
            shap_arr = np.array(shap_vals)
        else:
            shap_arr = shap_vals
        shap_dir = OUTPUT_DIR / 'shap'
        shap_dir.mkdir(parents=True, exist_ok=True)
        X_test_sample.to_csv(shap_dir / 'xgb_scaffold_shap_X_test_sample.csv', index=False)
        np.savez(
            shap_dir / 'xgb_scaffold_shap_values.npz',
            shap_values=shap_arr,
            expected_value=ev_arr,
            sample_indices=sample_idx,
            feature_names=np.array(final_feature_cols),
            class_names=np.array(class_names),
            seed_used=np.array([SEEDS[0]]),
        )
        print(f"  [OK] SHAP saved to {shap_dir / 'xgb_scaffold_shap_values.npz'}")
    except Exception as e:
        print(f"  [WARNING] SHAP failed: {e}")
        import traceback
        traceback.print_exc()



