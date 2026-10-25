#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for baseline, Optuna, and LORO evaluation scripts.
Provides consistent hyperparameter loading, feature construction, SMILES canonicalization,
label encoding, and probability alignment functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


# ============================================================================
# Hyperparameter Loading
# ============================================================================

def load_best_hyperparameters_lightgbm(optuna_csv_path: Path) -> Dict[str, Any]:
    """Load best LightGBM hyperparameters from Optuna CSV."""
    if not optuna_csv_path.exists():
        raise FileNotFoundError(f"Optuna CSV not found: {optuna_csv_path}")
    
    df = pd.read_csv(optuna_csv_path)
    if 'value' not in df.columns:
        raise ValueError(f"No 'value' column in {optuna_csv_path}")
    
    # Filter to only COMPLETE trials with valid values
    if 'state' in df.columns:
        df_complete = df[df['state'] == 'COMPLETE'].copy()
        print(f"  [INFO] Filtering to {len(df_complete)} COMPLETE trials (out of {len(df)} total)")
    else:
        df_complete = df.copy()
        print(f"  [INFO] No 'state' column found, using all {len(df)} trials")
    
    # Filter out NaN/empty values
    df_valid = df_complete[df_complete['value'].notna()].copy()
    if len(df_valid) == 0:
        raise ValueError(f"No valid trials with non-NaN values found in {optuna_csv_path}")
    
    print(f"  [INFO] {len(df_valid)} trials with valid values")
    print(f"  [INFO] value min/max: {df_valid['value'].min():.4f} / {df_valid['value'].max():.4f}")
    
    # Find best trial (highest value = best macro F1)
    best_idx = df_valid['value'].idxmax()
    best_row = df_valid.loc[best_idx]
    best_trial_number = best_row.get('number', best_idx) if 'number' in best_row.index else best_idx
    print(f"  [OK] Best trial {best_trial_number} (row {best_idx}): F1={best_row['value']:.4f}")
    
    # Parse max_depth: handle None, NaN, empty, or "None" string
    md = best_row.get('params_max_depth', None)
    try:
        if pd.isna(md) or str(md).strip().lower() in ("none", ""):
            max_depth_val = -1  # LightGBM: -1 means no limit
        else:
            max_depth_val = int(md)
    except (ValueError, TypeError):
        max_depth_val = -1
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,  # Will be set correctly in caller
        'metric': 'None',
        'verbosity': -1,
        'learning_rate': float(best_row.get('params_learning_rate', 0.05)),
        'num_leaves': int(best_row.get('params_num_leaves', 31)),
        'max_depth': max_depth_val,
        'min_child_samples': int(best_row.get('params_min_child_samples', 20)),
        'subsample': float(best_row.get('params_subsample', 0.8)),
        'colsample_bytree': float(best_row.get('params_colsample_bytree', 0.8)),
        'lambda_l1': float(best_row.get('params_lambda_l1', 0.0)),
        'lambda_l2': float(best_row.get('params_lambda_l2', 0.0)),
        'n_estimators': int(best_row.get('params_n_estimators', 200)),
        'random_state': 42,  # Will be overridden per seed
        'n_jobs': -1,
    }
    
    return params


def load_best_hyperparameters_xgboost(optuna_csv_path: Path) -> Dict[str, Any]:
    """Load best XGBoost hyperparameters from Optuna CSV."""
    if not optuna_csv_path.exists():
        raise FileNotFoundError(f"Optuna CSV not found: {optuna_csv_path}")
    
    df = pd.read_csv(optuna_csv_path)
    if 'value' not in df.columns:
        raise ValueError(f"No 'value' column in {optuna_csv_path}")
    
    # Filter to only COMPLETE trials with valid values
    if 'state' in df.columns:
        df_complete = df[df['state'] == 'COMPLETE'].copy()
        print(f"  [INFO] Filtering to {len(df_complete)} COMPLETE trials (out of {len(df)} total)")
    else:
        df_complete = df.copy()
        print(f"  [INFO] No 'state' column found, using all {len(df)} trials")
    
    # Filter out NaN/empty values
    df_valid = df_complete[df_complete['value'].notna()].copy()
    if len(df_valid) == 0:
        raise ValueError(f"No valid trials with non-NaN values found in {optuna_csv_path}")
    
    print(f"  [INFO] {len(df_valid)} trials with valid values")
    print(f"  [INFO] value min/max: {df_valid['value'].min():.4f} / {df_valid['value'].max():.4f}")
    
    # Find best trial (highest value = best macro F1)
    best_idx = df_valid['value'].idxmax()
    best_row = df_valid.loc[best_idx]
    best_trial_number = best_row.get('number', best_idx) if 'number' in best_row.index else best_idx
    print(f"  [OK] Best trial {best_trial_number} (row {best_idx}): F1={best_row['value']:.4f}")
    
    params = {
        'n_estimators': int(best_row.get('params_n_estimators', 200)),
        'learning_rate': float(best_row.get('params_learning_rate', 0.05)),
        'max_depth': int(best_row.get('params_max_depth', 6)),
        'min_child_weight': float(best_row.get('params_min_child_weight', 1.0)),
        'subsample': float(best_row.get('params_subsample', 0.8)),
        'colsample_bytree': float(best_row.get('params_colsample_bytree', 0.8)),
        'gamma': float(best_row.get('params_gamma', 0.0)),
        'reg_alpha': float(best_row.get('params_reg_alpha', 0.0)),
        'reg_lambda': float(best_row.get('params_reg_lambda', 0.0)),
        'objective': 'multi:softprob',
        'num_class': 3,  # Will be set correctly in caller
        'eval_metric': 'mlogloss',
        'random_state': 42,  # Will be overridden per seed
        'n_jobs': -1,
        'tree_method': 'hist',
    }
    
    return params


def load_best_hyperparameters_randomforest(optuna_csv_path: Path) -> Dict[str, Any]:
    """Load best Random Forest hyperparameters from Optuna CSV."""
    if not optuna_csv_path.exists():
        raise FileNotFoundError(f"Optuna CSV not found: {optuna_csv_path}")
    
    df = pd.read_csv(optuna_csv_path)
    if 'value' not in df.columns:
        raise ValueError(f"No 'value' column in {optuna_csv_path}")
    
    # Filter to only COMPLETE trials with valid values
    if 'state' in df.columns:
        df_complete = df[df['state'] == 'COMPLETE'].copy()
        print(f"  [INFO] Filtering to {len(df_complete)} COMPLETE trials (out of {len(df)} total)")
    else:
        df_complete = df.copy()
        print(f"  [INFO] No 'state' column found, using all {len(df)} trials")
    
    # Filter out NaN/empty values
    df_valid = df_complete[df_complete['value'].notna()].copy()
    if len(df_valid) == 0:
        raise ValueError(f"No valid trials with non-NaN values found in {optuna_csv_path}")
    
    print(f"  [INFO] {len(df_valid)} trials with valid values")
    print(f"  [INFO] value min/max: {df_valid['value'].min():.4f} / {df_valid['value'].max():.4f}")
    
    # Find best trial (highest value = best macro F1)
    best_idx = df_valid['value'].idxmax()
    best_row = df_valid.loc[best_idx]
    best_trial_number = best_row.get('number', best_idx) if 'number' in best_row.index else best_idx
    print(f"  [OK] Best trial {best_trial_number} (row {best_idx}): F1={best_row['value']:.4f}")
    
    # Parse class_weight: normalize None/NaN/empty to None
    cw = best_row.get('params_class_weight', None)
    if pd.isna(cw) or str(cw).strip().lower() in ("", "none", "nan"):
        class_weight_val = None
    else:
        class_weight_val = str(cw).strip()  # e.g., "balanced"
    
    # Parse max_depth: handle None, NaN, empty, or "None" string
    md = best_row.get('params_max_depth', None)
    try:
        if pd.isna(md) or str(md).strip().lower() in ("none", ""):
            max_depth_val = None
        else:
            max_depth_val = int(md)
    except (ValueError, TypeError):
        max_depth_val = None
    
    # Parse max_features: handle string or numeric
    mf = best_row.get('params_max_features', 'sqrt')
    if pd.isna(mf) or mf == '':
        max_features_val = 'sqrt'
    elif isinstance(mf, str):
        if mf.lower() in ['sqrt', 'log2']:
            max_features_val = mf.lower()
        else:
            try:
                max_features_val = float(mf)
            except (ValueError, TypeError):
                max_features_val = 'sqrt'
    else:
        max_features_val = mf
    
    # Parse bootstrap: handle bool("False") == True bug
    boot = best_row.get('params_bootstrap', True)
    if pd.isna(boot) or boot == "":
        bootstrap_val = True
    elif isinstance(boot, (bool, np.bool_)):
        bootstrap_val = bool(boot)
    else:
        s = str(boot).strip().lower()
        if s in ("true", "1", "yes"):
            bootstrap_val = True
        elif s in ("false", "0", "no"):
            bootstrap_val = False
        else:
            bootstrap_val = True
    
    params = {
        'n_estimators': int(best_row.get('params_n_estimators', 300)),
        'max_depth': max_depth_val,
        'min_samples_split': int(best_row.get('params_min_samples_split', 2)),
        'min_samples_leaf': int(best_row.get('params_min_samples_leaf', 1)),
        'max_features': max_features_val,
        'criterion': str(best_row.get('params_criterion', 'gini')),
        'bootstrap': bootstrap_val,
        'class_weight': class_weight_val,
        'random_state': 42,  # Will be overridden per seed
        'n_jobs': -1,
        'oob_score': False,
    }
    
    # Handle max_samples if bootstrap is True
    if bootstrap_val and 'params_max_samples' in best_row:
        params['max_samples'] = float(best_row.get('params_max_samples', 1.0))
    
    return params


# ============================================================================
# Probability Alignment
# ============================================================================

def align_proba(y_proba: np.ndarray, model_classes: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Align probability predictions to ensure shape (n_samples, n_classes).
    Handles cases where a class might be missing in training fold.
    
    Args:
        y_proba: Probability predictions from model (shape may be (n_samples, k) where k < n_classes)
        model_classes: Classes present in the model (from model.classes_)
        n_classes: Expected number of classes
    
    Returns:
        Aligned probability array of shape (n_samples, n_classes)
    """
    if y_proba.shape[1] == n_classes:
        return y_proba
    
    # Create full probability matrix with zeros for missing classes
    proba_full = np.zeros((y_proba.shape[0], n_classes))
    for j, cls in enumerate(model_classes):
        proba_full[:, int(cls)] = y_proba[:, j]
    
    return proba_full


# ============================================================================
# Data Loading and Feature Construction
# ============================================================================

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

LIGAND_FILE_KINDS = [
    ('agonist', 'agonist_enriched'),
    ('antagonist', 'antagonist_enriched'),
    ('inactive', 'non_active_compounds'),
]


def find_receptor_folder(receptor: str) -> Path:
    """Return WORKSPACE_ROOT/<receptor>; raise if it does not exist (no fallback folders)."""
    folder = WORKSPACE_ROOT / receptor
    if not folder.is_dir():
        raise FileNotFoundError(f"Receptor folder not found: {folder}")
    return folder


def drop_duplicate_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Drop pandas-renamed repeat columns (e.g. 'MolWt.1', 'MolWt.2') and keep the original,
    so every file ends up with the same descriptor set. Warn when a copy's values differ
    (e.g. 3D shape descriptors recomputed from a different random conformer).
    """
    to_drop, differing = [], []
    for col in df.columns:
        base, sep, suffix = str(col).rpartition('.')
        if not sep or not suffix.isdigit() or base not in df.columns:
            continue
        to_drop.append(col)
        a = pd.to_numeric(df[base], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True):
            differing.append(base)
    if differing:
        names = sorted(set(differing))
        print(f"    [WARNING] {source}: repeat columns differ from the original for {len(names)} descriptors "
              f"({', '.join(names[:5])}{', ...' if len(names) > 5 else ''}); keeping the original", flush=True)
    return df.drop(columns=to_drop)


def validate_ligand_descriptors(df: pd.DataFrame, source: str) -> None:
    """
    Raise if a ligand table has no rows, no descriptors, or any compound whose
    descriptors are missing. Missing descriptors become all-zero feature rows
    after fillna(0), which the models can exploit as a label shortcut.
    """
    if len(df) == 0:
        raise ValueError(f"{source}: file has no rows")
    if 'MolWt' not in df.columns:
        raise ValueError(
            f"{source}: no 'MolWt' column - RDKit/Mordred descriptors have not been computed for this file"
        )
    mol_wt = pd.to_numeric(df['MolWt'], errors='coerce')
    bad = mol_wt.isna() | (mol_wt <= 0)
    if bad.any():
        examples = df.loc[bad, 'SMILES'].head(3).tolist() if 'SMILES' in df.columns else []
        raise ValueError(
            f"{source}: {int(bad.sum())} of {len(df)} compounds have missing or zero MolWt "
            f"(descriptors not computed). Example SMILES: {examples}"
        )
    descriptor_cols = [c for c in df.columns if not should_exclude_column(c)]
    numeric = df[descriptor_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    blank = (numeric == 0).all(axis=1)
    if blank.any():
        raise ValueError(f"{source}: {int(blank.sum())} of {len(df)} compounds have all-zero/blank descriptors")


def load_table(base_path: Path) -> Optional[pd.DataFrame]:
    """
    Load {stem}_NEW.csv or {stem}_NEW.xlsx only. Plain {stem}.csv/.xlsx files are never read.
    Duplicate '.1' columns are dropped and descriptors are validated.
    
    Args:
        base_path: Base path (e.g., Path('folder/A3_agonist_enriched.csv')); stem is used to form {stem}_NEW.csv / .xlsx
    
    Returns:
        DataFrame (with attrs['source_path'] set) or None if no _NEW file exists
    """
    stem = base_path.stem[:-4] if base_path.stem.endswith('_NEW') else base_path.stem
    candidates = [
        base_path.with_name(stem + '_NEW.csv'),
        base_path.with_name(stem + '_NEW.xlsx'),
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False) if path.suffix.lower() == '.csv' else pd.read_excel(path)
        df = drop_duplicate_columns(df, path.name)
        validate_ligand_descriptors(df, path.name)
        df.attrs['source_path'] = str(path)
        return df
    return None


def load_receptor_ligand_data(
    receptors: List[str],
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and validate the agonist/antagonist/inactive _NEW files for every receptor.
    
    Prints the actual row count per class for each receptor and, if output_dir is given,
    writes <script name>_data_manifest.csv (receptor, class, file, rows) so every run records
    exactly which files and how many compounds it used.
    
    Returns:
        (data_df with 'Receptor' and 'Label_Type' columns, sorted ligand feature column names)
    """
    rows = []
    ligand_cols = set()
    manifest = []
    file_columns = {}
    for idx, rec in enumerate(receptors, 1):
        folder = find_receptor_folder(rec)
        counts = []
        for label, kind in LIGAND_FILE_KINDS:
            table = load_table(folder / f'{rec}_{kind}.csv')
            if table is None:
                manifest.append({'receptor': rec, 'class': label, 'file': '', 'rows': 0})
                counts.append(f'{label}=no _NEW file')
                continue
            source = Path(table.attrs.pop('source_path'))
            table['Receptor'] = rec
            table['Label_Type'] = label
            descriptor_cols = frozenset(c for c in table.columns if not should_exclude_column(c))
            file_columns[source.name] = descriptor_cols
            ligand_cols.update(descriptor_cols)
            rows.append(table)
            manifest.append({
                'receptor': rec,
                'class': label,
                'file': str(source.relative_to(WORKSPACE_ROOT)),
                'rows': len(table),
            })
            counts.append(f'{label}={len(table)}')
        print(f"  [{idx}/{len(receptors)}] {rec}: " + ', '.join(counts), flush=True)

    if not rows:
        raise ValueError("No ligand data loaded! Check receptor folders.")

    # A file missing columns that other files have would get 0 for them after fillna(0)
    from collections import Counter
    reference = Counter(file_columns.values()).most_common(1)[0][0]
    mismatched = {
        name: (len(reference - cols), len(cols - reference))
        for name, cols in file_columns.items() if cols != reference
    }
    if mismatched:
        details = '; '.join(f"{name}: {m} missing, {e} extra" for name, (m, e) in mismatched.items())
        raise ValueError(
            f"{len(mismatched)} files do not have the same {len(reference)} descriptor columns as the rest: {details}"
        )

    data_df = pd.concat(rows, ignore_index=True)
    data_df, removed_df = drop_conflicting_labels(data_df)

    manifest_df = pd.DataFrame(manifest)
    per_class = data_df['Label_Type'].value_counts().sort_index()
    print(f"  [OK] {len(data_df):,} compounds from {(manifest_df['rows'] > 0).sum()} _NEW files "
          f"({', '.join(f'{k}={v:,}' for k, v in per_class.items())})", flush=True)
    if output_dir is not None:
        import __main__
        run_name = Path(getattr(__main__, '__file__', 'run')).stem
        manifest_path = Path(output_dir) / f'{run_name}_data_manifest.csv'
        manifest_df.to_csv(manifest_path, index=False)
        print(f"  [OK] Data manifest saved: {manifest_path}", flush=True)
        if len(removed_df):
            removed_path = Path(output_dir) / f'{run_name}_removed_label_conflicts.csv'
            removed_df.to_csv(removed_path, index=False)
            print(f"  [OK] Removed rows listed in: {removed_path}", flush=True)
    return data_df, sorted(ligand_cols)


def drop_conflicting_labels(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The same compound (canonical SMILES) at the same receptor can appear in more than one class file.
    Those rows have identical features but different labels, so every such compound-receptor pair is
    removed entirely; pairs repeated with the same label are kept once.

    Returns (cleaned data_df, removed rows with Receptor, SMILES, Label_Type and the reason).
    """
    key = data_df['SMILES'].map(canonicalize_smiles).fillna(data_df['SMILES'].astype(str))
    pair = pd.DataFrame({'smiles': key, 'receptor': data_df['Receptor']})
    conflict = data_df.groupby([pair['smiles'], pair['receptor']])['Label_Type'].transform('nunique') > 1
    duplicate = ~conflict & pair.duplicated()

    removed = data_df.loc[conflict | duplicate, ['Receptor', 'SMILES', 'Label_Type']].copy()
    removed['reason'] = np.where(conflict[conflict | duplicate], 'conflicting labels', 'same-label duplicate')
    print(f"  [CLEAN] Removed {conflict.sum():,} rows where the same compound at the same receptor has "
          f"conflicting labels ({pair[conflict].drop_duplicates().shape[0]:,} pairs) and "
          f"{duplicate.sum():,} same-label duplicates; {len(data_df) - len(removed):,} rows remain", flush=True)
    return data_df[~(conflict | duplicate)].reset_index(drop=True), removed


def aggregate_receptor_features(receptor_name: str) -> Optional[Dict[str, float]]:
    """
    Extract all 31 receptor features from pocket residues CSV file.
    
    Returns:
        Dictionary with 31 features or None if not found
    """
    possible_paths = [
        WORKSPACE_ROOT / 'Josh_Receptor_Features' / receptor_name,
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
    
    # Count features (15)
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
    
    # Ratio features (14)
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
    
    # Distance/Conservation features (2)
    feats['avg_distance'] = float(df['distance_to_ligand'].mean()) if 'distance_to_ligand' in df.columns else 0.0
    feats['avg_conservation'] = float(df['conservation_score'].mean()) if 'conservation_score' in df.columns else 0.0
    
    return feats


def build_interaction_terms(merged: pd.DataFrame, receptor_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build 14 interaction terms (ligand × receptor feature products).
    
    Args:
        merged: Merged DataFrame with ligand and receptor features
        receptor_df: DataFrame with receptor features
    
    Returns:
        Tuple of (merged DataFrame with interaction terms, list of interaction column names)
    """
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
    
    return merged, interaction_cols


def should_exclude_column(col_name: str) -> bool:
    """
    Check if column should be excluded to prevent feature leakage.
    
    Args:
        col_name: Column name to check
    
    Returns:
        True if column should be excluded
    """
    exclude_base = {
        'Receptor', 'Label', 'ChEMBL_ID', 'AID', 'CID', 'Activity', 'SMILES', 'Label_Type',
        'Covalent_group', 'Warhead', 'Covalent_target_residue',  # metadata only, never as features
    }
    exclude_patterns = ['label', 'class', 'activity', 'outcome', 'type', 'assay', 'result', 'target']
    
    col_lower = str(col_name).lower()
    if col_name in exclude_base:
        return True
    for pattern in exclude_patterns:
        if pattern in col_lower:
            return True
    return False


# ============================================================================
# SMILES Canonicalization
# ============================================================================

def canonicalize_smiles(smiles_str: str) -> Optional[str]:
    """
    Canonicalize SMILES string to ensure consistent grouping.
    
    Args:
        smiles_str: Raw SMILES string
    
    Returns:
        Canonical SMILES or None if invalid
    """
    if pd.isna(smiles_str) or smiles_str == '' or str(smiles_str).strip() == '':
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smiles_str))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except (ImportError, Exception):
        return None


def canonicalize_smiles_series(merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Canonicalize SMILES for all rows in merged DataFrame.
    Filters out rows where canonicalization fails.
    
    Args:
        merged: DataFrame with 'SMILES' column
    
    Returns:
        Tuple of (filtered DataFrame, canonical SMILES Series)
    """
    print("  Canonicalizing SMILES for grouping...", flush=True)
    
    if 'SMILES' not in merged.columns:
        raise ValueError("SMILES column not found! Required for independent ligand validation.")
    
    # Filter invalid SMILES first
    smiles_check = merged['SMILES'].astype(str)
    valid_smiles_mask = (
        ~smiles_check.isna() & 
        (smiles_check != 'nan') & 
        (smiles_check != '') &
        (smiles_check.str.lower() != 'nan') &
        (smiles_check.str.strip() != '')
    )
    if not valid_smiles_mask.all():
        invalid_count = (~valid_smiles_mask).sum()
        print(f"  [WARNING] Found {invalid_count} invalid SMILES, filtering them out...", flush=True)
        merged = merged[valid_smiles_mask].reset_index(drop=True)
    
    # Canonicalize
    try:
        merged['SMILES_canonical'] = merged['SMILES'].astype(str).apply(canonicalize_smiles)
        canonical_valid_mask = merged['SMILES_canonical'].notna()
        canonical_invalid = (~canonical_valid_mask).sum()
        if canonical_invalid > 0:
            print(f"  [WARNING] {canonical_invalid} SMILES could not be canonicalized, filtering them out...", flush=True)
            merged = merged[canonical_valid_mask].reset_index(drop=True)
        else:
            print(f"  [OK] All SMILES canonicalized", flush=True)
    except ImportError:
        print("  [WARNING] RDKit not available, using raw SMILES", flush=True)
        merged['SMILES_canonical'] = merged['SMILES'].astype(str)
    except Exception as e:
        print(f"  [WARNING] SMILES canonicalization failed: {e}, using raw SMILES", flush=True)
        merged['SMILES_canonical'] = merged['SMILES'].astype(str)
    
    smiles_series = merged['SMILES_canonical']
    return merged, smiles_series


# ============================================================================
# Label Encoding
# ============================================================================

def encode_labels(merged: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Encode labels using explicit mapping (NOT LabelEncoder for consistency).
    
    Args:
        merged: DataFrame with 'Label_Type' column
    
    Returns:
        Tuple of (encoded labels array, class names array, class names list)
    """
    # Normalize labels
    merged['Label_Type_clean'] = merged['Label_Type'].astype(str).str.strip().str.lower()
    
    # Explicit mapping (consistent across all scripts)
    label_map = {'agonist': 0, 'antagonist': 1, 'inactive': 2}
    class_names = ['agonist', 'antagonist', 'inactive']
    
    y = merged['Label_Type_clean'].map(label_map).values
    
    # Check for unmapped labels
    unmapped = merged[merged['Label_Type_clean'].map(label_map).isna()]['Label_Type_clean'].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Unmapped labels found: {unmapped}. Expected: {list(label_map.keys())}")
    
    return y, np.array(class_names), class_names


# ============================================================================
# Global Split Management
# ============================================================================

def load_or_create_global_split(
    X: pd.DataFrame,
    y: np.ndarray,
    smiles_series: pd.Series,
    split_dir: Path,
    random_state: int = 42,
    test_size: float = 0.2,
    load_only: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or create the global independent ligand validation split.
    This split is shared across baseline, Optuna, and scaffold scripts.
    
    When load_only=True (e.g. for locked baseline): only load from file; if file is missing, raise.
    
    Args:
        X: Feature matrix (used for shape validation)
        y: Labels array
        smiles_series: Canonical SMILES series for grouping
        split_dir: Directory to save/load split file
        random_state: Random state for reproducibility
        test_size: Test set size (default 0.2 = 20%)
        load_only: If True, do not create split if missing; raise FileNotFoundError instead.
    
    Returns:
        Tuple of (train_indices, test_indices)
    """
    import json
    from sklearn.model_selection import GroupShuffleSplit
    
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / 'final_independent_ligand_80_20_canonical_smiles.json'
    
    # Convert smiles_series to numpy array if it's a Series
    if hasattr(smiles_series, 'values'):
        smiles_array = smiles_series.values
    else:
        smiles_array = np.array(smiles_series)
    
    if split_file.exists():
        print(f"  [INFO] Loading existing global split from {split_file}", flush=True)
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Check if split uses SMILES-based format (new) or index-based format (old)
        if 'test_smiles' in split_data or 'train_smiles' in split_data:
            # New format: split based on canonical SMILES
            test_smiles_set = set(split_data.get('test_smiles', []))
            train_smiles_set = set(split_data.get('train_smiles', []))
            
            # Find indices for current dataset based on SMILES
            current_smiles = smiles_array
            train_mask = np.array([smi in train_smiles_set for smi in current_smiles])
            test_mask = np.array([smi in test_smiles_set for smi in current_smiles])
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            # The saved split must cover exactly the current dataset; otherwise compounds are silently dropped
            missing_train = train_smiles_set - set(current_smiles)
            missing_test = test_smiles_set - set(current_smiles)
            n_unassigned = int((~(train_mask | test_mask)).sum())
            if missing_train or missing_test or n_unassigned:
                raise ValueError(
                    f"Saved split {split_file} does not match the current dataset: "
                    f"{len(missing_train)} train and {len(missing_test)} test SMILES are not in the data, and "
                    f"{n_unassigned} of {len(current_smiles)} compounds are in neither block. "
                    f"Delete or rename this file to create a new split from the current data."
                )
        else:
            # Old format: index-based (for backward compatibility)
            print(f"  [INFO] Found old index-based split format, converting...", flush=True)
            train_idx_old = np.array([int(i) for i in split_data['train']])
            test_idx_old = np.array([int(i) for i in split_data['test']])
            
            # Convert to SMILES-based and re-save
            train_smiles_set = set(smiles_array[train_idx_old])
            test_smiles_set = set(smiles_array[test_idx_old])
            
            # Find indices for current dataset
            current_smiles = smiles_array
            train_mask = np.array([smi in train_smiles_set for smi in current_smiles])
            test_mask = np.array([smi in test_smiles_set for smi in current_smiles])
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            # Re-save in new format
            split_data_new = {
                'train_smiles': sorted(list(train_smiles_set)),
                'test_smiles': sorted(list(test_smiles_set)),
                'random_state': split_data.get('random_state', random_state),
                'test_size': split_data.get('test_size', test_size),
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'format': 'smiles_based',
                'note': 'Global independent ligand validation split (80% dev block, 20% independent test). Split by canonical SMILES (no ligand overlap). SMILES-based format allows portability across different datasets. Shared across baseline, Optuna, and scaffold scripts. Independent test (20%) is HELD OUT and only used for final evaluation, never for tuning or CV.'
            }
            with open(split_file, 'w') as f:
                json.dump(split_data_new, f, indent=2)
            print(f"  [OK] Converted to SMILES-based format and re-saved", flush=True)
        
        print(f"  [OK] Loaded split: Dev block (80%) {len(train_idx):,}, Independent test (20%) {len(test_idx):,}", flush=True)
    else:
        if load_only:
            raise FileNotFoundError(
                f"Global split file not found: {split_file}. "
                "Baseline is locked to load-only; create the split once (e.g. with load_only=False) then use load_only=True."
            )
        print(f"  [INFO] Creating new global split (saving to {split_file})", flush=True)
        from sklearn.model_selection import GroupShuffleSplit
        
        # Robust split creation: Try multiple random seeds until all classes are present in test set
        # GroupShuffleSplit is not stratified, so this ensures robustness
        n_classes = len(np.unique(y))
        max_attempts = 50
        chosen_seed = random_state
        train_idx = None
        test_idx = None
        
        for attempt in range(max_attempts):
            try_seed = random_state + attempt
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=try_seed)
            train_idx_temp, test_idx_temp = next(gss.split(X, y, groups=smiles_array))
            
            # Check if all classes are present in test set
            test_classes = set(y[test_idx_temp])
            if len(test_classes) == n_classes:
                train_idx = train_idx_temp
                test_idx = test_idx_temp
                chosen_seed = try_seed
                print(f"  [INFO] Successfully created split with all classes in test set (seed={chosen_seed}, attempt={attempt+1})", flush=True)
                break
            elif attempt < 5:  # Print warning for first few attempts
                print(f"  [INFO] Attempt {attempt+1}: seed={try_seed} missing classes in test set: {set(range(n_classes)) - test_classes}", flush=True)
        
        if train_idx is None:
            # Fallback: use the last attempt (may be missing classes)
            print(f"  [WARNING] After {max_attempts} attempts, no split found with all classes in test set. Using last attempt.", flush=True)
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + max_attempts - 1)
            train_idx, test_idx = next(gss.split(X, y, groups=smiles_array))
            chosen_seed = random_state + max_attempts - 1
            test_classes = set(y[test_idx])
            missing_classes = set(range(n_classes)) - test_classes
            if missing_classes:
                print(f"  [ERROR] Test set missing classes: {missing_classes}. This will cause issues with AUC and per-class metrics.", flush=True)
                raise ValueError(f"Could not create split with all classes in test set after {max_attempts} attempts. Missing classes: {missing_classes}")
        
        # Get unique SMILES for each split
        train_smiles_set = set(smiles_array[train_idx])
        test_smiles_set = set(smiles_array[test_idx])
        
        # Save split based on SMILES (not indices) for portability
        split_data = {
            'train_smiles': sorted(list(train_smiles_set)),
            'test_smiles': sorted(list(test_smiles_set)),
            'random_state': chosen_seed,  # Save the actual seed used (may differ from input if needed)
            'original_random_state': random_state,  # Save the requested seed
            'test_size': test_size,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'format': 'smiles_based',
            'note': 'Global independent ligand validation split (80% dev block, 20% independent test). Split by canonical SMILES (no ligand overlap). SMILES-based format allows portability across different datasets. Shared across baseline, Optuna, and scaffold scripts. Independent test (20%) is HELD OUT and only used for final evaluation, never for tuning or CV. Split creation tries multiple seeds to ensure all classes are present in test set.'
        }
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"  [OK] Created and saved split: Dev block (80%) {len(train_idx):,}, Independent test (20%) {len(test_idx):,}", flush=True)
        print(f"  [INFO] Split saved as SMILES-based (portable across datasets)", flush=True)
        if chosen_seed != random_state:
            print(f"  [INFO] Note: Used seed={chosen_seed} (requested {random_state}) to ensure all classes in test set", flush=True)
    
    # Verify indices are valid
    assert len(set(train_idx) & set(test_idx)) == 0, "Train and test indices overlap!"
    
    # Verify no SMILES overlap
    train_smiles_in_split = set(smiles_array[train_idx])
    test_smiles_in_split = set(smiles_array[test_idx])
    overlap = train_smiles_in_split & test_smiles_in_split
    if overlap:
        print(f"  [ERROR] Found {len(overlap)} overlapping SMILES in train/test split!", flush=True)
        raise ValueError("Train and test sets have SMILES overlap! This should not happen.")
    
    return train_idx, test_idx



