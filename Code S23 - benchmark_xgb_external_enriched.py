#!/usr/bin/env python3
"""
Benchmark XGBoost Model on GPCRA17_enriched.csv
- Same structure as benchmark_rf_external_enriched.py
- GPCRA17 split: ILV (10% unique ligands) + 80/10/10 train/test/internal_val.
- 10 seeds; report mean ± std on test and ILV.
- Best-trial XGB params (Optuna trial 111); n_estimators=1843.
- Receptors with no pocket file are kept. Their receptor and interaction columns are 0, so those rows use ligand descriptors only.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent / 'ML_code'))
from shared_utilities import (
    aggregate_receptor_features,
    build_interaction_terms,
    should_exclude_column,
    canonicalize_smiles_series,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEEDS = list(range(42, 52))  # 10 seeds
N_CLASSES = 3
BENCHMARK_DIR = Path('outputs/benchmarking')
FEATURE_NAMES_PATH = BENCHMARK_DIR / 'feature_names.json'
SPLIT_DIR = BENCHMARK_DIR / 'gpcra17_splits'

RECEPTOR_MAPPING = {
    'D1R': 'D1', 'D2R': 'D2', 'D3R': 'D3', 'D4R': 'D4', 'D5R': 'D5',
    '5HT2A': '5-HT2A', '5HT2B': '5-HT2B', '5HT2C': '5-HT2C', '5HT6R': '5-HT6',
    'ADRA1A': 'alpha1A', 'ADRA1B': 'alpha1B',
    'ADRA2A': 'alpha2A', 'ADRA2B': 'alpha2B', 'ADRA2C': 'alpha2C',
    'ADRB1': 'beta1', 'ADRB2': 'beta2', 'ADRB3': 'beta3',
    'TA1R': 'TA1',
}

CLASS_ORDER = ['Agonist', 'Antagonist', 'Modulator']
NUMERIC_CLASS_MAP = {0: 0, 1: 1, 2: 2}

# Best-trial XGBoost hyperparameters (Optuna trial 111). internal_val is NOT used for training.
XGB_PARAMS = {
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
    'tree_method': 'hist',
    'n_jobs': -1,
}


def normalize_receptor(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    return RECEPTOR_MAPPING.get(s, s)


def make_gpcra17_split(df, seed, y_enc, ligand_col='SMILES_canonical'):
    col = ligand_col if ligand_col in df.columns else 'SMILES'
    if col not in df.columns:
        raise ValueError("Ligand column '{}' not in df".format(ligand_col))
    rng = np.random.RandomState(seed)
    ligands = df[col].astype(str).dropna()
    ligands = ligands[ligands.str.strip() != '']
    unique_ligands = ligands.unique().tolist()
    n_ligands = len(unique_ligands)
    n_ilv = max(1, int(round(0.10 * n_ligands)))
    ilv_ligands = set(rng.choice(unique_ligands, size=n_ilv, replace=False))
    ilv_mask = df[col].astype(str).isin(ilv_ligands)
    ilv_indices = np.where(ilv_mask)[0].tolist()
    rest_mask = ~ilv_mask
    rest_indices = np.where(rest_mask)[0]
    y_rest_enc = y_enc[rest_indices]
    idx_train, idx_rest = train_test_split(
        np.arange(len(rest_indices)), test_size=0.2, stratify=y_rest_enc, random_state=seed
    )
    y_rest_2 = y_rest_enc[idx_rest]
    idx_test, idx_val = train_test_split(idx_rest, test_size=0.5, stratify=y_rest_2, random_state=seed)
    train_indices = rest_indices[idx_train].tolist()
    test_indices = rest_indices[idx_test].tolist()
    internal_val_indices = rest_indices[idx_val].tolist()
    return {
        'ilv_ligands': list(ilv_ligands),
        'ilv_indices': ilv_indices,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'internal_val_indices': internal_val_indices,
    }


def encode_classes(y_labels):
    out = []
    for c in y_labels:
        if isinstance(c, (int, np.integer)) or (isinstance(c, (float, np.floating)) and not np.isnan(c) and c == int(c)):
            val = int(c)
            if val in NUMERIC_CLASS_MAP:
                out.append(NUMERIC_CLASS_MAP[val])
                continue
        s = str(c).strip()
        if s in CLASS_ORDER:
            out.append(CLASS_ORDER.index(s))
        else:
            low = s.lower()
            if low == 'agonist':
                out.append(0)
            elif low == 'antagonist':
                out.append(1)
            elif low in ('modulator', 'inactive'):
                out.append(2)
            else:
                raise ValueError("Unknown class '{}'; expected one of {} or numeric 0/1/2".format(c, CLASS_ORDER))
    return np.array(out, dtype=np.int64)


def compute_macro_specificity(y_true_enc, y_pred_enc, n_classes):
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(n_classes))
    specs = []
    for i in range(n_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    return np.mean(specs) if specs else 0.0


def compute_metrics(y_true_enc, y_pred_enc, y_proba, class_names):
    prec = precision_score(y_true_enc, y_pred_enc, average='macro', zero_division=0)
    rec = recall_score(y_true_enc, y_pred_enc, average='macro', zero_division=0)
    f1 = f1_score(y_true_enc, y_pred_enc, average='macro', zero_division=0)
    spec = compute_macro_specificity(y_true_enc, y_pred_enc, len(class_names))
    auc = np.nan
    if len(np.unique(y_true_enc)) >= len(class_names):
        try:
            auc = roc_auc_score(y_true_enc, y_proba, average='macro', multi_class='ovr')
        except Exception:
            auc = np.nan
    return {'precision_macro': prec, 'recall_macro': rec, 'f1_macro': f1, 'specificity_macro': spec, 'auc_macro': auc}


# ---- 1. Load ----
print("="*80)
print("BENCHMARKING XGBOOST ON GPCRA17_enriched.csv (GPCRA17 split, 10 seeds)")
print("="*80)

print("\n[1/6] Loading GPCRA17_enriched.csv...")
df = pd.read_csv('GPCRA17_enriched.csv', low_memory=False)
if 'Smiles' in df.columns and 'SMILES' not in df.columns:
    df['SMILES'] = df['Smiles']
if 'Action' in df.columns:
    df['class'] = df['Action']

print("  df['class'].dtype: {}".format(df['class'].dtype))
print("  df['class'].value_counts().head(10):")
print(df['class'].value_counts().head(10).to_string())
print("  Sample rows (verify numeric class mapping against dataset docs):")
print(df[['Receptor', 'SMILES', 'class']].head(20).to_string())
# NUMERIC_CLASS_MAP: dataset value -> our index (0=Agonist, 1=Antagonist, 2=Modulator)
# Verify from printed sample: if e.g. 0=Antagonist, 1=Agonist, 2=Modulator use {0:1, 1:0, 2:2}

df['Receptor'] = df['Receptor'].apply(normalize_receptor)
df, _ = canonicalize_smiles_series(df)
print("  [OK] Loaded {} compounds (after canonicalization)".format(len(df)))

meta_cols = {'Receptor', 'receptor_id', 'Ligand', 'Action', 'Smiles', 'SMILES', 'SMILES_canonical', 'class'}
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
ligand_cols = [c for c in numeric_cols if c not in meta_cols and not should_exclude_column(c)]

# ---- 2. Receptor features ----
print("\n[2/6] Adding receptor features and interaction terms...")
receptor_feats = {}
for rec in df['Receptor'].unique():
    feats = aggregate_receptor_features(rec)
    if feats:
        receptor_feats[rec] = feats

receptor_df = pd.DataFrame(receptor_feats).T if receptor_feats else pd.DataFrame()
if len(receptor_df) > 0:
    receptor_df.index.name = 'Receptor'
    receptor_df_reset = receptor_df.reset_index()
    merged = df.merge(receptor_df_reset, on='Receptor', how='left')
    receptor_cols = list(receptor_df.columns)
    missing_mask = merged[receptor_cols].isna().all(axis=1)
    # GPCRA17 names these four ligand columns differently from the interaction pairs.
    for dest, src in (
        ('HBD', 'nHBDon'),
        ('HBA', 'nHBAcc'),
        ('AromaticRings', 'NumAromaticRings'),
        ('RotatableBonds', 'NumRotatableBonds'),
    ):
        if dest not in merged.columns and src in merged.columns:
            merged[dest] = merged[src]
    merged, interaction_cols = build_interaction_terms(merged, receptor_df)
    print("  Interaction terms built: {}".format(len(interaction_cols)))
    if missing_mask.any():
        merged.loc[missing_mask, receptor_cols + interaction_cols] = 0
        kept = merged.loc[missing_mask, 'Receptor'].value_counts()
        print("  No pocket file; kept on ligand descriptors only:")
        for rec, count in kept.items():
            print("    {}: {}".format(rec, count))
        print("  Ligand-descriptor-only rows: {}".format(int(missing_mask.sum())))
else:
    merged = df.copy()
    interaction_cols = []

feature_cols = ligand_cols + list(receptor_df.columns) + interaction_cols if len(receptor_df) > 0 else ligand_cols + interaction_cols
feature_cols = [c for c in feature_cols if c in merged.columns]

print("  Rows per receptor:")
for rec, count in merged['Receptor'].value_counts().sort_index().items():
    print("    {}: {}".format(rec, count))

# ---- 3. Feature schema ----
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
if FEATURE_NAMES_PATH.exists():
    feature_names = json.load(open(FEATURE_NAMES_PATH))
    missing = [f for f in feature_names if f not in merged.columns]
    if missing:
        raise ValueError("feature_names.json contains columns not in merged: {}".format(missing[:20]))
    print("\n[3/6] Loaded feature schema from {} ({} features)".format(FEATURE_NAMES_PATH, len(feature_names)))
else:
    feature_names = [c for c in feature_cols if c in merged.columns]
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("\n[3/6] Saved feature schema to {} ({} features)".format(FEATURE_NAMES_PATH, len(feature_names)))

X = merged[feature_names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float64)
y_labels = merged['class'].values
y_enc = encode_classes(y_labels)
class_names = CLASS_ORDER

# ---- 4. Run 10 seeds ----
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
all_test_metrics, all_ilv_metrics, all_internal_val_metrics = [], [], []

for seed in SEEDS:
    split = make_gpcra17_split(merged, seed, y_enc, ligand_col='SMILES_canonical')
    with open(SPLIT_DIR / 'split_seed{}.json'.format(seed), 'w') as f:
        json.dump(split, f, indent=2)

    tr_idx = split['train_indices']
    te_idx = split['test_indices']
    ilv_idx = split['ilv_indices']
    val_idx = split['internal_val_indices']

    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y_enc[tr_idx], y_enc[te_idx]
    X_ilv = X.iloc[ilv_idx]
    y_ilv = y_enc[ilv_idx]
    X_val = X.iloc[val_idx]
    y_val = y_enc[val_idx]

    params = {k: v for k, v in XGB_PARAMS.items()}
    params['random_state'] = seed
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    for name, X_eval, y_eval, idx in [
        ('test', X_test, y_test, te_idx),
        ('ilv', X_ilv, y_ilv, ilv_idx),
        ('internal_val', X_val, y_val, val_idx),
    ]:
        pred = model.predict(X_eval)
        proba = model.predict_proba(X_eval)
        m = compute_metrics(y_eval, pred, proba, class_names)
        if name == 'test':
            all_test_metrics.append(m)
        elif name == 'ilv':
            all_ilv_metrics.append(m)
        else:
            all_internal_val_metrics.append(m)

        if name in ('test', 'ilv'):
            sub = merged.iloc[idx].copy().reset_index(drop=True)
            sub['y_true'] = [class_names[i] for i in y_eval]
            sub['y_pred'] = [class_names[i] for i in pred]
            for k in range(proba.shape[1]):
                sub['proba_class{}'.format(k)] = proba[:, k]
            smi_col = 'SMILES_canonical' if 'SMILES_canonical' in sub.columns else 'SMILES'
            out_cols = [smi_col, 'Receptor', 'y_true', 'y_pred'] + [c for c in sub.columns if c.startswith('proba_')]
            sub[[c for c in out_cols if c in sub.columns]].to_csv(
                BENCHMARK_DIR / 'predictions_xgb_seed{}_{}.csv'.format(seed, name), index=False
            )

    with open(BENCHMARK_DIR / 'metrics_xgb_seed{}.json'.format(seed), 'w') as f:
        json.dump({
            'test': all_test_metrics[-1],
            'ilv': all_ilv_metrics[-1],
            'internal_val': all_internal_val_metrics[-1],
        }, f, indent=2)

# ---- 5. Summary ----
def mean_std(lst, key):
    vals = [m[key] for m in lst if m.get(key) is not None and not (isinstance(m[key], float) and np.isnan(m[key]))]
    if not vals:
        return np.nan, np.nan
    return np.mean(vals), np.std(vals)

summary = {'dataset': 'GPCRA17_enriched', 'model': 'XGBoost', 'n_seeds': len(SEEDS), 'test': {}, 'ilv': {}, 'internal_val': {}}
for metric in ['precision_macro', 'recall_macro', 'f1_macro', 'specificity_macro', 'auc_macro']:
    m, s = mean_std(all_test_metrics, metric)
    summary['test'][metric] = {'mean': float(m), 'std': float(s) if not np.isnan(s) else None}
    m, s = mean_std(all_ilv_metrics, metric)
    summary['ilv'][metric] = {'mean': float(m), 'std': float(s) if not np.isnan(s) else None}
    m, s = mean_std(all_internal_val_metrics, metric)
    summary['internal_val'][metric] = {'mean': float(m), 'std': float(s) if not np.isnan(s) else None}

with open(BENCHMARK_DIR / 'xgb_gpcra17_enriched_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n  Saved: {}".format(BENCHMARK_DIR / 'xgb_gpcra17_enriched_results.json'))

rows = []
for subset in ('test', 'ilv', 'internal_val'):
    for metric in ['precision_macro', 'recall_macro', 'f1_macro', 'specificity_macro', 'auc_macro']:
        d = summary[subset][metric]
        rows.append({'subset': subset, 'metric': metric, 'mean': d['mean'], 'std': d['std']})
pd.DataFrame(rows).to_csv(BENCHMARK_DIR / 'xgb_gpcra17_enriched_summary.csv', index=False)
print("  Saved: {}".format(BENCHMARK_DIR / 'xgb_gpcra17_enriched_summary.csv'))

# ---- 6. Print ----
print("\n" + "="*80)
print("RESULTS (mean ± std over {} seeds)".format(len(SEEDS)))
print("="*80)
for subset, label in [('test', 'Test (10%)'), ('ilv', 'ILV (10% ligands)'), ('internal_val', 'Internal val (10%)')]:
    print("\n{}:".format(label))
    for metric in ['precision_macro', 'recall_macro', 'f1_macro', 'specificity_macro', 'auc_macro']:
        m = summary[subset][metric]['mean']
        s = summary[subset][metric]['std']
        if s is not None:
            print("  {}: {:.4f} ± {:.4f}".format(metric, m, s))
        else:
            print("  {}: {:.4f}".format(metric, m))
print("\n" + "="*80)
