#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared setup for the Optuna tuning scripts (pan_gpcr_rf_optuna.py, pan_gpcr_xgb_optuna.py,
pan_gpcr_lightgbm_optuna.py).

- Data: validated _NEW files only (shared_utilities.load_receptor_ligand_data), same features as the
  evaluation scripts (ligand descriptors + 31 receptor features + 14 interaction terms).
- Split: the global 80/20 split in MLcodes/pan_gpcr_results/splits (shared with the baseline scripts).
  Only the 80% dev block is used for tuning; the 20% test block is never touched.
- Resume: every trial is stored in a SQLite study (pan_gpcr_results/optuna/<model>_optuna.db);
  rerunning a script continues where it stopped. A study refuses to resume on different data.
- Outputs (updated after every trial): <model>_optuna_trials.csv and <model>_best_params.json.
- The feature matrix is cached after the first build and rebuilt automatically when any data file,
  receptor feature file or this code changes.
"""

import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ML_code'))
from shared_utilities import (  # noqa: E402
    WORKSPACE_ROOT,
    LIGAND_FILE_KINDS,
    aggregate_receptor_features,
    build_interaction_terms,
    canonicalize_smiles_series,
    encode_labels,
    load_or_create_global_split,
    load_receptor_ligand_data,
)

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

RESULTS_DIR = WORKSPACE_ROOT / 'MLcodes' / 'pan_gpcr_results'
SPLIT_DIR = RESULTS_DIR / 'splits'
OPTUNA_DIR = RESULTS_DIR / 'optuna'
CACHE_FILE = OPTUNA_DIR / 'cache' / 'tuning_dataset.pkl'

SEED = 42
EARLY_STOPPING_FRACTION = 0.1


def _rel(path):
    try:
        return path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return path


def sanitize_feature_name(name):
    """Same renaming as the evaluation scripts (LightGBM rejects JSON special characters)."""
    sanitized = re.sub(r'[^\w\-.]', '_', str(name))
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized if sanitized else 'feature_' + str(hash(name) % 10000)


def _source_files():
    files = [Path(__file__).resolve(), WORKSPACE_ROOT / 'ML_code' / 'shared_utilities.py']
    for rec in RECEPTORS:
        for _, kind in LIGAND_FILE_KINDS:
            files += [p for p in (WORKSPACE_ROOT / rec / f'{rec}_{kind}_NEW.csv',
                                  WORKSPACE_ROOT / rec / f'{rec}_{kind}_NEW.xlsx') if p.exists()]
        files += sorted((WORKSPACE_ROOT / 'Josh_Receptor_Features' / rec).glob('*_pocket_residues_with_conservation.csv'))
    return files


def _cache_key():
    h = hashlib.sha256()
    for p in _source_files():
        st = p.stat()
        h.update(f"{p.relative_to(WORKSPACE_ROOT).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def build_dataset():
    """Full feature matrix (all 40k compounds) built exactly like the evaluation scripts."""
    key = _cache_key()
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as f:
            cached = pickle.load(f)
        if cached['key'] == key:
            print(f"[DATA] Loaded cached feature matrix {cached['X'].shape} ({CACHE_FILE.name}); "
                  f"source files unchanged", flush=True)
            return cached['X'], cached['y'], cached['smiles'], cached['class_names']
        print("[DATA] Data files changed since the cache was built; rebuilding", flush=True)

    print("[DATA] Aggregating receptor features...", flush=True)
    receptor_feats = {rec: f for rec in RECEPTORS if (f := aggregate_receptor_features(rec))}
    receptor_df = pd.DataFrame(receptor_feats).T.fillna(0)
    receptor_df.index.name = 'Receptor'
    print(f"  [OK] {len(receptor_df)} receptors with features", flush=True)

    print("[DATA] Loading ligand data (_NEW only, validated)...", flush=True)
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    data_df, ligand_cols = load_receptor_ligand_data(list(receptor_df.index), OPTUNA_DIR)

    merged = data_df.merge(receptor_df.reset_index(), on='Receptor', how='left')
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
    merged, interaction_cols = build_interaction_terms(merged, receptor_df)
    merged, smiles = canonicalize_smiles_series(merged)

    feature_cols = [c for c in ligand_cols + list(receptor_df.columns) + interaction_cols if c in merged.columns]
    X = (merged[feature_cols].apply(pd.to_numeric, errors='coerce')
         .replace([np.inf, -np.inf], 0).fillna(0).select_dtypes(include=[np.number]))
    X.columns = [sanitize_feature_name(c) for c in X.columns]
    y, _, class_names = encode_labels(merged)
    smiles = smiles.reset_index(drop=True)
    print(f"  [OK] Feature matrix {X.shape}: {len(ligand_cols)} ligand, {len(receptor_df.columns)} receptor, "
          f"{len(interaction_cols)} interaction columns", flush=True)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump({'key': key, 'X': X, 'y': y, 'smiles': smiles, 'class_names': class_names}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_FILE)
    print(f"  [OK] Cached feature matrix to {_rel(CACHE_FILE)}", flush=True)
    return X, y, smiles, class_names


def get_dev_block():
    """80% dev block of the global split, plus a fingerprint identifying this exact data."""
    X, y, smiles, class_names = build_dataset()
    train_idx, test_idx = load_or_create_global_split(
        X, y, smiles, SPLIT_DIR, random_state=SEED, test_size=0.2, load_only=False
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    X_dev = X.iloc[train_idx].reset_index(drop=True)
    y_dev = np.asarray(y)[train_idx]
    smiles_dev = smiles.iloc[train_idx].reset_index(drop=True)

    fingerprint = hashlib.sha256(
        train_idx.tobytes() + '|'.join(X.columns).encode() + np.asarray(y, dtype=np.int64).tobytes()
    ).hexdigest()

    print(f"\n[SPLIT] Tuning on the 80% dev block only: {len(X_dev):,} compounds "
          f"({smiles_dev.nunique():,} unique canonical SMILES), {X_dev.shape[1]} features", flush=True)
    print(f"[SPLIT] 20% test block held out and never used: {len(test_idx):,} compounds", flush=True)
    for i, name in enumerate(class_names):
        print(f"        {name}: {(y_dev == i).sum():,}", flush=True)
    return X_dev, y_dev, smiles_dev, list(class_names), fingerprint


def make_folds(y_dev, smiles_dev, n_splits):
    """Ligand-grouped CV folds; each training part also gets a grouped 10% early-stopping subset."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    groups = smiles_dev.to_numpy()
    folds = []
    for train_idx, val_idx in sgkf.split(np.zeros(len(y_dev)), y_dev, groups=groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=EARLY_STOPPING_FRACTION, random_state=SEED)
        fit_pos, es_pos = next(gss.split(train_idx, groups=groups[train_idx]))
        folds.append({'train': train_idx, 'val': val_idx,
                      'fit': train_idx[fit_pos], 'early_stop': train_idx[es_pos]})
    return folds


def _atomic_write(path, write_fn):
    tmp = path.with_suffix(path.suffix + '.tmp')
    write_fn(tmp)
    try:
        tmp.replace(path)
    except PermissionError:
        print(f"  [WARNING] Could not update {path.name} (open in another program?); "
              f"latest copy left in {tmp.name}", flush=True)


def export_results(study, model_name, fingerprint, finalize_params):
    trials_csv = OPTUNA_DIR / f'{model_name}_optuna_trials.csv'
    _atomic_write(trials_csv, lambda p: study.trials_dataframe().to_csv(p, index=False))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best = study.best_trial
    summary = {
        'model': model_name,
        'best_trial': best.number,
        'cv_macro_f1_mean': best.value,
        'cv_macro_f1_std': best.user_attrs.get('std_macro_F1'),
        'n_trials_completed': len(completed),
        'data_fingerprint': fingerprint,
        'params': finalize_params(best),
    }
    best_json = OPTUNA_DIR / f'{model_name}_best_params.json'
    _atomic_write(best_json, lambda p: p.write_text(json.dumps(summary, indent=2)))


def run_study(model_name, objective, fingerprint, n_trials, finalize_params):
    """Create or resume the study and run until n_trials trials have completed."""
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db_path = OPTUNA_DIR / f'{model_name}_optuna.db'
    study = optuna.create_study(
        study_name=f'{model_name}_multiclass_macro_f1',
        storage=f'sqlite:///{db_path.as_posix()}',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    saved = study.user_attrs.get('data_fingerprint')
    if saved is None:
        study.set_user_attr('data_fingerprint', fingerprint)
    elif saved != fingerprint:
        raise ValueError(
            f"{db_path.name} was created with different data, features or split. "
            f"Rename or delete it to start a new study."
        )

    # A resumed run must not replay the random draws of the first run, so the sampler seed
    # advances with the number of trials already in the study (still reproducible)
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials))

    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(n_trials - n_done, 0)
    print(f"\n[OPTUNA] {model_name}: {n_done} trials already completed, {remaining} to run "
          f"(target {n_trials}). Study: {_rel(db_path)}", flush=True)

    def callback(study, trial):
        export_results(study, model_name, fingerprint, finalize_params)

    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=[callback], gc_after_trial=True)
    export_results(study, model_name, fingerprint, finalize_params)

    best = study.best_trial
    print(f"\n[OPTUNA] Best trial {best.number}: macro F1 = {best.value:.4f} "
          f"Â± {best.user_attrs.get('std_macro_F1', float('nan')):.4f}")
    for k, v in finalize_params(best).items():
        print(f"    {k}: {v}")
    print(f"[OPTUNA] Trials:      {_rel(OPTUNA_DIR / f'{model_name}_optuna_trials.csv')}")
    print(f"[OPTUNA] Best params: {_rel(OPTUNA_DIR / f'{model_name}_best_params.json')}")
    return study
