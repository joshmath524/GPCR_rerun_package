#!/usr/bin/env python3
"""Tiny stacking prototype: a few receptors, a few hundred rows, small trees.

Checks that the held-out test F1 and AUC are real numbers. Not a result to report.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_utilities import (
    aggregate_receptor_features,
    build_interaction_terms,
    canonicalize_smiles_series,
    encode_labels,
    load_receptor_ligand_data,
)

RECEPTORS = ['5-HT2A', 'D2']
MAX_ROWS = 300
N_ESTIMATORS = 20
SEED = 42


def main():
    print(f"Prototype receptors: {RECEPTORS}", flush=True)
    receptor_feats = {}
    for rec in RECEPTORS:
        feats = aggregate_receptor_features(rec)
        if not feats:
            raise RuntimeError(f"No pocket features for {rec}")
        receptor_feats[rec] = feats
    receptor_df = pd.DataFrame(receptor_feats).T
    receptor_df.index.name = 'Receptor'

    data_df, ligand_cols = load_receptor_ligand_data(RECEPTORS, output_dir=None)
    print(f"Loaded {len(data_df):,} rows before subsampling", flush=True)

    merged = data_df.merge(receptor_df.reset_index(), on='Receptor', how='left')
    merged, interaction_cols = build_interaction_terms(merged, receptor_df)
    merged, _ = canonicalize_smiles_series(merged)
    y, _, class_names = encode_labels(merged)

    feature_cols = [c for c in ligand_cols + list(receptor_df.columns) + interaction_cols if c in merged.columns]
    X = merged[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], 0).fillna(0)
    X = X.select_dtypes(include=[np.number])

    # Keep all three classes, then cap the size.
    keep_idx, _ = train_test_split(
        np.arange(len(X)), train_size=min(MAX_ROWS, len(X) - 3), random_state=SEED, stratify=y
    )
    X = X.iloc[keep_idx].reset_index(drop=True)
    y = y[keep_idx]
    print(f"Using {len(X)} rows, {X.shape[1]} features, classes {class_names}", flush=True)
    print(pd.Series(y).value_counts().sort_index().to_string(), flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    stack = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=8, n_jobs=-1, random_state=SEED)),
            ('xgb', xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.1,
                objective='multi:softprob', num_class=3, tree_method='hist',
                n_jobs=-1, random_state=SEED, verbosity=0,
            )),
            ('lgbm', lgb.LGBMClassifier(
                n_estimators=N_ESTIMATORS, num_leaves=15, learning_rate=0.1,
                objective='multiclass', num_class=3, subsample_freq=1,
                n_jobs=-1, random_state=SEED, verbose=-1,
            )),
        ],
        final_estimator=LogisticRegression(max_iter=500, random_state=SEED),
        cv=2,
        n_jobs=1,
    )
    print("Fitting small stack...", flush=True)
    stack.fit(X_train, y_train)
    proba = stack.predict_proba(X_test)
    pred = stack.predict(X_test)
    f1 = f1_score(y_test, pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_test, proba, average='macro', multi_class='ovr')
    print(f"Test rows: {len(y_test)}", flush=True)
    print(f"Test: F1={f1:.4f}, AUC={auc:.4f}", flush=True)
    if not np.isfinite(f1) or not np.isfinite(auc):
        raise SystemExit('Prototype failed: F1 or AUC is not a real number')
    print("Prototype OK", flush=True)


if __name__ == '__main__':
    main()
