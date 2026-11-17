#!/usr/bin/env python3
"""
Optuna helpers for curated-warhead feature matrix.

Builds X/y from the warhead scaffold parquet cache + pocket features
(including num_cys / has_pocket_cys and warhead×pocket interactions).
Tunes on the 80% ligand-independent dev block only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ML_code"))
sys.path.insert(0, str(ROOT / "MLcodes" / "optuna_tuning"))

from optuna_common import (  # noqa: E402
    EARLY_STOPPING_FRACTION,
    RECEPTORS,
    SEED,
    make_folds,
)
from shared_utilities import aggregate_receptor_features  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
OPTUNA_DIR = Path(__file__).resolve().parent
SEED = 42


def sanitize_feature_name(name: str) -> str:
    sanitized = re.sub(r"[^\w\-.]", "_", str(name))
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized if sanitized else "feature_" + str(hash(name) % 10000)


def build_warhead_dataset():
    pq = CACHE_DIR / "ligand_features.parquet"
    cols_path = CACHE_DIR / "feature_columns.json"
    if not pq.exists():
        raise FileNotFoundError(pq)

    data_df = pd.read_parquet(pq)
    ligand_cols = json.loads(cols_path.read_text())
    print(f"[DATA] Cache {data_df.shape}; ligand cols={len(ligand_cols)}", flush=True)

    # Extend receptor aggregation with Cys counts (same as curated scaffold scripts)
    print("[DATA] Aggregating receptor features (+ pocket Cys)...", flush=True)
    receptor_feats = {}
    for rec in RECEPTORS:
        feats = aggregate_receptor_features(rec)
        if not feats:
            continue
        # add Cys if not present in shared util
        if "num_cys" not in feats:
            folder = ROOT / "Josh_Receptor_Features" / rec
            files = list(folder.glob("*_pocket_residues_with_conservation.csv"))
            if files:
                rdf = pd.read_csv(files[0])
                if "residue_name" in rdf.columns:
                    feats["num_cys"] = int((rdf["residue_name"].astype(str).str.upper() == "CYS").sum())
                else:
                    feats["num_cys"] = 0
            else:
                feats["num_cys"] = 0
        feats["has_pocket_cys"] = 1.0 if feats.get("num_cys", 0) > 0 else 0.0
        receptor_feats[rec] = feats

    receptor_df = pd.DataFrame(receptor_feats).T.fillna(0)
    receptor_df.index.name = "Receptor"
    print(f"  [OK] {len(receptor_df)} receptors", flush=True)

    merged = data_df.merge(receptor_df.reset_index(), on="Receptor", how="left")
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)

    interaction_pairs = [
        ("LogP", "aromatic_ratio"), ("LogP", "num_basic"), ("TPSA", "acidic_ratio"),
        ("HBD", "num_hydroxyl"), ("HBA", "num_basic"), ("HBA", "num_amide"),
        ("AromaticRings", "aromatic_ratio"), ("AromaticRings", "num_aromatic"),
        ("FormalCharge", "num_charge_positive"), ("FormalCharge", "num_charge_negative"),
        ("MolWt", "num_residues"), ("RotatableBonds", "num_residues"),
        ("LogP", "avg_conservation"), ("TPSA", "avg_conservation"),
    ]
    interaction_cols = []
    for lig, rec in interaction_pairs:
        if lig in merged.columns and rec in merged.columns:
            name = f"INT_{lig}_X_{rec}"
            merged[name] = merged[lig] * merged[rec]
            interaction_cols.append(name)

    if "has_covalent_warhead" in merged.columns:
        for rec_col, tag in [("has_pocket_cys", "pocket_cys"), ("num_cys", "num_cys"), ("num_sulfur", "num_sulfur")]:
            if rec_col in merged.columns:
                name = f"INT_has_covalent_warhead_X_{tag}"
                merged[name] = merged["has_covalent_warhead"] * merged[rec_col]
                interaction_cols.append(name)

    feature_cols = [c for c in ligand_cols + list(receptor_df.columns) + interaction_cols if c in merged.columns]
    X = (
        merged[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .select_dtypes(include=[np.number])
    )
    X.columns = [sanitize_feature_name(c) for c in X.columns]

    enc = LabelEncoder()
    y = enc.fit_transform(merged["Label_Type"])
    # force agonist=0, antagonist=1, inactive=2 if possible
    class_names = list(enc.classes_)
    smiles = merged["SMILES"].astype(str).reset_index(drop=True)

    wh_n = sum(1 for c in X.columns if c.startswith("WHB_") or c == "has_covalent_warhead")
    print(
        f"  [OK] Feature matrix {X.shape}; warhead-related cols≈{wh_n}; "
        f"interactions={len(interaction_cols)}",
        flush=True,
    )
    return X, y, smiles, class_names


def get_dev_block():
    """Ligand-independent 80% dev block (test never used for tuning)."""
    X, y, smiles, class_names = build_warhead_dataset()

    # Deterministic ligand-level 80/20 split
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=smiles.to_numpy()))
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    # Persist split for reproducibility
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    split_path = OPTUNA_DIR / "warhead_optuna_ligand_split.npz"
    np.savez_compressed(split_path, train_idx=train_idx, test_idx=test_idx)

    X_dev = X.iloc[train_idx].reset_index(drop=True)
    y_dev = np.asarray(y)[train_idx]
    smiles_dev = smiles.iloc[train_idx].reset_index(drop=True)

    fingerprint = hashlib.sha256(
        train_idx.tobytes() + "|".join(X.columns).encode() + np.asarray(y, dtype=np.int64).tobytes()
        + b"|warhead_curated_v1"
    ).hexdigest()

    print(
        f"\n[SPLIT] Tuning on 80% ligand-indep. dev: {len(X_dev):,} rows "
        f"({smiles_dev.nunique():,} SMILES), {X_dev.shape[1]} features",
        flush=True,
    )
    print(f"[SPLIT] Held-out test block: {len(test_idx):,} (never used in Optuna)", flush=True)
    return X_dev, y_dev, smiles_dev, list(class_names), fingerprint


def _atomic_write(path, write_fn):
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    tmp.replace(path)


def export_results(study, model_name, fingerprint, finalize_params):
    trials_csv = OPTUNA_DIR / f"{model_name}_optuna_trials.csv"
    _atomic_write(trials_csv, lambda p: study.trials_dataframe().to_csv(p, index=False))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best = study.best_trial
    summary = {
        "model": model_name,
        "best_trial": best.number,
        "cv_macro_f1_mean": best.value,
        "cv_macro_f1_std": best.user_attrs.get("std_macro_F1"),
        "n_trials_completed": len(completed),
        "data_fingerprint": fingerprint,
        "feature_set": "curated_warhead_binary_plus_pocket_cys_interactions",
        "params": finalize_params(best),
    }
    best_json = OPTUNA_DIR / f"{model_name}_best_params.json"
    _atomic_write(best_json, lambda p: p.write_text(json.dumps(summary, indent=2)))


def run_study(model_name, objective, fingerprint, n_trials, finalize_params):
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db_path = OPTUNA_DIR / f"{model_name}_optuna.db"
    study = optuna.create_study(
        study_name=f"{model_name}_warhead_macro_f1",
        storage=f"sqlite:///{db_path.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    saved = study.user_attrs.get("data_fingerprint")
    if saved is None:
        study.set_user_attr("data_fingerprint", fingerprint)
    elif saved != fingerprint:
        raise ValueError(
            f"{db_path.name} was built with different data/features. "
            f"Delete it to retune from scratch."
        )
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials))
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(n_trials - n_done, 0)
    print(
        f"\n[OPTUNA] {model_name}: {n_done} done, {remaining} remaining (target {n_trials}). "
        f"DB={db_path.name}",
        flush=True,
    )

    def callback(study, trial):
        export_results(study, model_name, fingerprint, finalize_params)

    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=[callback], gc_after_trial=True)
    export_results(study, model_name, fingerprint, finalize_params)
    best = study.best_trial
    print(
        f"\n[OPTUNA] Best trial {best.number}: macro F1={best.value:.4f} "
        f"± {best.user_attrs.get('std_macro_F1', float('nan')):.4f}",
        flush=True,
    )
    for k, v in finalize_params(best).items():
        print(f"    {k}: {v}", flush=True)
    return study
