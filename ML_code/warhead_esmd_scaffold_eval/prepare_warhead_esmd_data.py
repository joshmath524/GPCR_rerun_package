#!/usr/bin/env python3
"""Assemble X/y/SMILES for warhead + ESMDynamic scaffold evaluations."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ML_code"
sys.path.insert(0, str(ML))

from shared_utilities import aggregate_receptor_features  # noqa: E402

HERE = Path(__file__).resolve().parent
WARHEAD_CACHE = ML / "warhead_scaffold_eval" / "cache"
ESMD_CACHE = HERE / "cache"

RECEPTORS = [
    "5-HT1A", "5-HT1B", "5-HT1D", "5-HT1F", "5-HT2A", "5-HT2B", "5-HT2C", "5-HT4",
    "5-HT5A", "5-HT6", "5-HT7", "A1", "A2A", "A2B", "A3", "alpha1A", "alpha1B", "alpha2C",
    "beta1", "beta2", "beta3", "BLT1", "CB1", "CB2", "CysLT1", "CysLT2", "D1", "D2", "D3",
    "D4", "D5", "EP2", "EP3", "EP4", "FFA1", "FFA2", "FFA3", "FFA4", "FP", "GPBA", "GPR119",
    "GPR139", "GPR174", "GPR35", "H1", "H2", "H3", "H4", "HCA2", "IP", "LPA1", "M1", "M2",
    "M3", "M4", "M5", "MT1", "MT2", "P2Y1", "P2Y12", "PAF", "S1P1", "S1P2", "S1P3", "S1P5",
    "succinate", "TA1", "TP",
]

INTERACTION_PAIRS = [
    ("LogP", "aromatic_ratio"), ("LogP", "num_basic"), ("TPSA", "acidic_ratio"),
    ("HBD", "num_hydroxyl"), ("HBA", "num_basic"), ("HBA", "num_amide"),
    ("AromaticRings", "aromatic_ratio"), ("AromaticRings", "num_aromatic"),
    ("FormalCharge", "num_charge_positive"), ("FormalCharge", "num_charge_negative"),
    ("MolWt", "num_residues"), ("RotatableBonds", "num_residues"),
    ("LogP", "avg_conservation"), ("TPSA", "avg_conservation"),
]


def sanitize_feature_name(name: str) -> str:
    sanitized = re.sub(r"[^\w\-.]", "_", str(name))
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized if sanitized else "feature_" + str(hash(name) % 10000)


def load_esmd_features() -> pd.DataFrame:
    path = ESMD_CACHE / "esmdynamic_receptor_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run build_esmdynamic_receptor_features.py first "
            "(requires ESMDynamic_68_receptors outputs)."
        )
    df = pd.read_csv(path)
    if "Receptor" not in df.columns:
        raise ValueError("ESMDynamic feature table missing Receptor column")
    return df


def build_warhead_esmd_dataset():
    """Return X (DataFrame), y (ndarray), smiles (Series), class_names (list)."""
    pq = WARHEAD_CACHE / "ligand_features.parquet"
    cols_path = WARHEAD_CACHE / "feature_columns.json"
    if not pq.exists():
        raise FileNotFoundError(f"Missing warhead cache {pq}")

    data_df = pd.read_parquet(pq)
    ligand_cols = json.loads(cols_path.read_text())
    n_wh_counts = sum(1 for c in ligand_cols if str(c).startswith("warhead_"))
    n_wh = n_wh_counts + (1 if "has_covalent_warhead" in ligand_cols else 0)
    print(
        f"[DATA] Warhead cache {data_df.shape}; ligand cols={len(ligand_cols)} "
        f"(warhead_counts={n_wh_counts})",
        flush=True,
    )
    if n_wh_counts < 10:
        raise RuntimeError(
            f"Expected Table-2 warhead_* count columns, found {n_wh_counts}. "
            "Run ML_code/warhead_scaffold_eval/rebuild_warhead_counts_cache.py"
        )
    if "has_covalent_warhead" not in ligand_cols:
        raise RuntimeError("Warhead cache missing has_covalent_warhead — rebuild counts cache")

    print("[DATA] Aggregating pocket receptor features (+ Cys)...", flush=True)
    receptor_feats = {}
    for rec in RECEPTORS:
        feats = aggregate_receptor_features(rec)
        if not feats:
            continue
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
    print(f"  [OK] pocket features for {len(receptor_df)} receptors", flush=True)

    esmd = load_esmd_features()
    esmd_cols = [c for c in esmd.columns if c != "Receptor"]
    print(f"  [OK] ESMDynamic features: {len(esmd_cols)} cols for {len(esmd)} receptors", flush=True)

    merged = data_df.merge(receptor_df.reset_index(), on="Receptor", how="left")
    merged[receptor_df.columns] = merged[receptor_df.columns].fillna(0)
    merged = merged.merge(esmd, on="Receptor", how="left")
    merged[esmd_cols] = merged[esmd_cols].fillna(0)

    interaction_cols = []
    for lig, rec in INTERACTION_PAIRS:
        if lig in merged.columns and rec in merged.columns:
            name = f"INT_{lig}_X_{rec}"
            merged[name] = merged[lig] * merged[rec]
            interaction_cols.append(name)

    # Warhead load / presence helpers from counts
    warhead_count_cols = [c for c in ligand_cols if str(c).startswith("warhead_")]
    derived_wh: list[str] = []
    if warhead_count_cols:
        merged["warhead_total_count"] = merged[warhead_count_cols].sum(axis=1)
        derived_wh.append("warhead_total_count")
        for motif in [
            "warhead_acrylamide",
            "warhead_vinyl_sulfone",
            "warhead_vinyl_sulfonamide",
            "warhead_chloroacetamide",
            "warhead_bromoacetamide",
            "warhead_iodoacetamide",
            "warhead_maleimide",
            "warhead_epoxide",
        ]:
            if motif in merged.columns:
                present = f"{motif}_present"
                merged[present] = (merged[motif] > 0).astype(float)
                derived_wh.append(present)

    # Warhead × pocket nucleophile / ESMD pocket-binned dynamics
    warhead_sides = []
    if "has_covalent_warhead" in merged.columns:
        warhead_sides.append(("has_covalent_warhead", "hasWH"))
    if "warhead_total_count" in merged.columns:
        warhead_sides.append(("warhead_total_count", "whLoad"))
    for motif in [
        "warhead_acrylamide",
        "warhead_vinyl_sulfone",
        "warhead_chloroacetamide",
        "warhead_maleimide",
    ]:
        if motif in merged.columns:
            warhead_sides.append((motif, motif.replace("warhead_", "wh_")))

    esmd_sides = [
        ("has_pocket_cys", "pocket_cys"),
        ("num_cys", "num_cys"),
        ("num_sulfur", "num_sulfur"),
        ("ESMD_pocket_degree_mean", "esmd_pocket_degree"),
        ("ESMD_pocket_pp_prob_mean", "esmd_pocket_pp"),
        ("ESMD_pocket_degree_delta_450_320", "esmd_pocket_deg_delta"),
        ("ESMD_shell_first_degree_mean_mean", "esmd_shell1_deg"),
        ("ESMD_shell_second_degree_mean_mean", "esmd_shell2_deg"),
        ("ESMD_region_TM_degree_mean_mean", "esmd_TM_deg"),
        ("ESMD_region_ECL_degree_mean_mean", "esmd_ECL_deg"),
        ("ESMD_sulfur_degree_mean_mean", "esmd_sulfur_deg"),
        ("ESMD_cys_degree_mean_mean", "esmd_cys_deg"),
        ("ESMD_shell_first_pp_mean_mean", "esmd_shell1_pp"),
        ("ESMD_global_prob_mean", "esmd_global"),
    ]

    for wcol, wtag in warhead_sides:
        for rcol, rtag in esmd_sides:
            if wcol in merged.columns and rcol in merged.columns:
                name = f"INT_{wtag}_X_{rtag}"
                merged[name] = merged[wcol] * merged[rcol]
                interaction_cols.append(name)

    if "SMILES" not in merged.columns:
        raise ValueError("SMILES column required")
    smiles_check = merged["SMILES"].astype(str)
    valid = (
        ~smiles_check.isna()
        & (smiles_check.str.lower() != "nan")
        & (smiles_check.str.strip() != "")
    )
    if not valid.all():
        print(f"  [WARNING] Dropping {(~valid).sum()} invalid SMILES rows", flush=True)
        merged = merged[valid].reset_index(drop=True)

    # refresh esmd cols after merge (includes newly built binned features)
    esmd_cols = [c for c in merged.columns if str(c).startswith("ESMD_")]
    feature_cols = (
        list(ligand_cols)
        + list(receptor_df.columns)
        + list(esmd_cols)
        + list(derived_wh)
        + interaction_cols
    )
    # unique preserve order
    seen = set()
    feature_cols = [c for c in feature_cols if c in merged.columns and not (c in seen or seen.add(c))]
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
    smiles = merged["SMILES"].astype(str).reset_index(drop=True)

    print(
        f"  [OK] Matrix {X.shape}; warhead≈{n_wh}; ESMD={len(esmd_cols)}; "
        f"interactions={len(interaction_cols)}; classes={list(enc.classes_)}",
        flush=True,
    )
    return X, y, smiles, list(enc.classes_)
