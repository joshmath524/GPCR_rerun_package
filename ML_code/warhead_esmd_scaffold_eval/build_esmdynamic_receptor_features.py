#!/usr/bin/env python3
"""
Build fixed-length receptor features from ESMDynamic dynamic contact maps.

Uses pocket residue IDs from Josh_Receptor_Features (1-indexed UniProt positions)
and ESMDynamic probability / predicted contact maps at 5 temperatures.

Writes:
  cache/esmdynamic_receptor_features.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "cache"
ESMD_OUT = ROOT / "ESMDynamic_68_receptors" / "ESMDynamic_68_out" / "ESMDynamic_68_out"
POCKET_ROOT = ROOT / "Josh_Receptor_Features"
TEMPS = ["320K", "348K", "379K", "413K", "450K"]


def pocket_residue_ids(receptor: str) -> np.ndarray:
    folder = POCKET_ROOT / receptor
    files = list(folder.glob("*_pocket_residues_with_conservation.csv"))
    if not files:
        raise FileNotFoundError(f"No pocket CSV for {receptor}")
    df = pd.read_csv(files[0])
    if "residue_id" not in df.columns:
        raise ValueError(f"residue_id missing in {files[0]}")
    ids = pd.to_numeric(df["residue_id"], errors="coerce").dropna().astype(int).to_numpy()
    return np.unique(ids)


def load_prob(receptor: str, temp: str) -> np.ndarray:
    path = ESMD_OUT / receptor / "dynamic" / f"{receptor}_dynamic_prob_{temp}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Bad contact map shape {arr.shape} for {path}")
    return arr.astype(np.float64)


def features_for_receptor(receptor: str) -> dict:
    pocket_ids = pocket_residue_ids(receptor)
    dens = []
    pocket_deg_mean = []
    pocket_deg_std = []
    pocket_pp = []
    for temp in TEMPS:
        prob = load_prob(receptor, temp)
        L = prob.shape[0]
        idx = pocket_ids[(pocket_ids >= 1) & (pocket_ids <= L)] - 1  # to 0-index
        if len(idx) == 0:
            raise ValueError(f"No in-range pocket residues for {receptor} (L={L})")

        # Symmetric contact probability density (exclude diagonal)
        mask = ~np.eye(L, dtype=bool)
        dens.append(float(prob[mask].mean()))

        # Mean contact probability per pocket residue (row mean excl. self)
        row = prob[idx].copy()
        row[:, idx] = row[:, idx]  # keep pocket-pocket
        # zero diagonal contributions for selected rows
        for i, ii in enumerate(idx):
            row[i, ii] = 0.0
        deg = row.sum(axis=1) / max(L - 1, 1)
        pocket_deg_mean.append(float(deg.mean()))
        pocket_deg_std.append(float(deg.std()))

        # Pocket–pocket block mean (excl. diagonal)
        block = prob[np.ix_(idx, idx)]
        if block.size <= 1:
            pocket_pp.append(0.0)
        else:
            bmask = ~np.eye(block.shape[0], dtype=bool)
            pocket_pp.append(float(block[bmask].mean()) if bmask.any() else 0.0)

    dens = np.asarray(dens)
    pdm = np.asarray(pocket_deg_mean)
    pds = np.asarray(pocket_deg_std)
    pp = np.asarray(pocket_pp)

    return {
        "Receptor": receptor,
        "ESMD_n_pocket": int(len(pocket_ids)),
        "ESMD_seq_len": int(load_prob(receptor, TEMPS[0]).shape[0]),
        "ESMD_global_prob_mean": float(dens.mean()),
        "ESMD_global_prob_std": float(dens.std()),
        "ESMD_global_prob_delta_450_320": float(dens[-1] - dens[0]),
        "ESMD_pocket_degree_mean": float(pdm.mean()),
        "ESMD_pocket_degree_std": float(pdm.std()),
        "ESMD_pocket_degree_delta_450_320": float(pdm[-1] - pdm[0]),
        "ESMD_pocket_degree_dispersion_mean": float(pds.mean()),
        "ESMD_pocket_pp_prob_mean": float(pp.mean()),
        "ESMD_pocket_pp_prob_std": float(pp.std()),
        "ESMD_pocket_pp_prob_delta_450_320": float(pp[-1] - pp[0]),
        # per-temp anchors (320 and 450)
        "ESMD_global_prob_320K": float(dens[0]),
        "ESMD_global_prob_450K": float(dens[-1]),
        "ESMD_pocket_degree_320K": float(pdm[0]),
        "ESMD_pocket_degree_450K": float(pdm[-1]),
        "ESMD_pocket_pp_prob_320K": float(pp[0]),
        "ESMD_pocket_pp_prob_450K": float(pp[-1]),
    }


def main() -> None:
    if not ESMD_OUT.exists():
        raise FileNotFoundError(
            f"Missing ESMDynamic outputs at {ESMD_OUT}. "
            "Place ESMDynamic_68_receptors/ under the package root."
        )
    receptors = sorted([p.name for p in ESMD_OUT.iterdir() if p.is_dir()])
    if len(receptors) != 68:
        print(f"[WARNING] Expected 68 receptors, found {len(receptors)}", flush=True)

    rows = []
    for i, rec in enumerate(receptors, 1):
        print(f"[{i}/{len(receptors)}] {rec}", flush=True)
        rows.append(features_for_receptor(rec))

    df = pd.DataFrame(rows).sort_values("Receptor").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "esmdynamic_receptor_features.csv"
    df.to_csv(out_csv, index=False)
    meta = {
        "n_receptors": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "temps": TEMPS,
        "source": str(ESMD_OUT),
        "feature_columns": [c for c in df.columns if c != "Receptor"],
    }
    (OUT_DIR / "esmdynamic_receptor_features_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[OK] Wrote {out_csv} shape={df.shape}", flush=True)


if __name__ == "__main__":
    main()
