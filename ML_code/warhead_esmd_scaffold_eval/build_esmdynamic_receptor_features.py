#!/usr/bin/env python3
"""
Build pocket-binned ESMDynamic receptor features from dynamic contact maps.

Bins pocket residues by:
  - shell (first / second)
  - region (TM / ECL / other) from gpcr_position
  - sulfur-containing pocket residues (Cys/Met)

Also keeps global + whole-pocket summaries and 450K−320K deltas.

Writes: cache/esmdynamic_receptor_features.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "cache"
ESMD_OUT = ROOT / "ESMDynamic_68_receptors" / "ESMDynamic_68_out" / "ESMDynamic_68_out"
POCKET_ROOT = ROOT / "Josh_Receptor_Features"
TEMPS = ["320K", "348K", "379K", "413K", "450K"]

SHELL_BINS = ("first", "second")
REGION_BINS = ("TM", "ECL", "other")


def _region_from_gpcr_position(pos: str) -> str:
    s = str(pos).upper()
    if s.startswith("TM"):
        return "TM"
    if "ECL" in s or "ICL" in s:
        return "ECL"
    return "other"


def load_pocket_table(receptor: str) -> pd.DataFrame:
    folder = POCKET_ROOT / receptor
    files = list(folder.glob("*_pocket_residues_with_conservation.csv"))
    if not files:
        raise FileNotFoundError(f"No pocket CSV for {receptor}")
    df = pd.read_csv(files[0])
    if "residue_id" not in df.columns:
        raise ValueError(f"residue_id missing in {files[0]}")
    df = df.copy()
    df["residue_id"] = pd.to_numeric(df["residue_id"], errors="coerce")
    df = df.dropna(subset=["residue_id"])
    df["residue_id"] = df["residue_id"].astype(int)
    df["shell_norm"] = df["shell"].astype(str).str.lower().str.strip() if "shell" in df.columns else "unknown"
    if "gpcr_position" in df.columns:
        df["region"] = df["gpcr_position"].map(_region_from_gpcr_position)
    else:
        df["region"] = "other"
    resn = df["residue_name"].astype(str).str.upper() if "residue_name" in df.columns else pd.Series([""] * len(df))
    sulfur_flag = df["sulfur"] if "sulfur" in df.columns else False
    if not isinstance(sulfur_flag, pd.Series):
        sulfur_flag = pd.Series([False] * len(df))
    sulfur_bool = sulfur_flag.fillna(False).astype(bool) | resn.isin(["CYS", "MET"])
    df["is_sulfur"] = sulfur_bool
    df["is_cys"] = resn.eq("CYS")
    return df.drop_duplicates(subset=["residue_id"], keep="first")


def load_prob(receptor: str, temp: str) -> np.ndarray:
    path = ESMD_OUT / receptor / "dynamic" / f"{receptor}_dynamic_prob_{temp}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Bad contact map shape {arr.shape} for {path}")
    return arr.astype(np.float64)


def _degree_stats(prob: np.ndarray, ids: np.ndarray) -> tuple[float, float]:
    """Mean/std contact degree for residues ids (1-indexed)."""
    L = prob.shape[0]
    idx = ids[(ids >= 1) & (ids <= L)] - 1
    if len(idx) == 0:
        return 0.0, 0.0
    row = prob[idx].copy()
    for i, ii in enumerate(idx):
        row[i, ii] = 0.0
    deg = row.sum(axis=1) / max(L - 1, 1)
    return float(deg.mean()), float(deg.std())


def _block_mean(prob: np.ndarray, ids: np.ndarray) -> float:
    L = prob.shape[0]
    idx = ids[(ids >= 1) & (ids <= L)] - 1
    if len(idx) <= 1:
        return 0.0
    block = prob[np.ix_(idx, idx)]
    mask = ~np.eye(block.shape[0], dtype=bool)
    return float(block[mask].mean()) if mask.any() else 0.0


def features_for_receptor(receptor: str) -> dict:
    pocket = load_pocket_table(receptor)
    all_ids = pocket["residue_id"].to_numpy()

    # group id arrays
    groups: dict[str, np.ndarray] = {"pocket": all_ids}
    for shell in SHELL_BINS:
        groups[f"shell_{shell}"] = pocket.loc[pocket["shell_norm"] == shell, "residue_id"].to_numpy()
    for region in REGION_BINS:
        groups[f"region_{region}"] = pocket.loc[pocket["region"] == region, "residue_id"].to_numpy()
    groups["sulfur"] = pocket.loc[pocket["is_sulfur"], "residue_id"].to_numpy()
    groups["cys"] = pocket.loc[pocket["is_cys"], "residue_id"].to_numpy()

    # per-temp collectors
    series: dict[str, list[float]] = {f"{g}_degree_mean": [] for g in groups}
    series.update({f"{g}_degree_std": [] for g in groups})
    series.update({f"{g}_pp_mean": [] for g in groups if g in {"pocket", "shell_first", "shell_second", "region_TM", "region_ECL"}})
    global_dens: list[float] = []

    for temp in TEMPS:
        prob = load_prob(receptor, temp)
        L = prob.shape[0]
        mask = ~np.eye(L, dtype=bool)
        global_dens.append(float(prob[mask].mean()))
        for gname, ids in groups.items():
            dmean, dstd = _degree_stats(prob, ids)
            series[f"{gname}_degree_mean"].append(dmean)
            series[f"{gname}_degree_std"].append(dstd)
            key = f"{gname}_pp_mean"
            if key in series:
                series[key].append(_block_mean(prob, ids))

    out = {
        "Receptor": receptor,
        "ESMD_n_pocket": int(len(all_ids)),
        "ESMD_seq_len": int(load_prob(receptor, TEMPS[0]).shape[0]),
        "ESMD_n_shell_first": int((pocket["shell_norm"] == "first").sum()),
        "ESMD_n_shell_second": int((pocket["shell_norm"] == "second").sum()),
        "ESMD_n_region_TM": int((pocket["region"] == "TM").sum()),
        "ESMD_n_region_ECL": int((pocket["region"] == "ECL").sum()),
        "ESMD_n_sulfur": int(pocket["is_sulfur"].sum()),
        "ESMD_n_cys": int(pocket["is_cys"].sum()),
        "ESMD_global_prob_mean": float(np.mean(global_dens)),
        "ESMD_global_prob_std": float(np.std(global_dens)),
        "ESMD_global_prob_delta_450_320": float(global_dens[-1] - global_dens[0]),
        "ESMD_global_prob_320K": float(global_dens[0]),
        "ESMD_global_prob_450K": float(global_dens[-1]),
    }

    # map group series -> feature names (keep prior names for pocket)
    rename = {
        "pocket_degree_mean": "ESMD_pocket_degree_mean",
        "pocket_degree_std": "ESMD_pocket_degree_dispersion_mean",  # mean over temps of per-temp std handled below
        "pocket_pp_mean": "ESMD_pocket_pp_prob_mean",
    }

    for key, values in series.items():
        arr = np.asarray(values, dtype=float)
        # stable feature names
        if key == "pocket_degree_mean":
            out["ESMD_pocket_degree_mean"] = float(arr.mean())
            out["ESMD_pocket_degree_std"] = float(arr.std())
            out["ESMD_pocket_degree_delta_450_320"] = float(arr[-1] - arr[0])
            out["ESMD_pocket_degree_320K"] = float(arr[0])
            out["ESMD_pocket_degree_450K"] = float(arr[-1])
        elif key == "pocket_degree_std":
            out["ESMD_pocket_degree_dispersion_mean"] = float(arr.mean())
        elif key == "pocket_pp_mean":
            out["ESMD_pocket_pp_prob_mean"] = float(arr.mean())
            out["ESMD_pocket_pp_prob_std"] = float(arr.std())
            out["ESMD_pocket_pp_prob_delta_450_320"] = float(arr[-1] - arr[0])
            out["ESMD_pocket_pp_prob_320K"] = float(arr[0])
            out["ESMD_pocket_pp_prob_450K"] = float(arr[-1])
        else:
            # e.g. shell_first_degree_mean -> ESMD_shell_first_degree_mean
            base = f"ESMD_{key}"
            out[f"{base}_mean"] = float(arr.mean())
            out[f"{base}_delta_450_320"] = float(arr[-1] - arr[0])
            if key.endswith("_degree_mean") or key.endswith("_pp_mean"):
                out[f"{base}_320K"] = float(arr[0])
                out[f"{base}_450K"] = float(arr[-1])

    return out


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
        "encoding": "pocket_binned_shell_region_sulfur_v2",
        "feature_columns": [c for c in df.columns if c != "Receptor"],
    }
    (OUT_DIR / "esmdynamic_receptor_features_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[OK] Wrote {out_csv} shape={df.shape}", flush=True)


if __name__ == "__main__":
    main()
