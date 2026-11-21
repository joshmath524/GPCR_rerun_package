#!/usr/bin/env python3
"""
Put the 27 Table-2 warhead_* count columns into the ligand feature cache.

Reads counts from receptor *_NEW.csv / *_NEW.xlsx tables (keyed by SMILES).
Replaces curated WHB_* exclusive binaries. Sets has_covalent_warhead if any
warhead_* count > 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache"
HAS_WARHEAD_COL = "has_covalent_warhead"


def _stems() -> list[Path]:
    """Unique NEW stems from csv/xlsx under receptor folders."""
    stems = {}
    for path in list(ROOT.glob("*/*_NEW.csv")) + list(ROOT.glob("*/*_NEW.xlsx")):
        stems[path.with_suffix("")] = True
    return sorted(stems.keys(), key=lambda p: str(p))


def collect_counts() -> tuple[dict[str, dict[str, float]], list[str]]:
    warhead_cols: list[str] | None = None
    by_smiles: dict[str, dict[str, float]] = {}
    stems = _stems()
    print(f"Scanning {len(stems)} *_NEW tables for warhead_* counts...", flush=True)
    for i, stem in enumerate(stems, 1):
        csv_path = stem.with_suffix(".csv")
        xlsx_path = stem.with_suffix(".xlsx")
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
        elif xlsx_path.exists():
            df = pd.read_excel(xlsx_path)
        else:
            continue
        cols = [c for c in df.columns if str(c).startswith("warhead_")]
        if not cols or "SMILES" not in df.columns:
            continue
        if warhead_cols is None:
            warhead_cols = list(cols)
        else:
            for c in cols:
                if c not in warhead_cols:
                    warhead_cols.append(c)
        sub = df[["SMILES", *cols]].copy()
        sub["SMILES"] = sub["SMILES"].astype(str)
        sub = sub[(sub["SMILES"].str.len() > 0) & (sub["SMILES"].str.lower() != "nan")]
        sub = sub.drop_duplicates(subset=["SMILES"], keep="first")
        for smi, rec in zip(sub["SMILES"].tolist(), sub[cols].to_dict("records")):
            if smi in by_smiles:
                continue
            by_smiles[smi] = {c: float(rec[c]) if pd.notna(rec[c]) else 0.0 for c in cols}
        if i % 25 == 0:
            print(f"  {i}/{len(stems)} tables; unique SMILES with counts={len(by_smiles):,}", flush=True)
    if not warhead_cols:
        raise RuntimeError("No warhead_* columns found on *_NEW tables")
    print(f"  [OK] {len(warhead_cols)} warhead count cols; {len(by_smiles):,} unique SMILES", flush=True)
    return by_smiles, warhead_cols


def main() -> None:
    pq = CACHE / "ligand_features.parquet"
    cols_path = CACHE / "feature_columns.json"
    if not pq.exists():
        raise FileNotFoundError(pq)

    print(f"Loading {pq}...", flush=True)
    df = pd.read_parquet(pq)
    print(f"  rows={len(df):,} cols={len(df.columns)}", flush=True)

    drop = [
        c
        for c in df.columns
        if str(c).startswith("warhead_") or str(c).startswith("WHB_") or c == HAS_WARHEAD_COL
    ]
    if drop:
        df = df.drop(columns=drop)
        print(f"  dropped {len(drop)} old warhead/binary columns", flush=True)

    by_smiles, warhead_cols = collect_counts()
    smiles = df["SMILES"].astype(str)
    missing = 0
    rows = []
    for smi in smiles:
        if smi in by_smiles:
            rows.append(by_smiles[smi])
        else:
            missing += 1
            rows.append({c: 0.0 for c in warhead_cols})
    wh = pd.DataFrame(rows, index=df.index)
    for c in warhead_cols:
        if c not in wh.columns:
            wh[c] = 0.0
    wh = wh[warhead_cols].astype(np.float32)
    wh[HAS_WARHEAD_COL] = (wh[warhead_cols].sum(axis=1) > 0).astype(np.float32)
    print(f"  missing SMILES after csv+xlsx: {missing:,}", flush=True)
    print(f"  ligands with any warhead count>0: {int((wh[HAS_WARHEAD_COL] > 0).sum()):,}", flush=True)

    old_feats = json.loads(cols_path.read_text()) if cols_path.exists() else []
    base_feats = [
        c
        for c in old_feats
        if not str(c).startswith("warhead_")
        and not str(c).startswith("WHB_")
        and c != HAS_WARHEAD_COL
        and c in df.columns
    ]
    # also keep any non-warhead columns present in df but not listed
    for c in df.columns:
        if c in {"SMILES", "Receptor", "Label_Type"}:
            continue
        if (
            c not in base_feats
            and not str(c).startswith("warhead_")
            and not str(c).startswith("WHB_")
            and c != HAS_WARHEAD_COL
        ):
            base_feats.append(c)
    new_feats = base_feats + warhead_cols + [HAS_WARHEAD_COL]

    out = pd.concat([df, wh], axis=1)
    out.to_parquet(pq, index=False, compression="zstd")
    cols_path.write_text(json.dumps(new_feats, indent=2))
    meta = {
        "n_rows": len(out),
        "n_features": len(new_feats),
        "warhead_features": warhead_cols + [HAS_WARHEAD_COL],
        "n_with_warhead": int((wh[HAS_WARHEAD_COL] > 0).sum()),
        "encoding": "table2_warhead_counts",
        "n_warhead_count_cols": len(warhead_cols),
        "n_missing_smiles": int(missing),
    }
    (CACHE / "warhead_encoding_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {pq} ({pq.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"Features: {len(new_feats)} (warhead counts={len(warhead_cols)} + {HAS_WARHEAD_COL})", flush=True)


if __name__ == "__main__":
    main()
