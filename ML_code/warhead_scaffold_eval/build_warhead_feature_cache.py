#!/usr/bin/env python3
"""
Stream _NEW tables into a parquet cache (one file at a time) for warhead scaffold evals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "ML_code"))
import shared_utilities as su  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(parents=True, exist_ok=True)


def name_level_drop_duplicates(columns: list) -> list[str]:
    keep, seen = [], set()
    for c in columns:
        s = str(c)
        base, sep, suffix = s.rpartition(".")
        if sep and suffix.isdigit() and base in seen:
            continue
        keep.append(s)
        seen.add(s)
    return keep


def reference_descriptor_columns() -> list[str]:
    path = ROOT / "5-HT1A" / "5-HT1A_agonist_enriched_NEW.csv"
    cols = name_level_drop_duplicates(list(pd.read_csv(path, nrows=0).columns))
    desc = [c for c in cols if not su.should_exclude_column(c)]
    assert sum(c.startswith("warhead_") for c in desc) == 27
    print(f"Reference descriptors: {len(desc)}", flush=True)
    return desc


def find_new_table(folder: Path, rec: str, kind: str) -> Path | None:
    for ext in (".csv", ".xlsx"):
        p = folder / f"{rec}_{kind}_NEW{ext}"
        if p.exists():
            return p
    return None


def load_one(path: Path, desc_cols: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        header = list(pd.read_csv(path, nrows=0).columns)
    else:
        header = list(pd.read_excel(path, nrows=0).columns)
    kept = set(name_level_drop_duplicates(list(map(str, header))))
    meta = [c for c in ("SMILES", "CID") if c in kept]
    cols = meta + [c for c in desc_cols if c in kept]

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=cols, low_memory=False)
    else:
        df = pd.read_excel(path, usecols=cols)

    # Align + cast in one shot
    values = {}
    for c in desc_cols:
        if c in df.columns:
            values[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
        else:
            values[c] = np.zeros(len(df), dtype=np.float32)
    out = pd.DataFrame(values)
    out.insert(0, "SMILES", df["SMILES"].astype(str).to_numpy())
    if "CID" in df.columns:
        out.insert(1, "CID", df["CID"].to_numpy())
    return out


def main() -> None:
    desc_cols = reference_descriptor_columns()
    receptors = [
        "5-HT1A","5-HT1B","5-HT1D","5-HT1F","5-HT2A","5-HT2B","5-HT2C",
        "5-HT4","5-HT5A","5-HT6","5-HT7","A1","A2A","A2B","A3",
        "alpha1A","alpha1B","alpha2C","beta1","beta2","beta3",
        "BLT1","CB1","CB2","CysLT1","CysLT2","D1","D2","D3","D4","D5",
        "EP2","EP3","EP4","FFA1","FFA2","FFA3","FFA4","FP",
        "GPBA","GPR119","GPR139","GPR174","GPR35","H1","H2","H3","H4","HCA2","IP",
        "LPA1","M1","M2","M3","M4","M5","MT1","MT2","P2Y1","P2Y12","PAF",
        "S1P1","S1P2","S1P3","S1P5","succinate","TA1","TP",
    ]

    writer = None
    total = 0
    for i, rec in enumerate(receptors, 1):
        folder = ROOT / rec
        if not folder.is_dir():
            continue
        for label, kind in su.LIGAND_FILE_KINDS:
            path = find_new_table(folder, rec, kind)
            if path is None:
                continue
            try:
                df = load_one(path, desc_cols)
            except Exception as e:
                print(f"FAIL {path}: {e}", flush=True)
                raise
            df.insert(1, "Receptor", rec)
            df.insert(2, "Label_Type", label)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT / "ligand_features.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            total += len(df)
            print(f"[{i}/{len(receptors)}] {rec} {label}: +{len(df)} (total {total:,}) {path.name}", flush=True)
            del df, table

    if writer is None:
        raise RuntimeError("No tables written")
    writer.close()

    # Conflict drop requires full reload — do in second pass with columns only needed
    print("Reloading parquet for conflict cleaning...", flush=True)
    data = pd.read_parquet(OUT / "ligand_features.parquet", columns=["SMILES", "Receptor", "Label_Type"])
    # Attach a row id
    data = data.reset_index(drop=True)
    data["_row"] = np.arange(len(data))
    cleaned, removed = su.drop_conflicting_labels(data)
    keep_idx = cleaned["_row"].to_numpy()
    print(f"Keeping {len(keep_idx):,} / {total:,} rows", flush=True)

    full = pd.read_parquet(OUT / "ligand_features.parquet")
    full = full.iloc[keep_idx].reset_index(drop=True)
    full.to_parquet(OUT / "ligand_features.parquet", index=False, compression="zstd")
    (OUT / "feature_columns.json").write_text(json.dumps(desc_cols, indent=2))
    print(f"Done. rows={len(full):,} cols={len(full.columns)} "
          f"size={ (OUT/'ligand_features.parquet').stat().st_size/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
