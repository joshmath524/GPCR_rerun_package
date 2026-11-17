#!/usr/bin/env python3
"""
Replace old warhead_* count columns in the scaffold cache with curated exclusive
binary warheads + has_covalent_warhead. Drops binary cols below MIN_SUPPORT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "ML_code"))
from warhead_smarts import (  # noqa: E402
    BINARY_PREFIX,
    CURATED_WARHEAD_SMARTS_PRIORITY,
    HAS_WARHEAD_COL,
    MIN_SUPPORT,
)

CACHE = Path(__file__).resolve().parent / "cache"


def compile_patterns():
    out = []
    for name, smarts in CURATED_WARHEAD_SMARTS_PRIORITY:
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            raise ValueError(f"Bad SMARTS {name}: {smarts}")
        out.append((name, mol))
    return out


def exclusive_binaries(smiles: str, patterns) -> dict[str, int]:
    """Priority winner gets 1; others 0. Also returns has_covalent_warhead."""
    names = [n for n, _ in patterns]
    result = {f"{BINARY_PREFIX}{n}": 0 for n in names}
    result[HAS_WARHEAD_COL] = 0
    if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
        return result
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result
    for name, pat in patterns:
        if mol.HasSubstructMatch(pat):
            result[f"{BINARY_PREFIX}{name}"] = 1
            result[HAS_WARHEAD_COL] = 1
            break  # exclusive: most specific wins
    return result


def main() -> None:
    pq = CACHE / "ligand_features.parquet"
    cols_path = CACHE / "feature_columns.json"
    if not pq.exists():
        raise FileNotFoundError(pq)

    print(f"Loading {pq}...", flush=True)
    df = pd.read_parquet(pq)
    print(f"  rows={len(df):,} cols={len(df.columns)}", flush=True)

    drop = [c for c in df.columns if str(c).startswith("warhead_") or str(c).startswith(BINARY_PREFIX) or c == HAS_WARHEAD_COL]
    if drop:
        df = df.drop(columns=drop)
        print(f"  dropped {len(drop)} old warhead columns", flush=True)

    patterns = compile_patterns()
    print(f"Compiled {len(patterns)} curated priority SMARTS", flush=True)

    # Cache by unique SMILES
    smiles = df["SMILES"].astype(str)
    uniq = smiles.unique()
    cache = {}
    for i, smi in enumerate(uniq):
        cache[smi] = exclusive_binaries(smi, patterns)
        if (i + 1) % 5000 == 0:
            print(f"  SMARTS {i+1:,}/{len(uniq):,}", flush=True)
    print(f"  SMARTS done for {len(uniq):,} unique SMILES", flush=True)

    wh = pd.DataFrame([cache[s] for s in smiles], index=df.index)
    # Support filter on binary class cols (keep has_covalent_warhead always)
    keep = [HAS_WARHEAD_COL]
    print("Support (positives / ligands):", flush=True)
    for c in wh.columns:
        if c == HAS_WARHEAD_COL:
            n = int((wh[c] > 0).sum())
            print(f"  {c}: {n}", flush=True)
            continue
        n = int((wh[c] > 0).sum())
        status = "KEEP" if n >= MIN_SUPPORT else "DROP"
        print(f"  {c}: {n} [{status}]", flush=True)
        if n >= MIN_SUPPORT:
            keep.append(c)
    wh = wh[keep].astype(np.float32)

    # Rebuild feature column list: non-warhead descriptors + new warhead bins
    old_feats = json.loads(cols_path.read_text()) if cols_path.exists() else []
    base_feats = [c for c in old_feats if not str(c).startswith("warhead_")]
    # Also strip any that vanished from df
    base_feats = [c for c in base_feats if c in df.columns]
    new_feats = base_feats + list(wh.columns)

    out = pd.concat([df, wh], axis=1)
    out.to_parquet(pq, index=False, compression="zstd")
    cols_path.write_text(json.dumps(new_feats, indent=2))
    meta = {
        "n_rows": len(out),
        "n_features": len(new_feats),
        "warhead_features": list(wh.columns),
        "min_support": MIN_SUPPORT,
        "n_with_warhead": int((wh[HAS_WARHEAD_COL] > 0).sum()),
        "encoding": "exclusive_binary_priority",
    }
    (CACHE / "warhead_encoding_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {pq} ({pq.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"Features: {len(new_feats)} (warhead bins kept: {len(wh.columns)})", flush=True)
    print(f"Ligands with has_covalent_warhead=1: {meta['n_with_warhead']:,}", flush=True)


if __name__ == "__main__":
    main()
