#!/usr/bin/env python3
"""
Add all 27 Table-2 warhead count columns to every ligand *_NEW spreadsheet.

Writes/updates paired *_NEW.csv for every receptor table (scaffold loaders
prefer CSV). Also rewrites *_NEW.xlsx. Parallelized across files.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from warhead_smarts import WARHEAD_COLUMNS, WARHEAD_SMARTS  # noqa: E402


def _compile_patterns():
    compiled = {}
    for name, smarts in WARHEAD_SMARTS.items():
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            raise ValueError(f"Invalid SMARTS for {name}: {smarts}")
        compiled[name] = mol
    return compiled


def _counts_for_smiles(smiles: str, patterns) -> list[int]:
    zeros = [0] * len(WARHEAD_COLUMNS)
    if not isinstance(smiles, str) or not smiles.strip() or smiles.strip().lower() == "nan":
        return zeros
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return zeros
    return [len(mol.GetSubstructMatches(patterns[c])) for c in WARHEAD_COLUMNS]


def process_stem(stem_path_str: str, write_xlsx: bool = True) -> dict:
    """
    stem_path_str: path without suffix, e.g. .../5-HT1A/5-HT1A_agonist_enriched_NEW
    Loads csv if present else xlsx, adds warheads, always writes CSV; optionally xlsx.
    """
    stem = Path(stem_path_str)
    csv_path = stem.with_suffix(".csv")
    xlsx_path = stem.with_suffix(".xlsx")

    # Prefer xlsx as source of truth when both exist and csv may be stale,
    # unless csv already has more recent warhead cols — always rebuild from
    # whichever source has SMILES; prefer xlsx for descriptor fidelity.
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        had_xlsx = True
    elif csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        had_xlsx = False
    else:
        return {"stem": stem_path_str, "ok": False, "error": "no file"}

    if "SMILES" not in df.columns:
        return {"stem": stem_path_str, "ok": False, "error": "no SMILES"}

    drop_cols = [c for c in df.columns if str(c).startswith("warhead_")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    patterns = _compile_patterns()
    cache = {}
    rows = []
    for smi in df["SMILES"].astype(str):
        if smi not in cache:
            cache[smi] = _counts_for_smiles(smi, patterns)
        rows.append(cache[smi])
    wh = pd.DataFrame(rows, columns=WARHEAD_COLUMNS, index=df.index)
    out = pd.concat([df, wh], axis=1)

    out.to_csv(csv_path, index=False)
    wrote_xlsx = False
    if write_xlsx and had_xlsx:
        out.to_excel(xlsx_path, index=False)
        wrote_xlsx = True

    n_any = int((wh.sum(axis=1) > 0).sum())
    return {
        "stem": stem.name,
        "ok": True,
        "rows": len(out),
        "unique_smiles": len(cache),
        "rows_with_warhead": n_any,
        "n_warhead_cols": int(wh.shape[1]),
        "wrote_xlsx": wrote_xlsx,
    }


def _worker(args):
    stem, write_xlsx = args
    return process_stem(stem, write_xlsx=write_xlsx)


def main() -> None:
    write_xlsx = "--csv-only" not in sys.argv
    stems = {}
    for p in sorted(ROOT.glob("*/*_NEW.xlsx")) + sorted(ROOT.glob("*/*_NEW.csv")):
        stems[str(p.with_suffix(""))] = p
    stem_list = sorted(stems.keys())
    print(f"Unique NEW tables: {len(stem_list)}")
    print(f"Warhead columns: {len(WARHEAD_COLUMNS)}")
    print(f"write_xlsx={write_xlsx}")

    _compile_patterns()
    print("SMARTS OK")

    t0 = time.time()
    ok = fail = 0
    workers = 6 if not write_xlsx else 4
    # Thread pool: Excel I/O bound; avoids ProcessPool semaphore limits on some hosts
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, (s, write_xlsx)): s for s in stem_list}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception as e:
                fail += 1
                print(f"  [FAIL] {futs[fut]}: {e}", flush=True)
                continue
            if res.get("ok"):
                ok += 1
            else:
                fail += 1
                print(f"  [FAIL] {res}", flush=True)
            if done % 10 == 0 or done == len(stem_list):
                print(
                    f"  [{done}/{len(stem_list)}] ok={ok} fail={fail} "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    print(f"\nDone. ok={ok} fail={fail} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
