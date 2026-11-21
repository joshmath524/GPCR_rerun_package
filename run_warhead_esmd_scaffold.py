#!/usr/bin/env python3
"""
Warhead + ESMDynamic scaffold evaluation (baseline hyperparameters, no Optuna).

After:  pip install -r requirements.txt
Run:    python run_warhead_esmd_scaffold.py

Requires:
  - ML_code/warhead_scaffold_eval/cache/ligand_features.parquet  (curated warheads)
  - ESMDynamic outputs OR precomputed
    ML_code/warhead_esmd_scaffold_eval/cache/esmdynamic_receptor_features.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESMD = ROOT / "ML_code" / "warhead_esmd_scaffold_eval"
LOG = ESMD / "logs"
FEATURE_CSV = ESMD / "cache" / "esmdynamic_receptor_features.csv"


def _maybe_fix_mac_openmp() -> None:
    if sys.platform != "darwin":
        return
    candidates = []
    for site in Path(sys.prefix).glob("lib/python*/site-packages/sklearn/.dylibs"):
        candidates.append(str(site))
    for brew in ("/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib"):
        if Path(brew).is_dir():
            candidates.append(brew)
    if not candidates:
        return
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    os.environ["DYLD_LIBRARY_PATH"] = ":".join(candidates + ([existing] if existing else []))


def run_step(label: str, script: Path, extra: list[str] | None = None) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(script), *(extra or [])]
    log_path = LOG / f"{label}.log"
    print(f"\n===== START {label} =====", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"{label} failed ({rc}); see {log_path}")
    print(f"===== DONE {label} =====", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "lgb"], choices=["rf", "xgb", "lgb"])
    parser.add_argument("--rebuild-esmd-features", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    _maybe_fix_mac_openmp()

    if args.rebuild_esmd_features or not FEATURE_CSV.exists():
        run_step("build_esmd_features", ESMD / "build_esmdynamic_receptor_features.py")

    for model in args.models:
        extra = ["--model", model]
        if args.skip_shap:
            extra.append("--skip-shap")
        run_step(f"scaffold_{model}", ESMD / "scaffold_eval_warhead_esmd.py", extra)

    print("\n===== ALL DONE =====", flush=True)
    print(f"Summaries: {ESMD}/*_warhead_esmd_scaffold_mean_std.csv")
    print(f"Logs:      {LOG}/")


if __name__ == "__main__":
    main()
