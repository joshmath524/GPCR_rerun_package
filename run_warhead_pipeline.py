#!/usr/bin/env python3
"""
One-command curated-warhead Optuna + scaffold pipeline.

After:  pip install -r requirements.txt
Run:    python run_warhead_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ML_code" / "warhead_scaffold_eval"
OPTUNA = OUT / "optuna"
CACHE = OUT / "cache"
LOG = OUT / "logs"


def _maybe_fix_mac_openmp() -> None:
    """Help LightGBM/sklearn find OpenMP on macOS Homebrew / sklearn bundles."""
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


def run_step(label: str, script: Path, extra_args: list[str] | None = None) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(script), *(extra_args or [])]
    log_path = LOG / f"{label}.log"
    print(f"\n===== START {label} =====", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    print(f"  log: {log_path}", flush=True)
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
        raise SystemExit(f"{label} failed with exit code {rc} (see {log_path})")
    print(f"===== DONE {label} =====", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna retune + curated warhead scaffold evals")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per model (default 100)")
    parser.add_argument("--skip-optuna", action="store_true", help="Only run scaffold evals")
    parser.add_argument("--skip-scaffold", action="store_true", help="Only run Optuna")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild curated warhead parquet cache before Optuna/scaffold",
    )
    args = parser.parse_args()

    _maybe_fix_mac_openmp()

    if not (CACHE / "ligand_features.parquet").exists():
        print("Cache missing; rebuilding curated warhead feature cache...", flush=True)
        run_step("rebuild_curated_cache", OUT / "rebuild_curated_warhead_cache.py")
    elif args.rebuild_cache:
        run_step("rebuild_curated_cache", OUT / "rebuild_curated_warhead_cache.py")

    if not args.skip_optuna:
        print(f"\n===== PHASE 1: OPTUNA ({args.n_trials} trials each) =====", flush=True)
        run_step("optuna_rf", OPTUNA / "rf_warhead_optuna.py", ["--n-trials", str(args.n_trials)])
        run_step("optuna_xgb", OPTUNA / "xgb_warhead_optuna.py", ["--n-trials", str(args.n_trials)])
        run_step("optuna_lightgbm", OPTUNA / "lightgbm_warhead_optuna.py", ["--n-trials", str(args.n_trials)])

    if not args.skip_scaffold:
        print("\n===== PHASE 2: SCAFFOLD EVALS =====", flush=True)
        run_step("scaffold_rf", OUT / "rf_scaffold_eval_warhead.py")
        run_step("scaffold_xgb", OUT / "xgb_scaffold_eval_warhead.py")
        run_step("scaffold_lightgbm", OUT / "lightgbm_scaffold_eval_warhead.py")

    print("\n===== ALL DONE =====", flush=True)
    print(f"  Optuna best params: {OPTUNA}/*_best_params.json")
    print(f"  Scaffold summaries:  {OUT}/*_scaffold*_mean_std.csv")
    print(f"  Logs:                {LOG}/")


if __name__ == "__main__":
    main()
