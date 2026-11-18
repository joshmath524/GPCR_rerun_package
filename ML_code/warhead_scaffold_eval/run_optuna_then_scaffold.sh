#!/usr/bin/env bash
# Thin wrapper — preferred entrypoint is: python run_warhead_pipeline.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
N_TRIALS="${1:-100}"
exec python "$ROOT/run_warhead_pipeline.py" --n-trials "$N_TRIALS"
