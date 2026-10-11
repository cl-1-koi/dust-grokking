#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# The Python runner also checks a shorter internal deadline and refuses
# checkpoints below 5 GiB free disk. Always choose a new output directory.
exec timeout --signal=TERM --kill-after=60s 72000s \
  python3 -u -m src.reproduce_seed "$@"
