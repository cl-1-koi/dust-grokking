#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-runs/prediction-stability-20261011}
if [[ -e "$out" || -e "${out}.log" || -e "${out}.exit" ]]; then
    echo 'Refusing existing stability diagnostic output' >&2
    exit 2
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
.venv/bin/python - "$(dirname "$out")" <<'PY'
import shutil,sys
assert shutil.disk_usage(sys.argv[1]).free >= 5*1024**3,'free disk below5GiB'
PY
set +e
timeout --signal=TERM --kill-after=30s 300s .venv/bin/python -u -m src.prediction_stability \
    --output "$out" > "${out}.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "${out}.exit"
exit "$result"
