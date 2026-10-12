#!/usr/bin/env bash
set -euo pipefail
# INPUT_PACKAGE NEW_OUTPUT_ROOT WORLD RANK MASTER_ADDR MASTER_PORT
if [[ $# != 6 ]]; then
    echo 'Usage: guard_followup_scale.sh INPUTS OUTPUT WORLD RANK MASTER_ADDR MASTER_PORT' >&2
    exit 2
fi
inputs=$1
output=$2
world=$3
rank=$4
master=$5
port=$6
if [[ -e "$output/rank$rank" || -e "${output}.rank${rank}.log" || -e "${output}.rank${rank}.exit" ]]; then
    echo 'Refusing an existing scale-out run' >&2
    exit 2
fi
mkdir -p "$(dirname "$output")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export GLOO_SOCKET_IFNAME=ens5
export WORLD_SIZE="$world" RANK="$rank" MASTER_ADDR="$master" MASTER_PORT="$port"
python_bin=${SCALE_PYTHON:-python3}
remaining=$("$python_bin" - "$inputs" <<'PY'
import json, math, sys, time
from pathlib import Path
from src.train import disk_guard
p=Path(sys.argv[1]); disk_guard(p)
c=json.loads((p/'config.json').read_text()); m=json.loads((p/'manifest.json').read_text())
d=float(m.get('original_external_deadline',c['start_unix']+72000))
assert math.isfinite(d) and d<=c['start_unix']+72000
remaining=math.floor(d-time.time())
assert remaining>180,'original external deadline exhausted'
print(remaining)
PY
)
set +e
timeout --signal=TERM --kill-after=180s "${remaining}s" \
    taskset -c 0-7 "$python_bin" -u -m src.followup_scale \
    --inputs "$inputs" --output "$output" > "${output}.rank${rank}.log" 2>&1 &
guard_pid=$!
trap 'kill -TERM "$guard_pid" 2>/dev/null || true' TERM INT
wait "$guard_pid"
result=$?
printf '%s\n' "$result" > "${output}.rank${rank}.exit"
exit "$result"
