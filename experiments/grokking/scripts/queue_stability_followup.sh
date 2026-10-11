#!/usr/bin/env bash
set -euo pipefail
# Arguments: predecessor directory; immutable input package; NEW output root.
# This wrapper installs its own independent outer guard, including queue wait.
if [[ "${1-}" != --inside-guard ]]; then
    exec timeout --signal=TERM --kill-after=60s 16500s bash "$0" --inside-guard "$@"
fi
shift
if [[ $# != 3 ]]; then
    echo 'Usage: queue_stability_followup.sh PREDECESSOR INPUTS NEW_OUTPUT_ROOT' >&2
    exit 2
fi
predecessor=$1
inputs=$2
root=$3
queue_start=$SECONDS
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
python_bin=${STABILITY_PYTHON:-python3}
active_guard=''
stop_queue() {
    if [[ -n "$active_guard" ]]; then
        kill -TERM "$active_guard" 2>/dev/null || true
    fi
    exit "$1"
}
trap 'stop_queue 143' TERM
trap 'stop_queue 130' INT
if [[ -e "$root" ]]; then
    echo "Refusing existing continuation output root: $root" >&2
    exit 2
fi
"$python_bin" -m src.stability_train --inputs "$inputs" --verify-inputs
"$python_bin" - "$root" <<'PY'
from pathlib import Path
import sys
from src.train import disk_guard
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
disk_guard(p.parent)
p.mkdir(exist_ok=False)
PY
while [[ ! -f "${predecessor}.exit" ]]; do
    if (( SECONDS-queue_start >= 5400 )); then
        echo 'Predecessor wait budget exhausted; no continuation launched' >&2
        exit 75
    fi
    sleep 30
done
if (( SECONDS-queue_start >= 5400 )); then
    echo 'Predecessor wait deadline passed; refusing launch' >&2
    exit 75
fi
predecessor_exit=$(cat "${predecessor}.exit")
if [[ "$predecessor_exit" != 0 || ! -f "$predecessor/completion.json" || -f "$predecessor/failure.json" ]]; then
    echo 'Predecessor did not exit cleanly; refusing continuation' >&2
    exit 76
fi
"$python_bin" - "$predecessor/completion.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
if c.get('state') not in ('HORIZON_COMPLETE','BUDGET_STOP') or c.get('signal_received',False):
    raise SystemExit('Predecessor completion is not clean')
PY
for arm in baseline confidence stability; do
    out="$root/$arm"
    if [[ -e "$out" || -e "${out}.log" || -e "${out}.exit" ]]; then
        echo "Refusing existing arm output: $out" >&2
        exit 2
    fi
    deadline=$(( $(date +%s)+3540 ))
    printf 'Starting arm=%s at %s; deadline=%s\n' "$arm" "$(date -u +%FT%TZ)" "$deadline"
    set +e
    timeout --signal=TERM --kill-after=60s 3600s "$python_bin" -u -m src.stability_train \
        --inputs "$inputs" --arm "$arm" --output "$out" --steps 200 \
        --seconds 3540 --deadline-unix "$deadline" --threads 8 > "${out}.log" 2>&1 &
    active_guard=$!
    wait "$active_guard"
    result=$?
    active_guard=''
    set -e
    printf '%s\n' "$result" > "${out}.exit"
    if [[ "$result" != 0 && "$result" != 124 ]]; then
        echo "Continuation arm failed: $arm (exit $result); no later arm launched" >&2
        exit "$result"
    fi
    "$python_bin" - "$out/completion.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
if c.get('state') not in ('HORIZON_COMPLETE','BUDGET_STOP'):
    raise SystemExit('Missing clean arm completion; no later arm launched')
PY
done
