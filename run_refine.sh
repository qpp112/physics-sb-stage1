#!/usr/bin/env bash
#
# Midpoint-refinement arm (exps/1c_refine.py): physics_refine + brownian_refine.
# Waits for the sigma sweep to finish first so the two jobs never share the GPU.
# Retry/resume on crash; each run self-evaluates (trained vs static) at the end.
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"

SWEEP_PID="${1:-}"   # optional: pid of run_sigsweep.sh to wait on

if [ -n "$SWEEP_PID" ]; then
  echo "=== waiting for sigma sweep (pid $SWEEP_PID) to finish ==="
  while kill -0 "$SWEEP_PID" 2>/dev/null; do sleep 60; done
  echo "=== sweep finished; starting refinement arm ==="
fi

latest_ckpt() { ls -t "$1"/ckpt_*.pt 2>/dev/null | head -1; }

run_with_retry() {
  local cfg="$1" name="$2" tries=0
  while :; do
    local ck; ck="$(latest_ckpt "runs/$name")"
    local args=(--config "$cfg" --seed 0 --run_name "$name")
    [ -n "$ck" ] && args+=(--resume "$ck")
    echo "=== $name  try=$((tries+1))  resume='${ck:-none}' ==="
    if python3 exps/1c_refine.py "${args[@]}"; then return 0; fi
    tries=$((tries+1)); [ "$tries" -ge 5 ] && { echo "FAIL: $name"; exit 1; }
    echo "=== $name crashed; retry in 10s ==="; sleep 10
  done
}

run_with_retry configs/physics_refine.yaml  physics_refine
run_with_retry configs/brownian_refine.yaml brownian_refine
echo "=== REFINEMENT ARM COMPLETE ==="
