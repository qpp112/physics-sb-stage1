#!/usr/bin/env bash
#
# Sigma-sweep probe: does raising the bridge noise fix the x_t-leak that makes
# the trained sampler underperform the static prior? (See prereg.md CORRECTION 3.)
#
# Trains physics at sigma in {0.6, 0.3, 0.1} for a reduced 20k-step budget, then
# runs the rollout diagnostic (free vs teacher vs static) on each. Highest sigma
# first so the strongest signal lands earliest. Retry/resume on crash.
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"

latest_ckpt() { ls -t "$1"/ckpt_*.pt 2>/dev/null | head -1; }

run_with_retry() {
  local cfg="$1" name="$2" tries=0
  while :; do
    local ck; ck="$(latest_ckpt "runs/$name")"
    local args=(--config "$cfg" --seed 0 --run_name "$name")
    [ -n "$ck" ] && args+=(--resume "$ck")
    echo "=== $name  try=$((tries+1))  resume='${ck:-none}' ==="
    if python3 exps/1b_train.py "${args[@]}"; then return 0; fi
    tries=$((tries+1)); [ "$tries" -ge 5 ] && { echo "FAIL: $name"; exit 1; }
    echo "=== $name crashed; retry in 10s ==="; sleep 10
  done
}

for arm in "configs/physics_sig06.yaml physics_sig06" \
           "configs/physics_sig03.yaml physics_sig03" \
           "configs/physics_sig01.yaml physics_sig01"; do
  set -- $arm
  run_with_retry "$1" "$2"
  echo "=== rollout diagnostic: $2 ==="
  python3 exps/diag_rollout.py "$2" 64 2>&1 | grep -vE "Warning|warn|LPIPS|Loading|Setting up"
done

echo "=== SIGMA SWEEP COMPLETE ==="
