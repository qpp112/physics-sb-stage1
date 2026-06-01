#!/usr/bin/env bash
#
# v3c retraining driver: v3 conditioning (u_ref + ref_mean) PLUS an occlusion /
# warp-reliability map (in_channels=16). brownian_v3c_s{0,1}, physics_v3c_s{0,1},
# then eval. Retry/resume on crash.
#
# Reuses exps/1b_train_v3.py + exps/1b_eval_v3.py (occlusion gated by
# bridge.condition_on_occlusion). Leaves v1 / v2 / v3 / v3b runs untouched.
# DO NOT launch while another pilot is using the GPU.
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"

latest_ckpt() { ls -t "$1"/ckpt_*.pt 2>/dev/null | head -1; }

run_with_retry() {
  local cfg="$1" name="$2" seed="$3" tries=0
  while :; do
    local ck; ck="$(latest_ckpt "runs/$name")"
    local args=(--config "$cfg" --seed "$seed" --run_name "$name")
    [ -n "$ck" ] && args+=(--resume "$ck")
    echo "=== $name  try=$((tries+1))  resume='${ck:-none}' ==="
    if python3 exps/1b_train_v3.py "${args[@]}"; then return 0; fi
    tries=$((tries+1)); [ "$tries" -ge 5 ] && { echo "FAIL: $name"; exit 1; }
    echo "=== $name crashed; retry in 10s ==="; sleep 10
  done
}

run_with_retry configs/physics_v3c.yaml  physics_v3c_s0  0
run_with_retry configs/brownian_v3c.yaml brownian_v3c_s0 0
run_with_retry configs/physics_v3c.yaml  physics_v3c_s1  1
run_with_retry configs/brownian_v3c.yaml brownian_v3c_s1 1

echo "=== v3c evaluation ==="
python3 exps/1b_eval_v3.py --config configs/base.yaml \
    --runs runs/brownian_v3c_s0 runs/brownian_v3c_s1 runs/physics_v3c_s0 runs/physics_v3c_s1 \
    --out  runs/1b_eval_v3c/results.json
echo "=== V3C PILOT COMPLETE -> runs/1b_eval_v3c/results.json ==="
