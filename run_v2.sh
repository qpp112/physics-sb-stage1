#!/usr/bin/env bash
#
# v2 retraining driver: brownian_v2_s{0,1}, physics_v2_s{0,1} (condition_on_reference=true),
# then stratified eval. Retry/resume on crash, like physics_only.sh.
#
# v1 runs (brownian_s*, physics_s*) are left untouched on disk for comparison.
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
    if python3 exps/1b_train.py "${args[@]}"; then return 0; fi
    tries=$((tries+1)); [ "$tries" -ge 5 ] && { echo "FAIL: $name"; exit 1; }
    echo "=== $name crashed; retry in 10s ==="; sleep 10
  done
}

# physics first so the early-signal val shows up fastest, then the rest.
run_with_retry configs/physics.yaml  physics_v2_s0  0
run_with_retry configs/brownian.yaml brownian_v2_s0 0
run_with_retry configs/physics.yaml  physics_v2_s1  1
run_with_retry configs/brownian.yaml brownian_v2_s1 1

echo "=== v2 evaluation ==="
python3 exps/1b_eval.py --config configs/base.yaml \
    --runs runs/brownian_v2_s0 runs/brownian_v2_s1 runs/physics_v2_s0 runs/physics_v2_s1 \
    --out  runs/1b_eval_v2/results.json
echo "=== V2 PILOT COMPLETE -> runs/1b_eval_v2/results.json ==="
