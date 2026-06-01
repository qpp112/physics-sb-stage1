#!/usr/bin/env bash
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
run_with_retry configs/physics.yaml physics_s0 0
run_with_retry configs/physics.yaml physics_s1 1
python3 exps/1b_eval.py --config configs/base.yaml \
    --runs runs/brownian_s0 runs/brownian_s1 runs/physics_s0 runs/physics_s1 \
    --out  runs/1b_eval/results.json
echo "=== PILOT COMPLETE -> runs/1b_eval/results.json ==="
