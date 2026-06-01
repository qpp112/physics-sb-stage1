#!/usr/bin/env bash
# resume_pilot.sh — pick up the Stage-1 pilot after the brownian_s0 crash.
# Skips sanity/1a/precompute (already done). Resumes each training run from its
# latest checkpoint, and AUTO-RETRIES from the latest checkpoint if a run dies
# again (the libc++abi native abort), so you are not exposed to another blind
# 10-hour babysit. Then runs the final eval.
#
# Run inside tmux:  tmux new -s pilot ;  ./resume_pilot.sh 2>&1 | tee -a ~/run.log

set -uo pipefail   # NOTE: no -e, so a single crash does not kill the whole driver
REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"

latest_ckpt() { ls -t "$1"/ckpt_*.pt 2>/dev/null | head -1; }

run_with_retry() {            # <config> <run_name> <seed>
  local cfg="$1" name="$2" seed="$3" tries=0
  while :; do
    local ck; ck="$(latest_ckpt "runs/$name")"
    local args=(--config "$cfg" --seed "$seed" --run_name "$name")
    [ -n "$ck" ] && args+=(--resume "$ck")
    echo "=== $name  try=$((tries+1))  resume='${ck:-none}' ==="
    if python3 exps/1b_train.py "${args[@]}"; then
      echo "=== $name DONE ==="; return 0
    fi
    tries=$((tries+1))
    [ "$tries" -ge 5 ] && { echo "FAIL: $name died 5x, stopping."; exit 1; }
    echo "=== $name crashed; retrying from latest checkpoint in 10s ==="; sleep 10
  done
}

run_with_retry configs/brownian.yaml brownian_s0 0   # resumes from ckpt_0060000.pt
run_with_retry configs/brownian.yaml brownian_s1 1
run_with_retry configs/physics.yaml  physics_s0  0
run_with_retry configs/physics.yaml  physics_s1  1

echo "=== experiment 1b: evaluation ==="
python3 exps/1b_eval.py --config configs/base.yaml \
    --runs runs/brownian_s0 runs/brownian_s1 runs/physics_s0 runs/physics_s1 \
    --out  runs/1b_eval/results.json
echo "=== PILOT COMPLETE -> runs/1b_eval/results.json ==="
