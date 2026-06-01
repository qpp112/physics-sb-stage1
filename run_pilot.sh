#!/usr/bin/env bash
#
# Stage-1 pilot driver (refined). Runs:
#   sanity gates -> 1a static prior -> precompute -> 4 training runs -> eval
#
# Before running:
#   1. edit configs/base.yaml -> data.root
#   2. pip install -r requirements.txt
#   3. run inside tmux (this is long): tmux new -s pilot
#
# Storage/RAM: precompute shrinks the working set to ~4-6 GB so it fits in RAM and
# the GPU is not starved. Without it, expect ~10x slower training.

set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"
mkdir -p runs

echo "=== sanity gates (pre-training) ==="
python3 exps/sanity_checks.py --config configs/base.yaml

echo "=== experiment 1a: static prior test ==="
python3 exps/1a_prior_eval.py --config configs/base.yaml --out runs/1a_prior_eval/results.json

echo "=== precompute compact tensor cache (one-time, GPU-bound training after) ==="
python3 precompute.py --config configs/base.yaml

echo "=== experiment 1b: training (4 runs) ==="
for seed in 0 1; do
  python3 exps/1b_train.py --config configs/brownian.yaml --seed $seed --run_name brownian_s${seed}
done
for seed in 0 1; do
  python3 exps/1b_train.py --config configs/physics.yaml  --seed $seed --run_name physics_s${seed}
done

echo "=== experiment 1b: evaluation ==="
python3 exps/1b_eval.py --config configs/base.yaml \
    --runs runs/brownian_s0 runs/brownian_s1 runs/physics_s0 runs/physics_s1 \
    --out  runs/1b_eval/results.json

echo
echo "Pilot complete."
echo "  H1: runs/1a_prior_eval/results.json (H1.overall_pass)"
echo "  H2: runs/1b_eval/results.json (__pilot_decision.verdict, __per_stratum)"
