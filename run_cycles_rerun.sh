#!/usr/bin/env bash
# Re-run waves 3-5 after fast_cycle.py main() bug fix (empty main body).
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock

echo "########## RERUN START $(date +%H:%M:%S) -- fixed harness" >> "$LOG"

# --- wave 3 (sweet spot probes) ---
run_cycle refine_resid physics  3000  3000  0.10  0.2  0.10  32  mse    none
run_cycle refine_resid physics  6000  3000  0.10  0.2  0.0   32  mse    none
run_cycle refine_resid physics  3000  6000  0.10  0.2  0.0   32  mse    none
run_cycle refine_resid brownian 3000  3000  0.10  0.2  0.0   32  mse    none
run_cycle refine        physics 3000  3000  0.10  0.2  0.0   32  mse    none

# --- wave 4 (push dominance) ---
run_cycle refine_resid physics  6000  3000  0.10  0.2  0.0   32  charb  cosine
run_cycle refine_resid physics  6000  3000  0.10  0.2  0.0   32  l1     cosine
run_cycle refine_resid physics  6000  6000  0.10  0.2  0.0   32  charb  cosine
run_cycle refine_resid physics  10000 3000  0.10  0.2  0.0   32  charb  cosine
run_cycle refine_resid brownian 6000  3000  0.10  0.2  0.0   32  charb  cosine
run_cycle refine_resid physics  6000  3000  0.10  0.2  0.0   48  charb  cosine

# --- wave 5 (128px scale) ---
run_cycle_128() {
  local ref=$1 steps=$2 ntr=$3 wd=$4 drop=$5 bch=$6 loss=$7
  echo "########## $(date +%H:%M:%S) 128px ref=$ref steps=$steps n_train=$ntr" >> "$LOG"
  python3 exps/fast_cycle.py --variant refine_resid --reference "$ref" --size 128 \
      --steps "$steps" --n_train "$ntr" --weight_decay "$wd" --dropout "$drop" \
      --base_ch "$bch" --loss "$loss" --lr_schedule cosine \
      2>&1 | grep -aE "RESULT|cache\]|train\]|Error|Traceback" >> "$LOG"
}
run_cycle_128 physics 8000  3000 0.10 0.2 32 charb
run_cycle_128 physics 8000  6000 0.10 0.2 32 charb
run_cycle_128 brownian 8000 3000 0.10 0.2 32 charb

echo "########## RERUN COMPLETE $(date +%H:%M:%S)" >> "$LOG"
