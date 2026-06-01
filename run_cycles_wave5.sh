#!/usr/bin/env bash
# WAVE 5 -- scale winners: 128px + longer train on top configs from wave 4.
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock

run_cycle_128() {
  local ref=$1 steps=$2 ntr=$3 wd=$4 drop=$5 noise=$6 bch=$7 loss=$8
  echo "########## $(date +%H:%M:%S) 128px ref=$ref steps=$steps n_train=$ntr" >> "$LOG"
  python3 exps/fast_cycle.py --variant refine_resid --reference "$ref" --size 128 \
      --steps "$steps" --n_train "$ntr" --weight_decay "$wd" --dropout "$drop" \
      --input_noise "$noise" --base_ch "$bch" --loss "$loss" --lr_schedule cosine \
      2>&1 | grep -aE "RESULT|cache\]|train\]|Error|Traceback" >> "$LOG"
}

run_cycle_128 physics 8000  3000 0.10 0.2 0.0  32 charb
run_cycle_128 physics 8000  6000 0.10 0.2 0.0  32 charb
run_cycle_128 physics 12000 3000 0.10 0.2 0.0  48 charb
run_cycle_128 brownian 8000 3000 0.10 0.2 0.0  32 charb
run_cycle_128 physics 8000  3000 0.10 0.2 0.10 32 charb
echo "########## CYCLES WAVE 5 COMPLETE $(date +%H:%M:%S)" >> "$LOG"
