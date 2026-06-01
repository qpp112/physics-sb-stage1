#!/usr/bin/env bash
# Fair paired H2 cycles -- only difference is reference path.
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock

LOG=cycles.log
echo "########## FAIR PAIRED CYCLES START $(date +%H:%M:%S)" >> "$LOG"

run_fair() {
  local size=$1 steps=$2 ntr=$3 wd=$4 drop=$5 loss=$6 lr=$7 noise=${8:-0.0}
  echo "########## $(date +%H:%M:%S) FAIR_PAIR px=$size steps=$steps n_train=$ntr" >> "$LOG"
  python3 exps/fair_pair.py --size "$size" --steps "$steps" --n_train "$ntr" \
      --weight_decay "$wd" --dropout "$drop" --loss "$loss" --lr_schedule "$lr" \
      --input_noise "$noise" \
      2>&1 | grep -aE "PAIRED_RESULT|FAIR|fair|Error|Traceback" >> "$LOG"
}

# Matched configs at 64 and 128
run_fair 64  3000 3000 0.10 0.2 mse   none   0.0
run_fair 64  3000 3000 0.10 0.2 mse   none   0.10
run_fair 128 8000 6000 0.10 0.2 charb cosine 0.0
run_fair 128 8000 6000 0.10 0.2 charb cosine 0.10
run_fair 128 12000 6000 0.10 0.2 charb cosine 0.0

echo "########## FAIR PAIRED CYCLES COMPLETE $(date +%H:%M:%S)" >> "$LOG"
