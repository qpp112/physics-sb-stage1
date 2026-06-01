#!/usr/bin/env bash
# Fail-fast experiment cycles. Each line: "variant reference steps n_train noise base_ch".
# Results append to cycles.log as one-line RESULT entries. Backgrounded; never blocks.
set -uo pipefail
cd "$(dirname "$0")"
# Single-instance lock: if a driver is already running, exit immediately. Protects
# against duplicate/replayed launches.
exec 9>/tmp/run_cycles.lock
if ! flock -n 9; then
  echo "another run_cycles.sh holds the lock; exiting" >&2
  exit 0
fi
LOG=cycles.log

# args: variant ref steps n_train wd dropout noise base_ch
run() {
  local v=$1 ref=$2 steps=$3 ntr=$4 wd=$5 drop=$6 noise=$7 bch=$8
  echo "########## $(date +%H:%M:%S) v=$v ref=$ref steps=$steps n_train=$ntr wd=$wd drop=$drop noise=$noise bch=$bch" >> "$LOG"
  python3 exps/fast_cycle.py --variant "$v" --reference "$ref" --steps "$steps" \
      --n_train "$ntr" --weight_decay "$wd" --dropout "$drop" --input_noise "$noise" \
      --base_ch "$bch" 2>&1 | grep -aE "RESULT|cache\]|train\]|Error|Traceback" >> "$LOG"
}

# WAVE 2 -- attack overfitting (data / regularization). Fast 3k-step budgets.
#    variant       ref       steps n_train  wd    drop  noise bch
run refine_resid  physics    3000  800      0.01  0.0   0.0   32   # baseline (fast harness)
run refine_resid  physics    3000  3000     0.01  0.0   0.0   32   # 4x data
run refine_resid  physics    3000  6000     0.01  0.0   0.0   32   # 7.5x data
run refine_resid  physics    3000  3000     0.10  0.2   0.0   32   # data + strong reg
run refine_resid  physics    3000  3000     0.01  0.0   0.10  32   # data + input-noise reg
run refine_resid  physics    3000  3000     0.01  0.0   0.0   16   # smaller model
run refine_resid  brownian   3000  3000     0.10  0.2   0.0   32   # reference comparison
echo "########## CYCLES WAVE 2 COMPLETE $(date +%H:%M:%S)" >> "$LOG"
