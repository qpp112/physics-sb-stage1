#!/usr/bin/env bash
# WAVE 3 -- probe around sweet spot (refine_resid + reg + data).
# Chains after run_cycles.sh: waits for lock, then runs ~8 fast cycles.
set -uo pipefail
cd "$(dirname "$0")"
exec 9>/tmp/run_cycles.lock
echo "waiting for wave 2 lock..."
while ! flock -n 9; do sleep 30; done
echo "lock acquired, starting wave 3"

LOG=cycles.log
run() {
  local v=$1 ref=$2 steps=$3 ntr=$4 wd=$5 drop=$6 noise=$7 bch=$8
  echo "########## $(date +%H:%M:%S) v=$v ref=$ref steps=$steps n_train=$ntr wd=$wd drop=$drop noise=$noise bch=$bch" >> "$LOG"
  python3 exps/fast_cycle.py --variant "$v" --reference "$ref" --steps "$steps" \
      --n_train "$ntr" --weight_decay "$wd" --dropout "$drop" --input_noise "$noise" \
      --base_ch "$bch" 2>&1 | grep -aE "RESULT|cache\]|train\]|Error|Traceback" >> "$LOG"
}

# Sweet spot from wave 2: refine_resid, n_train>=3000, wd~0.1, drop~0.2 -> +0.4 to +0.7 dB
#    variant       ref       steps n_train  wd    drop  noise bch
run refine_resid  physics    3000  3000     0.10  0.2   0.10  32   # combine reg + input noise
run refine_resid  physics    6000  3000     0.10  0.2   0.0   32   # 2x train steps at best reg
run refine_resid  physics    3000  6000     0.10  0.2   0.0   32   # best reg + max data
run refine_resid  physics    3000  3000     0.05  0.1   0.0   32   # lighter reg (interp)
run refine_resid  physics    3000  3000     0.10  0.2   0.0   48   # slightly larger model
run refine_resid  brownian   3000  3000     0.10  0.2   0.0   32   # H2: does physics beat brownian?
run refine        physics    3000  3000     0.10  0.2   0.0   32   # non-resid at sweet spot (expect lose)
run refine_resid  physics    1500  3000     0.10  0.2   0.0   32   # half steps (sample efficiency)
echo "########## CYCLES WAVE 3 COMPLETE $(date +%H:%M:%S)" >> "$LOG"
