#!/usr/bin/env bash
# Fair H2 @ 128px: physics vs brownian, matched sweet-spot config, scored vs GT only.
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock

echo "########## H2 FAIR 128px START $(date +%H:%M:%S)" >> "$LOG"

# Sweet spot: refine_resid, wd=0.1, drop=0.2, charb+cosine, 8k steps, n_train=3000
run_cycle refine_resid physics  8000 3000 0.10 0.2 0.0 32 charb cosine 128
run_cycle refine_resid brownian 8000 3000 0.10 0.2 0.0 32 charb cosine 128
# More data for physics (best prior run used 6000)
run_cycle refine_resid physics  8000 6000 0.10 0.2 0.0 32 charb cosine 128
run_cycle refine_resid brownian 8000 6000 0.10 0.2 0.0 32 charb cosine 128
# Input-noise reg (physics winner at 64px)
run_cycle refine_resid physics  8000 3000 0.10 0.2 0.10 32 charb cosine 128

echo "########## H2 FAIR 128px COMPLETE $(date +%H:%M:%S)" >> "$LOG"
python3 exps/cycles_summary.py >> "$LOG" 2>&1
