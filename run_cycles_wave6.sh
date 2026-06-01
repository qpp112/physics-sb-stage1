#!/usr/bin/env bash
# WAVE 6 -- push physics @ 128px (current best scale: trained ~21.17 dB).
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock
echo "########## WAVE 6 START $(date +%H:%M:%S) -- push 128px physics" >> "$LOG"

# Best so far: 128 charb cosine 8k n_train=6000 -> 21.168 dB
#    variant       ref      steps ntr   wd   drop noise bch  loss   lr      size
run_cycle refine_resid physics 12000 6000 0.10 0.2 0.0  32 charb cosine 128
run_cycle refine_resid physics  8000 6000 0.10 0.2 0.10 32 charb cosine 128
run_cycle refine_resid physics  8000 6000 0.15 0.3 0.0  32 charb cosine 128
run_cycle refine_resid physics  8000 6000 0.10 0.2 0.0  48 charb cosine 128
run_cycle refine_resid physics  8000 6000 0.10 0.2 0.0  32 l1    cosine 128
run_cycle refine_resid physics 10000 6000 0.10 0.2 0.0  32 charb cosine 128
run_cycle refine_resid physics  8000 3000 0.10 0.2 0.10 32 charb cosine 128
run_cycle refine_resid physics  8000 6000 0.10 0.1 0.0  32 charb cosine 128

echo "########## WAVE 6 COMPLETE $(date +%H:%M:%S)" >> "$LOG"
python3 exps/cycles_summary.py >> "$LOG" 2>&1
