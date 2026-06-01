#!/usr/bin/env bash
# WAVE 7 -- scale best recipe to 256px (full precompute res); physics only.
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock
echo "########## WAVE 7 START $(date +%H:%M:%S) -- 256px physics scale-up" >> "$LOG"

run_cycle refine_resid physics  6000 3000 0.10 0.2 0.0 32 charb cosine 256
run_cycle refine_resid physics  8000 6000 0.10 0.2 0.0 32 charb cosine 256
run_cycle refine_resid physics  8000 3000 0.10 0.2 0.10 32 charb cosine 256
run_cycle refine_resid physics  8000 6000 0.10 0.2 0.10 32 charb cosine 256

echo "########## WAVE 7 COMPLETE $(date +%H:%M:%S)" >> "$LOG"
python3 exps/cycles_summary.py >> "$LOG" 2>&1
