#!/usr/bin/env bash
# WAVE 4 -- push sweet spot: best reg + loss + schedule + steps + H2 head-to-head.
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock

# Best base: refine_resid physics n_train=3000 wd=0.1 drop=0.2
#    v            ref       steps ntr   wd   drop noise bch  loss   lr
run_cycle refine_resid physics  6000  3000  0.10 0.2  0.0  32  mse    cosine
run_cycle refine_resid physics  6000  3000  0.10 0.2  0.0  32  l1     cosine
run_cycle refine_resid physics  6000  3000  0.10 0.2  0.0  32  charb  cosine
run_cycle refine_resid physics  6000  6000  0.10 0.2  0.0  32  charb  cosine
run_cycle refine_resid physics  6000  3000  0.10 0.2  0.10 32  charb  cosine
run_cycle refine_resid physics  10000 3000  0.10 0.2  0.0  32  charb  cosine
run_cycle refine_resid physics  6000  3000  0.15 0.3  0.0  32  charb  cosine
run_cycle refine_resid brownian 6000  3000  0.10 0.2  0.0  32  charb  cosine
run_cycle refine_resid physics  6000  3000  0.10 0.2  0.0  48  charb  cosine
run_cycle refine_resid physics  6000  3000  0.10 0.2  0.0  64  charb  cosine
echo "########## CYCLES WAVE 4 COMPLETE $(date +%H:%M:%S)" >> "$LOG"
