#!/usr/bin/env bash
# Quick remediation probes when 128px gate fails (before full plan).
set -uo pipefail
cd "$(dirname "$0")"
LOG=cycles.log

run_fair() {
  local size=$1 steps=$2 ntr=$3 bch=$4 loss=$5 lr=$6
  echo "########## SCALE_FIX $(date +%H:%M:%S) px=$size steps=$steps bch=$bch" >> "$LOG"
  python3 exps/fair_pair.py --size "$size" --steps "$steps" --n_train "$ntr" \
      --weight_decay 0.10 --dropout 0.2 --loss "$loss" --lr_schedule "$lr" \
      --base_ch "$bch" --batch 16 \
      2>&1 | grep -aE "PAIRED_RESULT|Error|Traceback" >> "$LOG"
}

# Extra 128px fair @ smaller model width (faster) or longer steps if queue skipped 12k
run_fair 128 12000 6000 32 charb cosine
# Apples-to-apples 128 vs 64: same n_train=3000
run_fair 128 8000 3000 32 charb cosine
