#!/usr/bin/env bash
# Resume after OOM fix: re-run scale gate (256 probe only) then full 256px if PASS.
# Skips wait_for_lock — fair queue already finished.
set -uo pipefail
cd "$(dirname "$0")"

LOG=cycles.log
FULL_LOG=full_fair.log

echo "########## RESUME FULL FAIR $(date +%H:%M:%S)" | tee -a "$FULL_LOG" >> "$LOG"
echo "OOM fix: fair_pair eval_batch=4 @ 256px (was hardcoded 64)" | tee -a "$FULL_LOG"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! bash ./run_scale_gate.sh 2>&1 | tee -a "$FULL_LOG"; then
  echo "########## RESUME ABORTED — scale gate failed $(date +%H:%M:%S)" | tee -a "$FULL_LOG" >> "$LOG"
  exit 1
fi

source runs/scale_gate_env.sh
BASE_CH=${SCALE_FULL_BASE_CH:-32}
BATCH=${SCALE_FULL_BATCH:-4}
EVAL_BATCH=${SCALE_FULL_EVAL_BATCH:-4}
STEPS=${SCALE_FULL_STEPS:-12000}

prebuild_cache() {
  python3 - <<'PY' 2>&1 | tee -a "$FULL_LOG"
import time
from exps.fast_cycle import build_cache
t0 = time.time()
build_cache(256, 3000, 1000, 40.0,
            "/home/ubuntu/datasets/FlyingThings3D", "cleanpass")
print(f"[prebuild] done in {time.time()-t0:.0f}s")
PY
}

run_full_fair() {
  local seed=$1
  echo "########## FULL_FAIR seed=$seed px=256 bch=$BASE_CH batch=$BATCH" | tee -a "$FULL_LOG" >> "$LOG"
  python3 exps/fair_pair_full.py \
    --size 256 --steps "$STEPS" --n_train 3000 --n_test 1000 \
    --weight_decay 0.10 --dropout 0.2 --loss charb --lr_schedule cosine \
    --input_noise 0.0 --batch "$BATCH" --eval_batch "$EVAL_BATCH" --base_ch "$BASE_CH" \
    --seed "$seed" \
    2>&1 | tee -a "$FULL_LOG" | grep -aE "FULL_PAIRED|PAIRED_RESULT|prereg|Error|Traceback|\[full\]" >> "$LOG"
}

prebuild_cache
run_full_fair 0
run_full_fair 1
echo "########## FULL FAIR COMPLETE $(date +%H:%M:%S)" | tee -a "$FULL_LOG" >> "$LOG"
python3 exps/cycles_summary.py | tee -a "$FULL_LOG"
