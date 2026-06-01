#!/usr/bin/env bash
# Full-scale fair paired H2 @ 256px — only after scale gate PASS.
# Waits for run_cycles_fair.sh to release /tmp/run_cycles.lock.
set -uo pipefail
cd "$(dirname "$0")"
source ./cycles_common.sh
wait_for_lock

LOG=cycles.log
FULL_LOG=full_fair.log

echo "########## FULL FAIR 256px QUEUE START $(date +%H:%M:%S)" | tee -a "$FULL_LOG" >> "$LOG"

# Scalability gate: 64->128 trend + cheap 256 probe; sets SCALE_FULL_* in runs/scale_gate_env.sh
if ! bash ./run_scale_gate.sh 2>&1 | tee -a "$FULL_LOG"; then
  echo "########## FULL FAIR ABORTED — scale gate failed $(date +%H:%M:%S)" | tee -a "$FULL_LOG" >> "$LOG"
  echo "See scale_gate.log and runs/scale_verdict*.json for strategies."
  exit 1
fi
# shellcheck source=/dev/null
source runs/scale_gate_env.sh

BASE_CH=${SCALE_FULL_BASE_CH:-32}
BATCH=${SCALE_FULL_BATCH:-4}
EVAL_BATCH=${SCALE_FULL_EVAL_BATCH:-4}
STEPS=${SCALE_FULL_STEPS:-12000}
echo "[full] using base_ch=$BASE_CH train_batch=$BATCH eval_batch=$EVAL_BATCH steps=$STEPS" | tee -a "$FULL_LOG"

prebuild_cache() {
  echo "########## $(date +%H:%M:%S) prebuild cache 256/3000/1000" | tee -a "$FULL_LOG"
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
  echo "########## $(date +%H:%M:%S) FULL_FAIR seed=$seed px=256 bch=$BASE_CH steps=$STEPS" \
    | tee -a "$FULL_LOG" >> "$LOG"
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

echo "########## FULL FAIR 256px COMPLETE $(date +%H:%M:%S)" | tee -a "$FULL_LOG" >> "$LOG"
python3 exps/cycles_summary.py | tee -a "$FULL_LOG"
