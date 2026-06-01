# Shared by run_cycles_wave*.sh -- source after cd to repo root.
LOG=cycles.log

# args: variant ref steps n_train wd drop noise bch [loss] [lr_sched] [size]
run_cycle() {
  local v=$1 ref=$2 steps=$3 ntr=$4 wd=$5 drop=$6 noise=$7 bch=$8
  local loss=${9:-mse} lr_sched=${10:-none} size=${11:-64}
  echo "########## $(date +%H:%M:%S) v=$v ref=$ref px=$size steps=$steps n_train=$ntr wd=$wd drop=$drop noise=$noise bch=$bch loss=$loss lr=$lr_sched" >> "$LOG"
  python3 exps/fast_cycle.py --variant "$v" --reference "$ref" --size "$size" --steps "$steps" \
      --n_train "$ntr" --weight_decay "$wd" --dropout "$drop" --input_noise "$noise" \
      --base_ch "$bch" --loss "$loss" --lr_schedule "$lr_sched" \
      2>&1 | grep -aE "RESULT|cache\]|train\]|Error|Traceback" >> "$LOG"
}

wait_for_lock() {
  exec 9>/tmp/run_cycles.lock
  echo "waiting for cycles lock..."
  while ! flock -n 9; do sleep 30; done
  echo "lock acquired"
}
