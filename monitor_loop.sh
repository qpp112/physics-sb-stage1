#!/usr/bin/env bash
# Background monitor — appends status every 5 min until pipeline done.
set -uo pipefail
cd "$(dirname "$0")"
LOG=monitor.log
INTERVAL=300

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== monitor started ==="

while true; do
  {
    echo ""
    echo "======== $(date '+%Y-%m-%d %H:%M:%S') ========"
    echo "--- processes ---"
    pgrep -af 'run_cycles_fair|run_full_fair|fair_pair|scale_verify|fair_pair_full' 2>/dev/null || echo "(none)"
    echo "--- gpu ---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi n/a"
    echo "--- cycles.log tail ---"
    tail -8 cycles.log 2>/dev/null || true
    echo "--- PAIRED / SCALE / FULL lines ---"
    grep -aE 'PAIRED_RESULT|SCALE_VERDICT|FULL_PAIRED|FAIR PAIRED CYCLES COMPLETE|SCALE GATE|FULL FAIR' cycles.log 2>/dev/null | tail -12 || true
    echo "--- fair queue remaining (from run_cycles_fair.sh) ---"
    if pgrep -f 'run_cycles_fair.sh' >/dev/null; then
      echo "  run_cycles_fair: RUNNING"
    else
      echo "  run_cycles_fair: done"
    fi
    if pgrep -f 'fair_pair.py.*128' >/dev/null; then
      echo "  current: 128px fair_pair active"
    elif pgrep -f 'fair_pair.py' >/dev/null; then
      echo "  current: fair_pair (other)"
    fi
    if pgrep -f 'run_scale_gate' >/dev/null; then
      echo "  scale_gate: RUNNING"
    elif pgrep -f 'fair_pair_full' >/dev/null; then
      echo "  full_fair: RUNNING"
    elif pgrep -f 'run_full_fair.sh' >/dev/null; then
      echo "  run_full_fair: waiting or gate"
    fi
  } >> "$LOG" 2>&1

  # Stop when nothing left to watch for 2 consecutive checks
  if ! pgrep -f 'run_cycles_fair|run_full_fair|fair_pair|scale_verify|fair_pair_full' >/dev/null; then
    log "=== all pipeline processes finished — monitor stopping ==="
    python3 exps/cycles_summary.py >> "$LOG" 2>&1 || true
    grep -aE 'PAIRED_RESULT|SCALE_VERDICT|FULL_PAIRED' cycles.log >> "$LOG" 2>&1 || true
    break
  fi
  sleep "$INTERVAL"
done
