#!/usr/bin/env bash
# Queue full 256px fair eval after the current run_cycles_fair.sh finishes.
# Does NOT restart fair cycles (already running).
set -uo pipefail
cd "$(dirname "$0")"

if pgrep -f "run_cycles_fair.sh" >/dev/null; then
  echo "run_cycles_fair.sh still running — full queue will wait on lock"
else
  echo "no run_cycles_fair.sh — full queue acquires lock when started"
fi

if pgrep -f "run_full_fair.sh" >/dev/null; then
  echo "run_full_fair.sh already queued/running"; exit 0
fi

nohup bash ./run_full_fair.sh >> queue_all.log 2>&1 &
echo "Queued run_full_fair.sh PID=$!  log: queue_all.log full_fair.log cycles.log"
