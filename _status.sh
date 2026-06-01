#!/usr/bin/env bash
echo "drivers (run_cycles):"
ps -eo pid,etime,args | grep -E "[r]un_cycles\.sh" || echo "  none"
echo "workers (fast_cycle):"
ps -eo pid,etime,args | grep -E "[f]ast_cycle\.py" | sed 's/--root.*//' || echo "  none"
echo "GPU mem:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== cycles.log tail ==="
tail -n 20 cycles.log 2>/dev/null
