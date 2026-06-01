#!/usr/bin/env bash
# Kill any lingering experiment processes. This script's own command line does
# not contain the match pattern, so it won't self-terminate.
for p in $(pgrep -f "run_cycles"); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f "variant"); do kill -9 "$p" 2>/dev/null; done
sleep 2
echo "remaining drivers: $(pgrep -cf 'run_cycles')  workers: $(pgrep -cf 'variant')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
