#!/usr/bin/env bash
# Bundle irreplaceable artifacts before shutdown (~100–200 MB).
# Usage:  ./save_essentials.sh [output.tar.gz]
#
# Skips runs/_fastcache (~30G) and multi-GB bridge checkpoints (rebuildable).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d)"
OUT="${1:-$HOME/physics-sb-stage1-essentials-${STAMP}.tar.gz}"

cd "$ROOT"
echo "Creating: $OUT"

tar -czf "$OUT" \
  --exclude='runs/_fastcache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  model exps bridges data configs \
  *.md *.sh \
  arch_cycles.log cycles.log cycles_queue.log monitor.log \
  latent_*.out arch_cycle_64.out \
  runs/latent \
  runs/fair_full \
  runs/scale_verdict.json \
  2>/dev/null || \
tar -czf "$OUT" \
  model exps bridges data \
  RESULTS.md architecture.md HANDOFF.md FAIRNESS.md prereg.md README.md \
  arch_cycles.log cycles.log \
  runs/latent runs/fair_full

ls -lh "$OUT"
echo ""
echo "Copy off-server:"
echo "  scp ubuntu@<host>:$OUT ."
