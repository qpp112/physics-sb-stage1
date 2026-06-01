# Overnight monitor snapshot

**Written:** 2026-05-31 ~07:45 (when you went to sleep)

## What's running

| Process | Status |
|---------|--------|
| `run_cycles_fair.sh` | **Active** — holds GPU lock |
| `fair_pair.py` 128px 8k/6000 charb cosine | **Training** (physics pass, then brownian) |
| `run_full_fair.sh` | **Waiting** on lock — will run scale gate then 256px full |

## Fair queue progress

| # | Config | Status |
|---|--------|--------|
| 1 | 64px 3k mse | Done (+1.55 dB) |
| 2 | 64px 3k mse noise 0.1 | Done (+1.55 dB) |
| 3 | **128px 8k 6k charb** | **In progress** |
| 4 | 128px 8k noise 0.1 | Queued |
| 5 | 128px 12k 6k charb | Queued |

## After 128px queue (automatic)

1. `run_cycles_fair.sh` exits → releases lock  
2. `run_full_fair.sh` wakes → **`run_scale_gate.sh`**  
   - Phase A: 64→128 scalability checks  
   - Phase B: cheap 256px probe (~1h)  
3. If **PASS** → full 256px × 2 seeds (many hours)  
4. If **FAIL** → full plan **blocked**; see `runs/scale_verdict*.json`

## Logs to check when you wake up

```bash
cd ~/physics-sb-stage1
tail -50 monitor.log          # periodic snapshots (every 5 min)
tail -30 scale_gate.log       # gate verdict (after 128 queue)
grep SCALE_VERDICT cycles.log
grep PAIRED_RESULT cycles.log
python3 exps/cycles_summary.py
```

## Background monitor

`monitor_loop.sh` appends to `monitor.log` every 5 minutes until all jobs finish.
