# Resolution scalability gate (64 → 128 → 256)

Before committing GPU days to full 256px × 1000 clips × 2 seeds, the pipeline verifies that H2 signal and training quality **scale across resolution**.

## When it runs

`run_full_fair.sh` calls `run_scale_gate.sh` immediately after `run_cycles_fair.sh` releases the cycles lock (128px queue finished).

## Phase A — 64 vs 128 (from `cycles.log`)

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| 128px fair `mean_delta` | ≥ 0.30 dB | Prereg H2 minimum |
| Physics win rate | ≥ 85% | Paired clips |
| 128/64 delta ratio | ≥ 0.35 | H2 must not collapse at higher res |
| 128px gain vs static | ≥ 0.45 dB | Learned headroom preserved |

If Phase A fails → `run_scale_fixes.sh` (extra 128px fair runs), then re-check.

## Phase B — cheap 256px probe

One fair paired run: **4000 steps**, 200 test clips, `charb+cosine`, same recipe as 128px winner.

| Probe | Pass |
|-------|------|
| `base_ch=32` | `mean_delta` ≥ 0.22 dB |
| else `base_ch=48` | retry probe |

On pass, writes `runs/scale_gate_env.sh`:

- `SCALE_FULL_BASE_CH`, `SCALE_FULL_BATCH`, `SCALE_FULL_STEPS` (12k or 16k if wider model needed)

On fail → **full plan blocked**; see `runs/scale_verdict*.json` strategies.

## Manual commands

```bash
# After 128px fair queue completes:
python3 exps/scale_verify.py --log cycles.log --require_128_fair

# With 256 probe only:
python3 exps/scale_verify.py --probe256 --skip_trend

# Full gate (what run_full_fair uses):
bash run_scale_gate.sh
```

## If not scalable — strategy order

1. **Widen model at 256** — `base_ch=48` or `64` (fast harness uses 32; full `base.yaml` uses 64).
2. **More steps** — scale ~linearly with pixels (16k–20k at 256px).
3. **Match train budget** — fair 128px with `n_train=3000` vs 6000 for apples-to-apples with 64px.
4. **Regularization** — try `wd=0.05`, `drop=0.1` at high res.
5. **Progressive finetune** — train 128px, finetune 256px (needs `--init_ckpt` support).
