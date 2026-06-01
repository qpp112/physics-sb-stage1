# Fair comparison protocol (physics vs brownian)

## Problem
Separate `fast_cycle.py` runs are **not fair** for H2:
- `gain = trained - static` uses different static baselines per reference (~2.3 dB apart)
- Runs may differ in config, init, or batch order unless explicitly matched

## Fair protocol (`exps/fair_pair.py`)
One script trains **both** references under identical conditions:

| Locked equal | Different (hypothesis) |
|--------------|------------------------|
| Same train/test triplets (shared cache) | Reference midpoint `mid(t=0.5)` |
| Same UNet architecture & hyperparameters | Reference drift `u_ref` |
| Same weight initialization | Residual target `M - mid` |
| Same SGD batch indices each step | |
| Eval on identical test clips | |

## Primary H2 metric
**`mean_delta = mean(PSNR_physics - PSNR_brownian)`** on the same test samples.

Also report: `physics_wins N/200`, `hi_delta` on `[100,∞) px` stratum.

Do **not** use raw `gain` across references for H2.

## Commands
```bash
cd ~/physics-sb-stage1
python3 exps/fair_pair.py --size 128 --steps 8000 --n_train 6000 \
    --weight_decay 0.1 --dropout 0.2 --loss charb --lr_schedule cosine
python3 exps/cycles_summary.py   # shows FAIR PAIRED table first
./run_cycles_fair.sh             # queued fair sweep
```
