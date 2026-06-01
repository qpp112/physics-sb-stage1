# Physics-constrained Schrödinger Bridge — Stage 1 pilot (refined)

Tests whether a flow-warped physics prior beats a Brownian prior as the reference
process in an SB for video frame interpolation, on FlyingThings3D with GT flow.
See `prereg.md` for locked hypotheses, decision rules, and the corrections logged
during the first run.

This refined version folds in every fix from the first attempt: corrected EPE
metric, relative flow-chain sanity gate, YAML config fix, fast index build,
a progress bar on training, per-stratum H2 reporting, and — most importantly —
a **precompute** step that keeps the GPU fed.

## Hard-won setup requirements (read this first)

1. **Disk.** The optical_flow archive is ~310 GB compressed / ~430 GB extracted.
   You do NOT download it to disk — you stream-extract, keeping only
   `into_future/left`, which lands ~108 GB (or ~40 GB if you also keep one TRAIN
   letter). The frames archive is ~37 GB, ~18 GB kept (left only).
2. **RAM.** Training reads ~40 GB of raw clips per epoch; if that exceeds RAM the
   OS can't cache it and the GPU starves (~10x slowdown). The **precompute** step
   shrinks the working set to ~4-6 GB so it fits in RAM. Do not skip it.
3. **Run everything in tmux.** Downloads and training are long; a dropped SSH
   without tmux kills them.

## Get the data (streaming extract, keeps only what's used)

```bash
mkdir -p ~/datasets/FlyingThings3D && cd ~/datasets/FlyingThings3D
BASE=http://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D

# frames (left only) — ~37 GB download, ~18 GB kept
wget -O - --show-progress $BASE/raw_data/flyingthings3d__frames_cleanpass.tar \
  | tar -x --exclude '*/right'

# flow (into_future/left only) — ~310 GB download, ~108 GB kept, SLOW from LMB (~7 MB/s)
wget -O - --show-progress $BASE/derived_data/flyingthings3d__optical_flow.tar.bz2 \
  | pbzip2 -dc | tar -x --exclude '*/right' --exclude '*/into_past'
```

LMB is a single slow server; the flow download can take many hours. If your
instance allows torrents and ≥400 GB scratch, torrenting the archive is faster.
To trim disk further, add `--exclude 'optical_flow/TRAIN/B' --exclude 'optical_flow/TRAIN/C'`
(and the matching frames dirs) — one TRAIN letter has plenty for the pilot.

## Install + configure

```bash
python3 -m pip install -r requirements.txt --break-system-packages
sed -i "s|^  root:.*|  root: $HOME/datasets/FlyingThings3D|" configs/base.yaml
```

## Run

```bash
tmux new -s pilot
./run_pilot.sh 2>&1 | tee pilot.log
```

Order: sanity gates -> 1a static prior -> **precompute** -> 4 training runs -> eval.

| Phase | Notes |
|---|---|
| TRAIN index build | one-time, fast (single-step filter + early-stop), progress bar |
| 1a static prior | no training; prints H1 verdict |
| precompute | one-time ~60-90 min; builds compact RAM-fitting cache |
| training | 4 runs; with precompute the GPU is fed, ~1.5-3 h each on an A100 |
| eval | prints `__pilot_decision` + per-stratum gaps |

**Verify the GPU isn't starved early.** A few minutes into the first training
run, in another window run `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 2`.
It should hold high. If it sawtooths to 0%, precompute didn't take effect (check
for `DONE` flags in `~/datasets/FlyingThings3D/.pilot_cache/precomp_*/`).

## Reading results

- **H1**: `runs/1a_prior_eval/results.json` -> `H1.overall_pass` (4 metrics favor physics).
- **H2**: `runs/1b_eval/results.json` -> `__pilot_decision.verdict`, and
  `__per_stratum` for the gap at each motion level. The `[100, inf)` row is the
  decisive one: did training close the high-motion gap the static prior couldn't?

## What the first run taught us (now fixed in code)

- **EPE metric** warped by the wrong sign (-0.5F vs +0.5F); corrected.
- **Flow-chain gate** used an absolute threshold too strict for large-motion data;
  now a relative test (warp must beat no-warp by a margin).
- **YAML** parsed `1.0e9` as a string; use a plain integer.
- **Index build** read both flows + chained per triplet across all of TRAIN
  (~3.5 h); TRAIN now selects by single-step magnitude with early-stop.
  TEST still uses chained magnitude so the evaluation strata stay correct.
- **Training was IO-bound** (GPU starved). precompute fixes this; it also skips
  corrupt files once, up front, instead of crashing mid-run.

## Layout

```
prereg.md            Locked hypotheses + decision rules + corrections log
precompute.py        Build compact tensor cache (run before training)
run_pilot.sh         End-to-end driver
configs/             base.yaml + per-condition overrides
data/                FT3D loader (single-step/chained filter, precompute read-path)
bridges/             Reference paths, interpolant, integrator
model/               UNet (~24M params)
metrics/             PSNR/SSIM/LPIPS, EPE (corrected), bootstrap + pilot decision
exps/                sanity_checks, 1a_prior_eval, 1b_train, 1b_eval
```
