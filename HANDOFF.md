# Physics-SB Stage 1 — Agent Handoff Document

**Repo:** `/home/ubuntu/physics-sb-stage1`  
**Dataset:** `/home/ubuntu/datasets/FlyingThings3D` (cleanpass, min flow 40px)  
**GPU:** RTX A6000 (~48 GB)  
**Prereg:** `prereg.md` (frozen 2026-05-18; CORRECTIONS 1–3 document bridge failure)

---

## 1. Executive summary

| Question | Status |
|----------|--------|
| **H1** (static physics prior vs brownian blend) | **PASS** — physics wins PSNR/SSIM/LPIPS/EPE; benefit decreases at high motion (logged, not abort) |
| **H2 via bridge matching** (original prereg) | **FAIL / abandoned** — exposure bias + `x_t` leak at σ=0.1; σ sweep {0.1,0.3,0.6} all fail |
| **H2 via refine_resid** (one-shot midpoint) | **PASS at 64px & 128px** under **fair paired** protocol; physics +1.5–1.8 dB vs brownian |
| **256px full pilot** | **In progress** — first probe OOM’d (eval bug, fixed); `run_resume_full.sh` running 256px probe |

**Strategic pivot:** Stop bridge open-loop training. Use **direct midpoint refinement**:

```
M_hat = mid + f(mid, x_A, x_B, u_ref)   # variant refine_resid
```

No SDE integration at inference → no exposure bias; residual form floors at static prior.

---

## 2. Key findings

### 2.1 Bridge path (dead)

- Trained v2 models ~**16 dB** free-running vs static prior ~**20 dB** (physics) / ~**17 dB** (brownian).
- Physics ≈ brownian on trained PSNR (reference-agnostic failure).
- **Root cause** (`prereg.md` CORRECTION 3): At σ=0.1, `x_t = I_t + noise` with `I_{0.5} = M` leaks GT midpoint; teacher-forced ~53 dB, free-running ~16 dB.
- **Sigma sweep** (`sigsweep.log`, `run_sigsweep.sh`): σ ∈ {0.1, 0.3, 0.6} — all **GATE: FAIL** (free << static, teacher−free ~35 dB).
- Integrator verified OK (oracle drift ~35 dB); problem is training/inference mismatch, not integrator bug.
- v3/v3b configs only add `condition_on_reference_mean`; same σ/leak — do not invest further without architectural change.

### 2.2 Refine path (works)

- **Unpaired** fast cycles @ 128px: physics **21.17 dB**, gain **+0.86 dB** vs static (charb+cosine, 8k steps, n_train=6000).
- **Fair paired** (only valid H2 metric): same data, init, batch order; only reference differs.

| Resolution | Config | mean Δ (phys−brwn) | Physics PSNR | Brownian PSNR | Wins |
|------------|--------|-------------------|--------------|---------------|------|
| 64px | mse, 3k steps | +1.55 dB | 20.87 | 19.32 | 194–195/200 |
| 128px | charb+cosine, 8k | +1.73–1.76 dB | 21.21 | 19.45–19.49 | 192–196/200 |
| 128px | charb+cosine, 12k | +1.76 dB | **21.23** | 19.48 | 192/200 |

- High-motion stratum `[100,∞)` at 128px: **+1.26 to +1.41 dB** paired delta.
- Scale gate 64→128: delta ratio **0.85**, gain **+0.76 dB** — **SCALABLE** (see `runs/scale_verdict.json`).

### 2.3 Fairness trap (important)

- **Do not** rank H2 by `gain = trained − static` across separate runs: brownian static is ~2.3 dB worse, so brownian can show larger gain while worse absolute PSNR.
- **Do** use `mean_delta = PSNR_physics − PSNR_brownian` on identical test clips (`exps/fair_pair.py`).
- Documented in `FAIRNESS.md`, `exps/analyze_h2_gap.py`.

### 2.4 256px OOM incident (fixed, not scalability failure)

- First `SCALE_PROBE256` completed **training** (~11 min/model) then **OOM on eval**: `eval_ref()` used hardcoded **batch=64** at 256px → 16 GB alloc.
- **Fix:** `--eval_batch` in `fair_pair.py` (auto: 64@64px, 16@128, 4@256); eval after each model + `del` + `empty_cache()`; probe uses train/eval batch **4**.
- Resume: `run_resume_full.sh` (started 18:22); Phase A re-passed; 256px probe in flight at handoff write time.

---

## 3. Code & infrastructure added/changed

### 3.1 New experiment scripts

| File | Purpose |
|------|---------|
| `exps/fast_cycle.py` | Fast hyperparam search @ 64/128px; disk cache `runs/_fastcache/`; variants `refine`, `refine_resid` |
| `exps/fair_pair.py` | Fair paired physics vs brownian; shared init & batch indices; `PAIRED_RESULT` log line |
| `exps/fair_pair_full.py` | Full 256px eval: 1000 test clips, PSNR/SSIM/LPIPS/EPE, JSON + prereg gate |
| `exps/scale_verify.py` | 64→128 trend checks + optional 256px probe; writes `runs/scale_verdict.json` |
| `exps/diag_rollout.py` | Teacher vs free vs static gate (replaces misleading sanity gate 3) |
| `exps/cycles_summary.py` | Leaderboard from `cycles.log`; FAIR PAIRED + FULL_PAIRED tables |
| `exps/analyze_h2_gap.py` | Explains gain vs mean_delta artifact |
| `exps/1c_refine.py` | Full-scale refine trainer (DataLoader path, EMA, strata JSON) |

### 3.2 Shell queues

| Script | Role |
|--------|------|
| `run_cycles.sh`, `run_cycles_wave*.sh`, `run_cycles_rerun.sh` | Unpaired fast cycles (historical) |
| `run_cycles_fair.sh` | Fair paired @ 64 & 128 — **COMPLETE** |
| `run_scale_gate.sh` | Scalability gate before full 256px |
| `run_scale_fixes.sh` | Extra 128px fair if gate fails |
| `run_full_fair.sh` | Gate → prebuild cache → 2 seeds @ 256px |
| `run_resume_full.sh` | Resume after OOM fix (skip lock wait) |
| `run_queue_all.sh` | nohup launcher for full fair |
| `monitor_loop.sh` | Appends to `monitor.log` every 5 min |
| `cycles_common.sh` | Shared `run_cycle()`, flock `/tmp/run_cycles.lock` |

### 3.3 Docs

| File | Content |
|------|---------|
| `FAIRNESS.md` | Fair paired protocol |
| `SCALE_GATE.md` | Resolution scalability criteria |
| `MONITOR_STATUS.md` | Overnight snapshot (may be stale) |
| `HANDOFF.md` | This file |

### 3.4 Configs (bridge — legacy)

- `configs/physics_sig{01,03,06}.yaml` — sigma sweep arms
- `configs/physics_refine.yaml`, `brownian_refine.yaml` — 1c_refine arms
- `configs/base.yaml` — full model: base_ch=64, 75k steps (not used in fast/fair harness)

### 3.5 `fair_pair.py` API (representative)

```bash
python3 exps/fair_pair.py \
  --size 128 --steps 8000 --n_train 6000 \
  --weight_decay 0.1 --dropout 0.2 \
  --loss charb --lr_schedule cosine \
  --batch 32 --eval_batch 16   # eval_batch auto if omitted
```

### 3.6 Winning recipe (128px fair)

- `variant=refine_resid`
- `loss=charb`, `lr_schedule=cosine`
- `weight_decay=0.1`, `dropout=0.2`
- `steps=8000–12000`, `n_train=6000` (cache may load **4500** clips — dataset limit)
- `base_ch=32`, UNet small (fast harness)

---

## 4. Results log reference

**Primary log:** `cycles.log` — grep `PAIRED_RESULT`, `SCALE_VERDICT`, `FULL_PAIRED_RESULT`

```
PAIRED_RESULT size=128 steps=8000 ... | phys_tr=21.209 brwn_tr=19.448 mean_delta=+1.761 hi_delta=+1.407 phys_wins=196/200 verdict=physics
```

**Other logs:** `sigsweep.log`, `full_fair.log`, `scale_gate.log`, `resume_full.log`, `monitor.log`

**Artifacts:** `runs/fair_full/s256_*_seed*/fair_full_eval.json` (after full run)

---

## 5. Prereg alignment (for next agent)

Original `prereg.md` H2 targets **trained bridge**. Current evidence supports:

- **Recommend CORRECTION 4** (not yet in prereg): H2 redefined on `refine_resid` + fair paired `mean_delta ≥ 0.3 dB` + metric direction checks on full 256px × 2 seeds × 1000 clips.

**H2 gate (refine, fair):**

- `mean_delta ≥ +0.3 dB` aggregate PSNR
- SSIM ↑, LPIPS ↓, EPE ↓ same direction as PSNR (`fair_pair_full.py` `prereg_verdict()`)
- 2 seeds, non-overlapping ranges (full plan)

**Do not proceed on bridge** without fixing σ/path or new architecture.

---

## 6. Pipeline state (updated 2026-05-31 ~19:20)

```
[done] H1 static prior (1a_prior_eval)
[done] Bridge v2 train + diag — FAIL
[done] Sigma sweep — FAIL
[done] Fast cycles waves 2–7 + rerun
[done] run_cycles_fair.sh (64 + 128 paired)
[done] Scale gate Phase A (64→128 SCALABLE)
[done] Scale gate Phase B @ 256 probe — PASS (+1.815 dB, eval_batch=4 fix)
[done] Full 256px seed=0 — prereg PASS (1000 clips, all metrics)
[in progress] Full 256px seed=1 — training (~50 min GPU so far)
[pending] cycles_summary after seed=1; prereg CORRECTION 4 writeup
```

**Seed 0 full @ 256px:** `FULL_PAIRED_RESULT` — mean_delta **+1.762 dB**, wins **969/1000**, SSIM/LPIPS/EPE all favor physics, **`prereg=PASS`**. Artifact: `runs/fair_full/s256_st12000_..._seed0/fair_full_eval.json`.

Check live status:

```bash
cd ~/physics-sb-stage1
pgrep -af 'fair_pair|resume|run_full'
tail -f resume_full.log cycles.log
grep -E 'PAIRED_RESULT|SCALE_VERDICT|FULL_PAIRED' cycles.log
python3 exps/cycles_summary.py
```

---

## 7. What’s left

1. **Finish 256px probe** — confirm `PAIRED_RESULT size=256` in `cycles.log` with mean_delta ≥ ~0.22 dB (relaxed for 4k steps).
2. **Full fair eval** — `fair_pair_full.py` @ 256, 12k steps (or 16k if base_ch=48), batch=4, 1000 test, seeds 0 & 1.
3. **Prereg decision** — if full PASS → proceed to full plan on **refine** path (3 seeds, shuffled control, etc.).
4. **Optional:** Add CORRECTION 4 to `prereg.md`; `--init_ckpt` for 128→256 progressive finetune if probe marginal.

**Not recommended:** More bridge training at any σ; ranking H2 on unpaired `gain`; v3 bridge variants without leak fix.

---

## 8. Data & cache

- Precompute: `.pilot_cache/` (full pipeline)
- Fast cache: `runs/_fastcache/ft3d_{size}_{n_train}_{n_test}_40.pt` — built at 256 native, downsampled in-memory for smaller sizes
- 256 full cache: `ft3d_256_3000_1000_40.pt` (built by `run_full_fair.sh` prebuild)

---

## 9. Model architecture notes

- **Fast/fair harness:** `UNet` base_ch=32, channel_mult=(1,2,2), in_channels=12 (with u_ref), `image_size` arg
- **Full plan (`base.yaml`):** base_ch=64, mult (1,2,4,4), ~30M params @ 256 — may be needed if base_ch=32 probe passes but full underperforms

---

## 10. Quick commands cheat sheet

```bash
cd ~/physics-sb-stage1

# Leaderboard
python3 exps/cycles_summary.py

# Fair run (single)
python3 exps/fair_pair.py --size 128 --steps 8000 --n_train 6000 \
  --weight_decay 0.1 --dropout 0.2 --loss charb --lr_schedule cosine

# Scale gate only
bash run_scale_gate.sh

# Full pipeline (after lock free)
bash run_full_fair.sh
# or after OOM fix:
bash run_resume_full.sh

# Diagnose bridge (historical)
python3 exps/diag_rollout.py --config configs/physics_v2.yaml --run_name physics_v2_s0
```

---

## 11. Agent context

- User goal: dominate / validate physics reference for video frame interpolation on FT3D; fail-fast small cycles then scale winners.
- User cares about **fair** physics vs brownian comparison and **scalability** before full GPU commit.
- Conversation transcript: `~/.cursor/projects/home-ubuntu/agent-transcripts/88849613-eb49-4684-8619-a072dc132a39/88849613-eb49-4684-8619-a072dc132a39.jsonl`

---

*Generated for handoff to a new Claude agent. Update §6 when 256px probe/full runs complete.*
