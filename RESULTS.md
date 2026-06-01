# Stage 2 — Gate Results Log

One line per gate: metric + pass/fail. Append-only. Fair paired only
(`mean_delta = PSNR_physics − PSNR_brownian` on identical clips). Never rank by gain.

| Date | Gate | Metric | Threshold | Result | Pass/Fail |
|------|------|--------|-----------|--------|-----------|
| 2026-05-31 | P0 seed0 | full256 mean_delta=+1.762 dB, wins=969/1000, SSIM/LPIPS/EPE all favor physics | prereg PASS | +1.762 dB | PASS |
| 2026-05-31 | P0 seed1 | full256 mean_delta=+1.775 dB, wins=971/1000, SSIM/LPIPS/EPE all favor physics | prereg PASS | +1.775 dB | PASS |
| 2026-05-31 | A1 | static@128, 200 clips: V2(physics_v2)=20.210 dB vs V1(physics)=20.312 dB, delta=-0.103 | V2 > V1 | -0.103 dB | **FAIL — HALT** |

## A1 FAILURE — HALT (per invariant 3)

Acceleration-aware warp `PhysicsReferenceV2` does **not** beat V1. Measured on the
faithful pipeline (load@256 → downsample@128, so V1 reproduces the documented
20.31 dB baseline). V2 is worse by 0.10 dB, uniformly across strata:

| stratum | V1 PSNR | V2 PSNR |
|---------|---------|---------|
| [40,60)   n=47  | 23.534 | 23.391 |
| [60,100)  n=45  | 21.533 | 21.427 |
| [100,inf) n=108 | 18.402 | 18.319 |

**Interpretation:** FT3D 3-frame motion is near-linear; the acceleration proxy
`a = f2 - f1` (from chained half-flows F_{A->M}, F_{M->B}) is dominated by flow
noise/occlusion artifacts, so the 2nd-order `0.125*a` correction adds more error
than signal. V2 reduces EXACTLY to V1 when `a=0` (verified), so this is a clean
negative result, not a wiring bug. Halting per spec; not improvising a new prior.

**Harness note:** initial run used native-128 loading and read V1 at 18.87 dB
(≠ documented 20.3). Fixed `eval_static.py` to load@256 then downsample (matching
`fast_cycle`), which reproduces V1=20.31 dB. The gate verdict is unchanged (FAIL)
under either pipeline — V2 < V1 both ways. This was a harness-fidelity fix, not a
gate change.

## Architecture cycling (post-A1) — learned priors beat the static floor

A1 killed the *hand-crafted* acceleration prior (V2). Per user direction we then
cycled *learned* midpoint architectures (Options 2 & 3) via `exps/arch_cycle.py`,
ranking by midpoint PSNR vs GT @64px with physics-V1 as the floor (20.177 dB @64px).
All learned archs use a residual-around-V1 design (zero-init head == V1 exactly,
a strict floor — same trick that made `refine_resid` work).

### Cycle 1 (@64px, 4000 steps, n_train=3000)

| arch | PSNR | hi[100,∞) | vs V1 | hi vs V1 | SSIM | params | verdict |
|------|------|-----------|-------|----------|------|--------|---------|
| `flow_synth` (Opt 2) | 20.274 | 18.492 | **+0.096** | +0.050 | 0.631 | 1.5M | **WIN** |
| `physics_v1` (floor) | 20.177 | 18.442 | — | — | 0.627 | 0 | — |
| `feat_warp` (Opt 3) | 19.894 | **18.804** | −0.283 | **+0.362** | 0.578 | 1.1M | lose / hi-win |
| `feat_flow` (2+3) | 19.378 | 18.152 | −0.799 | −0.290 | 0.563 | 2.7M | lose |

**Interpretation:**
- `flow_synth` (learn `F_{A→M}` as residual of `0.5·F_AB`, then warp+blend) is the
  first *trained* prior to beat the static floor overall — opposite of A1/V2.
- `feat_warp` validates Option 3's occlusion thesis: loses overall (lossy
  autoencoder blurs easy regions) but **wins the hardest large-motion stratum
  by +0.36 dB**. The decoder bottleneck (no pixel skip) sinks global fidelity.
- Fix tested in Cycle 2: feature correction as a **residual on the V1 floor**
  (`feat_resid`) + occlusion-aware blend on the winner (`flow_synth_occ`).

### Cycle 2 (@64px, 8000 steps, n_train=3000)

| arch | PSNR | hi[100,∞) | vs V1 | hi vs V1 | SSIM | train_loss | verdict |
|------|------|-----------|-------|----------|------|-----------|---------|
| `flow_synth` | 20.284 | 18.499 | **+0.107** | +0.057 | 0.631 | 0.128 | **WIN** |
| `physics_v1` | 20.177 | 18.442 | — | — | 0.627 | — | floor |
| `flow_synth_occ` | 20.168 | 18.373 | −0.009 | −0.069 | **0.643** | 0.118 | tie |
| `feat_resid` | 19.729 | 17.837 | −0.449 | −0.605 | 0.620 | **0.099** | lose |

**Decisive lessons (shape the Option-3 design):**
1. **`flow_synth` saturates near +0.10 dB** at 64px (4k→8k steps: +0.096→+0.107).
   A learned warp can only resample existing pixels → it generalizes but its
   ceiling as a *prior* is small. Real gains live in the refiner/bridge on top.
2. **`feat_resid` overfits** — lowest train loss, worst test. Residual-anchoring
   guarantees the V1 floor only at *init*, not after training: a free pixel
   decoder memorizes corrections that add blur everywhere at test. **A latent /
   decode path MUST be a faithful codec (recon loss) with pixel skips, or it
   degrades easy regions.** This is the central constraint for Option 3 + latent bridge.
3. **`flow_synth_occ` has the best SSIM (0.643)** — the occlusion mask improves
   *structure* but not PSNR. Useful signal for perceptual metrics (LPIPS/SSIM).

## Option 3 + cross-attention + latent codec (staged build)

Refined the user's Option-3 pipeline with four data-driven fixes: (R1) decode a
**residual on the pixel-warp floor** (not a replacement), (R2) **pixel skip
connections** from both frames, (R3) cross-attention at the **bottleneck** (/8),
(R5) a **faithful-codec pre-gate**. Built in `model/latent_bridge.py`,
trained by `exps/latent_stage.py`.

### S0 — codec pre-gate (AE-GATE)
`SharedAE` (enc/dec + pixel skips), L1+MSE+edge recon, 6k steps @64px:
**recon PSNR = 50.17 dB, SSIM 0.9997 → PASS** (gate ≥32). Near-lossless codec, so
the latent path is not bottlenecked — the exact failure mode that sank `feat_resid`.

### S1 — latent prior (cross-attn + f_interp + residual decode), 8k steps @64px

| variant | PSNR | hi[100,∞) | vs V1 | hi vs V1 | SSIM | params |
|---------|------|-----------|-------|----------|------|--------|
| s1_scratch (no AE-init, all trainable) | **20.605** | **18.778** | **+0.428** | **+0.336** | 0.654 | 4.18M |
| s1_warm (AE-init, all trainable) | 20.602 | 18.750 | +0.424 | +0.308 | **0.657** | 4.18M |
| s1_frozen (AE-init, enc frozen) | 20.338 | 18.693 | +0.160 | +0.251 | 0.630 | 2.58M |
| physics_v1 floor | 20.177 | 18.442 | — | — | 0.627 | — |

**This is the new best architecture by a wide margin** — +0.43 dB (≈4× `flow_synth`'s
+0.107), wins the hard stratum (+0.34), best SSIM (0.66). Validates Option 3 + cross-
attention once the codec is faithful and the decode is residual-on-floor with pixel skips.

**Ablation:** scratch (+0.428) ≈ warm (+0.424) ≫ frozen (+0.160). So S0 pretraining is
*not* the source of the gain (end-to-end reaches the same place); the **trainable encoder**
is essential (freezing it to pure-recon features loses 0.27 dB). S0's value was the
feasibility pre-gate (proving a faithful codec exists), not warm weights.
Next: S2 latent bridge (`z_t = z_mid + σε`), C-GATE = bridge must beat this one-shot prior.

### S2 — latent bridge (Stage 2C): C-GATE **FAIL** (all configs)

Denoiser refines `z_mid → z_M` from `z_t = z_mid + σε` (noised from the prior, never a
GT interpolant); zero-init residual ⇒ starts exactly at the S1 prior. Bar = 20.605 (S1).

| config | PSNR | C-GATE Δ (vs prior) | note |
|--------|------|--------------------|------|
| frozen, σmax 0.5, lat_w 0.1 | 20.403 | −0.202 | FAIL |
| frozen, σmax 1.0, lat_w 0.1 | 20.408 | −0.197 | FAIL |
| joint (no-freeze), σmax 0.5 | NaN | — | diverged |
| frozen, σmax 0.5, **lat_w 0** (clean) | 20.467 | −0.139 | FAIL (best of the four) |

**Verdict:** the latent bridge does **not** beat the one-shot S1 prior — even with pure
decode loss and a zero-init (do-no-harm) head, the refiner overfits and the sensitive
decoder amplifies the latent perturbation. This **repeats the Stage-1 finding**
(`refine_resid` one-shot ≻ bridge/SDE). The bridge framing keeps failing the gate.

**Best architecture found:** `LatentPrior` (Option 3 + cross-attention, residual-decode-
on-warp-floor with pixel skips), used **one-shot**: +0.43 dB over V1 @64px, +0.34 on the
hard stratum, best SSIM.

### Scale check — `LatentPrior` (S1) @128px

| size | floor (V1) | LatentPrior | vs V1 | hi[100,∞) vs V1 | SSIM |
|------|-----------|-------------|-------|------------------|------|
| 64px  | 20.177 | 20.605 | +0.428 | +0.336 | 0.654 |
| **128px** | 20.312 | **21.185** | **+0.873** | **+0.765** | 0.668 |

**The gain nearly doubles with resolution** (+0.43→+0.87 overall, +0.34→+0.77 hard
stratum), and the floor reproduces the documented V1@128 (20.31). The architecture
**scales positively** — cross-attention + feature-warp exploit the extra spatial detail.
This is the strongest single-architecture result in the project. Recommended next:
promote to 256px + fair physics-vs-brownian Δ inside this arch (flow-conditioned vs not).

## Baseline summary (Stage 1, to extend)

- Method: `refine_resid`, one-shot `M_hat = mid + f(mid, x_A, x_B, u_ref)`.
- Fair paired @256px, 12k steps, n_train=3000, charb+cosine, wd=0.1, drop=0.2, batch=4.
- **Baseline Δ = +1.77 dB** (mean of seeds 0/1: +1.762, +1.775).
- Physics static prior @256px (1000 clips): ~19.88 dB; physics trained: ~20.80 dB.
- High-motion `[100,inf)` paired Δ: +1.26 dB.

## Notes
- P0 ran via `exps/fair_pair_full.py` (completed 20:00). Artifacts in
  `runs/fair_full/s256_st12000_ntr3000_wd0.1_drop0.2_charb_seed{0,1}/fair_full_eval.json`.
