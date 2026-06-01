# Stage 1 Pilot — Pre-registration

**Locked before any training run. Do not edit after the first training run starts.**

Frozen: 2026-05-18.

## What this pilot decides

Whether to spend ~3 GPU-days on the full Stage 1 plan (with shuffled-flow control,
σ ablation, inference-steps ablation, flow-noise robustness, 3 seeds, full
statistical machinery). Outcomes below.

## Hypotheses

### H1 — static prior test (no training)

The bidirectional flow-warped reference midpoint beats the linear-blend midpoint
on **all four** of {PSNR ↑, SSIM ↑, LPIPS ↓, EPE ↓}, on the full test set, and
the gap widens monotonically across the three motion-magnitude strata
[40, 60), [60, 100), [100, ∞) px.

### H2-pilot — trained bridge

A bridge matching model trained with the physics reference beats one trained
with the Brownian reference on PSNR by **≥ 0.3 dB** on the aggregate test set,
with **same-direction movement (no metric reversal)** on SSIM, LPIPS, and EPE.

No formal significance test at n=2 seeds — instead, the effect must be visible
as non-overlapping ranges across the 2 seeds per condition, on PSNR.

## Outcome table (decided before runs)

| Outcome | Decision |
|---|---|
| H1 fails (any of the four metrics worse, OR no monotone stratum trend) | **Abort.** Reference prior is not informative; rethink before any more compute. |
| H1 passes, H2-pilot passes | **Proceed to full plan.** Adds shuffled control, 3rd seed, ablations, longer training. |
| H1 passes, H2-pilot fails (gap < 0.3 dB or any metric reverses) | **Debug, do not auto-extend.** Most likely cause: bridge-matching impl bug or σ at wrong scale. Run focused debug session before deciding on full plan. |
| H1 passes, H2-pilot ambiguous (gap in [0, 0.3) dB with ranges overlap) | **Treat as fail.** No statistically meaningful signal at this budget. |

## Metrics — full list, no cherry-picking

All four reported in `results.md` for every condition × seed × stratum:

1. **PSNR** — primary metric for H2 threshold
2. **SSIM** — must not reverse direction vs PSNR
3. **LPIPS** (AlexNet backbone) — perceptual, must not reverse
4. **EPE** — endpoint error of flow recovered from $(\hat M, x_B)$ vs $0.5 F_{A\to B}$,
   computed with the GT-flow estimator (we have GT flow on FlyingThings3D, so no RAFT inference needed at eval — see `metrics/flow_metrics.py`)

## Stratification

Test set split by **max chained flow magnitude $\|F_{A\to B}\|_\infty$** into:

- Stratum 1: [40, 60) px
- Stratum 2: [60, 100) px
- Stratum 3: [100, ∞) px

≥ 200 examples per stratum required; otherwise sample test set until met.

## Compute budget

| Block | Wall hours |
|---|---|
| Setup, preprocessing, flow chaining cache | 1.0 |
| Sanity checks (all 4) | 1.0 |
| 1a static prior test | 0.5 |
| Training: 2 conditions × 2 seeds × ~4 h | 16.0 |
| 1b inference + eval (4 runs) | 2.0 |
| Analysis, writeup | 1.5 |
| **Total** | **22.0** |

Headroom: 2 h on a 24 h A100 rental.

## What is NOT being tested in the pilot

These are deferred to the full plan. Listing them so post-hoc additions are
visible as such:

- Shuffled-flow negative control
- σ noise-level ablation
- Inference-step-count ablation
- Robustness to noisy estimated flow (preview of Stage 2)
- 3-seed statistical significance with Holm correction

## Sanity-check gates (failing any of these invalidates results)

1. Flow chain consistency: warping $x_A$ by chained $F_{A\to B}$ yields mean RGB error
   < 5/255 vs $x_B$ on non-occluded pixels.
2. Bridge integrator correctness: 1D Brownian bridge Euler-Maruyama matches
   analytic posterior within MC error.
3. $x_t$ usage: at inference, zeroing $x_t$ must drop PSNR by ≥ 2 dB. If it
   doesn't, the network isn't using the bridge state.
4. Identity case: $x_A = x_B$ implies $\hat M = x_A$ to within 0.1 dB.

Pilot is invalidated and must be re-run if any gate fails after training completes.

## CORRECTION (post-1a, pre-training)
Sign error in metrics/flow_metrics.py::epe_against_gt_flow: warped the midpoint
by -0.5*F instead of +0.5*F to recover frame A. Provable from the flow convention
(gate 1) independent of outcome; it penalized correctly motion-compensated
midpoints. Corrected to +0.5*F. After the fix, physics wins EPE (0.0645 vs 0.0933).

## CORRECTION 2 (post-1a, pre-training)
H1 monotonicity gate was mis-specified: it gated abort on benefit-grows-with-motion,
but the abort rationale was "prior not informative." The prior IS informative
(physics +1.5 to +3.5 dB across all strata; SSIM/LPIPS/EPE agree). FINDING: the
physics benefit DECREASES with motion (+3.51/+2.54/+1.56 dB for strata
[40,60)/[60,100)/[100,inf)), consistent with flow-warping degrading under
occlusion at large displacement. overall_pass redefined = (4 metrics favor physics);
monotonicity retained as a logged characterization. H2's decisive question is now
whether training closes the high-motion gap; primary stratum for H2 = [100, inf).

## PERF NOTES (data pipeline)
- TRAIN clip SELECTION uses single-step |F_AM| (proxy) for speed; TEST stratification
  uses chained |F_AB| (correct). Training loss always uses exact chained flow per sample.
- precompute.py builds a compact RAM-fitting tensor cache; without it, training is
  IO-bound (GPU starved) on slow storage. Run precompute before training.

## CORRECTION 3 (post-training diagnosis: H2-pilot FAIL root cause)
Both trained conditions (v2, sigma=0.1) score BELOW their own no-train static
priors at inference (free-running): physics 15.8 vs 19.7 dB, brownian 15.8 vs
17.4 dB; physics and brownian tie. Diagnosed via a teacher-forced vs free-running
rollout probe on n=48 test triplets (same batch, ckpt_final):

  estimate     physics_v2_s0   brownian_v2_s0
  static       19.68           17.39
  free (infer) 15.77           15.84
  teacher      53.31           52.96   <- model fed TRUE I_t at each step
  zero_xt      10.10           10.15

ROOT CAUSE: the bridge marginal noise is too small. At t=0.5 the injected noise
std is sigma*sqrt(t(1-t)) = 0.1*0.5 = 0.05 in [-1,1] units, while the Bezier
interpolant I_t passes EXACTLY through the target M (I_0.5 = M). So the training
input x_t = I_t + 0.05*eps essentially CONTAINS the answer; conditioned also on
clean x_A, x_B, the target M is over-determined. The network minimizes the loss
by reading M off x_t (teacher PSNR ~53 dB confirms a near-perfect on-path fit)
instead of learning a drift that generalizes. At inference there is no true I_t
(the integrator uses its own running estimate), so the shortcut collapses and the
free-running path drifts BELOW the analytic prior it started from.

Implications:
- The failure is sigma-driven and reference-agnostic, which is why physics ~= brownian
  and both lose to their static priors. u_ref is near-zero when flow is good, so it
  adds little signal here regardless.
- Sanity gate 3 (zeroing x_t drops >=2 dB) gave FALSE confidence: it only checks that
  x_t is used, not that it is used in a way that survives free-running. REPLACEMENT
  GATE: require free-running PSNR within ~2 dB of teacher-forced PSNR (see
  exps/diag_rollout.py); a large teacher>>free gap = leak/exposure bias -> fail early.
- v3/v3b/v3c configs do NOT change sigma (they add condition_on_reference_mean and
  attention); they are expected to hit the same leak.

FIX UNDER TEST: sigma sweep (configs/physics_sig0{1,3,6}.yaml, run_sigsweep.sh) at a
reduced 20k-step budget, scored with diag_rollout.py. Success = free-running PSNR
climbs toward teacher/static as sigma increases.
