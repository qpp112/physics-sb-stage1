"""Why does brownian show higher *gain* than physics on fast cycles?

Compares both references on the IDENTICAL test triplets: static prior quality,
residual difficulty (||M - mid||), and a quick trained-vs-static head-to-head.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridges.reference import make_reference
from metrics.image_metrics import psnr, ssim

CACHE = Path("runs/_fastcache/ft3d_64_3000_200_40.pt")
T_MID = 0.5


@torch.no_grad()
def ref_mid(reference, xA, xB, fAB):
    b = xA.shape[0]
    t = torch.full((b,), T_MID, device=xA.device)
    return reference.mean(xA, xB, fAB, t)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(CACHE, map_location=device)
    te = {k: v for k, v in blob["test"].items() if torch.is_tensor(v)}
    xA, xM, xB, fAB = te["x_A"], te["x_M"], te["x_B"], te["flow_AB"]
    strata, edges = blob["strata"], blob["edges"]

    phys_r = make_reference("physics")
    brwn_r = make_reference("brownian")
    mid_p = ref_mid(phys_r, xA, xB, fAB)
    mid_b = ref_mid(brwn_r, xA, xB, fAB)

    p_ps = psnr(mid_p, xM)
    p_bs = psnr(mid_b, xM)
    gain_static = (p_ps.mean() - p_bs.mean()).item()

    # residual difficulty in [0,1] image space
    res_p = (xM - mid_p).pow(2).mean(dim=(1, 2, 3)).sqrt()
    res_b = (xM - mid_b).pow(2).mean(dim=(1, 2, 3)).sqrt()

    print("=== Same 200 test clips @ 64px (n_train=3000 cache) ===\n")
    print("STATIC prior PSNR vs GT midpoint M:")
    print(f"  physics:  {p_ps.mean():.3f} dB  (std {p_ps.std():.2f})")
    print(f"  brownian: {p_bs.mean():.3f} dB  (std {p_bs.std():.2f})")
    print(f"  physics - brownian static: {gain_static:+.3f} dB  "
          f"(physics prior is already {gain_static:.2f} dB better)\n")

    print("Residual ||M - mid|| (lower = easier refinement task):")
    print(f"  physics:  mean={res_p.mean():.4f}  median={res_p.median():.4f}")
    print(f"  brownian: mean={res_b.mean():.4f}  median={res_b.median():.4f}")
    print(f"  ratio brownian/physics: {(res_b.mean()/res_p.mean()).item():.2f}x larger residual\n")

    print("Per-stratum STATIC PSNR (physics - brownian):")
    for b_i, idx in enumerate(strata):
        if not idx:
            continue
        it = torch.tensor(idx, device=device)
        rng = (f"[{int(edges[b_i])},{int(edges[b_i+1])})"
               if b_i < len(edges) - 2 else f"[{int(edges[b_i])},inf)")
        gp = (p_ps[it].mean() - p_bs[it].mean()).item()
        print(f"  {rng:12s} n={len(idx):3d}  phys {p_ps[it].mean():.2f}  brwn {p_bs[it].mean():.2f}  "
              f"delta {gp:+.2f} dB")

  # matched fast-cycle RESULT lines (from log)
    print("\n=== Matched fast-cycle runs (wd=0.1, drop=0.2, n_train=3000, 3k steps) ===")
    rows = [
        ("physics  mse",     20.814, 20.177, 0.637, 0.724),
        ("brownian mse",     19.305, 17.860, 1.445, 1.290),
        ("physics  charb 6k", 20.694, 20.177, 0.516, 0.474),
        ("brownian charb 6k", 19.206, 17.860, 1.346, 1.044),
        ("physics  128 charb 8k", 21.124, 20.312, 0.812, 0.700),
        ("brownian 128 charb 8k", 19.235, 17.860, 1.375, 1.016),
    ]
    print(f"  {'config':22s}  trained  static   gain    hi_gain")
    for name, tr, st, g, hg in rows:
        print(f"  {name:22s}  {tr:6.3f}  {st:6.3f}  {g:+.3f}  {hg:+.3f}")
    print("\n  ABSOLUTE PSNR to GT: physics trained ~20.7-21.1 dB vs brownian ~19.2 dB")
    print("  -> physics is CLOSER to GT; brownian only wins on relative GAIN.\n")

    # ceiling analysis: how much headroom left above static?
    print("Headroom above static prior (trained - static):")
    for name, tr, st in [("physics best", 21.168, 20.312), ("brownian best", 19.305, 17.860)]:
        print(f"  {name}: {tr-st:+.3f} dB absolute improvement over its own static")
    print("\n  Ceiling to GT from static:")
    print(f"    physics static->GT gap:  {20.177 - p_ps.mean().item():.3f} dB below trained-best static")
    # actually static PSNR is vs M, trained is vs M - headroom = trained - static in same metric
    print(f"    physics: static {p_ps.mean():.2f} dB, room to improve ~{20.855-20.177:.2f} dB seen in cycles")
    print(f"    brownian: static {p_bs.mean():.2f} dB, room ~{19.305-17.860:.2f} dB — easier target\n")

    print("=== Verdict ===")
    print("1. METRIC ARTIFACT: 'gain' = trained - static uses DIFFERENT static baselines per ref.")
    print("   Brownian static is ~2.3 dB worse -> refiners get credit for 'fixing' a blurrier prior.")
    print("2. ABSOLUTE QUALITY: physics trained models score ~1.5 dB HIGHER PSNR vs GT than brownian.")
    print("3. TASK DIFFICULTY: ||M-mid|| is ~2x larger for brownian -> larger gradients, easier MSE fit.")
    print("4. H2 at fast scale: DOMINATE on gain favors brownian; H1 still holds (physics static > brownian).")
    print("5. For fair H2: compare trained_psnr directly, or normalize gain by static gap to GT.")


if __name__ == "__main__":
    main()
