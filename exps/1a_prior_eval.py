"""
Experiment 1a — static prior test.

For each test triplet (x_A, x_M, x_B, flow_AB):
  - mu_brownian = 0.5 (x_A + x_B)
  - mu_physics  = bidirectional flow-warped blend at t=0.5
Compute PSNR / SSIM / LPIPS / EPE  vs  x_M.

Report aggregate and per-stratum (motion bin).

Also report the bridge-matching residual norm at random t:
    || dot{I}_t  -  u^ref(I_t, t) ||_2
which is the *target* the network would have to learn -- a smaller value
means the reference already explains the motion. This is independent of any
network so it predicts H2 outcome.

No GPU strictly required for this; runs on CPU in minutes for ~1k samples.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps._config import load_config
from data import FlyingThings3D, stratify_by_motion
from bridges.reference import BrownianReference, PhysicsReference
from bridges import interpolant
from metrics.image_metrics import psnr, ssim, lpips_metric
from metrics.flow_metrics import epe_against_gt_flow, flow_residual_norm


@torch.no_grad()
def evaluate_reference(ref, ds, batch_size: int, device: str,
                       n_residual_t: int = 4):
    dl = DataLoader(ds, batch_size=batch_size, num_workers=ds.items and 4)
    per_psnr, per_ssim, per_lpips, per_epe, per_res = [], [], [], [], []
    half = torch.tensor([0.5], device=device)

    for batch in dl:
        xA = batch["x_A"].to(device); xM = batch["x_M"].to(device)
        xB = batch["x_B"].to(device); fAB = batch["flow_AB"].to(device)
        t05 = half.expand(xA.shape[0])

        mu = ref.mean(xA, xB, fAB, t05)
        per_psnr.append(psnr(mu, xM).cpu().numpy())
        per_ssim.append(ssim(mu, xM).cpu().numpy())
        per_lpips.append(lpips_metric(mu, xM).cpu().numpy())
        per_epe.append(epe_against_gt_flow(mu, xA, fAB).cpu().numpy())

        # residual norm at random t (averaged over n_residual_t)
        rs = []
        for _ in range(n_residual_t):
            t = torch.rand(xA.shape[0], device=device).clamp(0.05, 0.95)
            dot_I = interpolant.velocity(xA, xM, xB, t)
            u_ref = ref.velocity(xA, xB, fAB, t)
            rs.append(flow_residual_norm(dot_I, u_ref).cpu().numpy())
        per_res.append(np.mean(np.stack(rs), axis=0))

    return dict(
        psnr=np.concatenate(per_psnr),
        ssim=np.concatenate(per_ssim),
        lpips=np.concatenate(per_lpips),
        epe=np.concatenate(per_epe),
        residual_norm=np.concatenate(per_res),
    )


def summarize(metric, indices=None):
    arr = metric if indices is None else metric[indices]
    return dict(mean=float(arr.mean()), std=float(arr.std()), n=int(len(arr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="runs/1a_prior_eval/results.json")
    ap.add_argument("--n_test", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = FlyingThings3D(
        root=cfg["data"]["root"], split="TEST",
        size=cfg["data"]["size"], min_flow=cfg["data"]["min_flow"],
        max_clips=args.n_test or cfg["data"]["test_clips"],
        pass_name=cfg["data"]["pass"], seed=0,
    )
    print(f"Test set: {len(ds)} triplets")

    strata = stratify_by_motion(ds, cfg["eval"]["strata_edges"])
    print(f"Strata sizes: {[len(s) for s in strata]}")

    refs = {"brownian": BrownianReference(), "physics": PhysicsReference()}
    all_results = {}
    for name, ref in refs.items():
        print(f"\n--- {name} reference ---")
        r = evaluate_reference(ref, ds, batch_size=8, device=device)
        for k, v in r.items():
            print(f"  {k:14s}  mean={v.mean():.4f}  std={v.std():.4f}")
        per_stratum = {}
        for b, idx in enumerate(strata):
            if len(idx) == 0:
                continue
            per_stratum[f"stratum_{b}"] = {
                k: summarize(r[k], idx) for k in r
            }
        all_results[name] = dict(
            aggregate={k: summarize(r[k]) for k in r},
            strata=per_stratum,
            n=len(ds),
        )

    # H1 evaluation
    agg_b = all_results["brownian"]["aggregate"]
    agg_p = all_results["physics"]["aggregate"]
    h1 = dict(
        psnr_pass=agg_p["psnr"]["mean"] > agg_b["psnr"]["mean"],
        ssim_pass=agg_p["ssim"]["mean"] > agg_b["ssim"]["mean"],
        lpips_pass=agg_p["lpips"]["mean"] < agg_b["lpips"]["mean"],
        epe_pass=agg_p["epe"]["mean"] < agg_b["epe"]["mean"],
    )
    monotone = True
    last_gap = None
    for b in range(len(strata)):
        key = f"stratum_{b}"
        if key in all_results["physics"]["strata"]:
            gap = (all_results["physics"]["strata"][key]["psnr"]["mean"]
                   - all_results["brownian"]["strata"][key]["psnr"]["mean"])
            if last_gap is not None and gap < last_gap:
                monotone = False
            last_gap = gap
    h1["psnr_gap_monotone_in_motion"] = monotone
    h1["overall_pass"] = all(h1[k] for k in
                             ("psnr_pass", "ssim_pass", "lpips_pass", "epe_pass"))

    all_results["H1"] = h1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== H1 (static prior) ===")
    for k, v in h1.items():
        print(f"  {k:32s}  {v}")
    if not h1["overall_pass"]:
        print("\nH1 FAILED. Per prereg.md, abort the pilot.")
        sys.exit(2)
    print("\nH1 passed. Proceed to 1b training.")


if __name__ == "__main__":
    main()
