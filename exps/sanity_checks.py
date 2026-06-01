"""
Sanity checks — gates from prereg.md.

Run before any training or evaluation. Each gate must pass; failures are loud.

Gates:
  1. Flow chain consistency.
  2. Bridge integrator correctness (1D Brownian bridge analytic check).
  3. x_t-usage at inference (run after training; included here for reference).
  4. Identity case (x_A == x_B implies M-hat == x_A).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# allow `python exps/sanity_checks.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps._config import load_config
from data import FlyingThings3D
from data.flow_utils import warp_consistency_error
from bridges.reference import make_reference
from bridges import interpolant
from bridges.integrator import toy_1d_brownian_bridge_test
from model import UNet


# ----------------------------- gate 1 -----------------------------

def gate1_flow_chain(cfg: dict, n_samples: int = 50, max_ratio: float = 0.7):
    print("[gate 1] Flow chain consistency …")
    from data.flow_utils import backward_warp
    ds = FlyingThings3D(
        root=cfg["data"]["root"], split="TEST", size=cfg["data"]["size"],
        min_flow=cfg["data"]["min_flow"], max_clips=n_samples,
        pass_name=cfg["data"]["pass"], seed=0,
    )
    warp_errs, base_errs = [], []
    for i in range(min(n_samples, len(ds))):
        s = ds[i]
        xA = (s["x_A"].unsqueeze(0) + 1.0) * 0.5
        xB = (s["x_B"].unsqueeze(0) + 1.0) * 0.5
        warped = backward_warp(xB, s["flow_AB"].unsqueeze(0), padding_mode="border")
        warp_errs.append((warped - xA).abs().mean().item())
        base_errs.append((xB - xA).abs().mean().item())
    warp_mean = float(np.mean(warp_errs)); base_mean = float(np.mean(base_errs))
    ratio = warp_mean / max(base_mean, 1e-6)
    print(f"  warp err {warp_mean:.4f} | no-warp baseline {base_mean:.4f} | ratio {ratio:.3f}")
    if ratio < max_ratio:
        print(f"  PASS (ratio {ratio:.3f} < {max_ratio}) — flow explains motion")
        return True
    print(f"  FAIL (ratio {ratio:.3f} >= {max_ratio})")
    return False


# ----------------------------- gate 2 -----------------------------

def gate2_integrator(tol: float = 0.02):
    print("[gate 2] Bridge integrator (1D Brownian bridge) …")
    r = toy_1d_brownian_bridge_test(n_paths=20000, n_steps=200, sigma=1.0, seed=0)
    print(f"  mean: emp={r['mean_emp']:.4f}  ana={r['mean_ana']:.4f}  err={r['mean_err']:.4f}")
    print(f"  var:  emp={r['var_emp']:.4f}  ana={r['var_ana']:.4f}  err={r['var_err']:.4f}")
    ok = r['mean_err'] < tol and r['var_err'] < tol
    print("  PASS" if ok else "  FAIL")
    return ok


# ----------------------------- gate 3 -----------------------------

def gate3_xt_usage(cfg: dict, ckpt_path: str, n_samples: int = 32,
                   min_drop_db: float = 2.0):
    """Run after training — replaces x_t with noise and checks PSNR drops."""
    print(f"[gate 3] x_t-usage ablation on {ckpt_path} …")
    from exps._inference import run_inference
    psnr_normal = run_inference(cfg, ckpt_path, n_samples=n_samples,
                                xt_corruption=None)
    psnr_zeroxt = run_inference(cfg, ckpt_path, n_samples=n_samples,
                                xt_corruption="zero")
    drop = float(np.mean(psnr_normal) - np.mean(psnr_zeroxt))
    print(f"  PSNR normal: {np.mean(psnr_normal):.3f}  zero-x_t: {np.mean(psnr_zeroxt):.3f}  drop: {drop:.3f} dB")
    if drop >= min_drop_db:
        print(f"  PASS (drop >= {min_drop_db} dB)")
        return True
    print(f"  FAIL — network is not using x_t (drop < {min_drop_db} dB)")
    return False


# ----------------------------- gate 4 -----------------------------

def gate4_identity(cfg: dict, ckpt_path: str, threshold_db: float = 30.0):
    """When x_A == x_B (zero-motion sample), M-hat should equal x_A."""
    print(f"[gate 4] identity case on {ckpt_path} …")
    from exps._inference import run_inference_identity
    psnr_val = run_inference_identity(cfg, ckpt_path, n_samples=8)
    print(f"  PSNR (M-hat vs x_A) on synthetic identity inputs: {psnr_val:.2f} dB")
    if psnr_val >= threshold_db:
        print(f"  PASS (>= {threshold_db} dB)")
        return True
    print(f"  FAIL — model corrupts identity inputs")
    return False


# ----------------------------- driver -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="if given, also run post-training gates 3 and 4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = {}
    results["gate1_flow_chain"] = gate1_flow_chain(cfg)
    results["gate2_integrator"] = gate2_integrator()
    if args.ckpt is not None:
        results["gate3_xt_usage"] = gate3_xt_usage(cfg, args.ckpt)
        results["gate4_identity"] = gate4_identity(cfg, args.ckpt)

    print("\n=== Sanity summary ===")
    for k, v in results.items():
        print(f"  {k:24s}  {'PASS' if v else 'FAIL'}")
    if not all(results.values()):
        print("\nOne or more gates failed. Pilot results from this state are INVALID.")
        sys.exit(1)
    print("\nAll gates passed.")


if __name__ == "__main__":
    main()
