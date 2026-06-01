"""Rollout diagnostic / proposed replacement for sanity gate 3.

For a small test batch (GT midpoint x_M available) compares four midpoint
estimates at t = 0.5:

  static   : reference.mean(0.5)                  -- the no-train analytic prior
  free     : full free-running Euler integration  -- what real inference does
  teacher  : Euler step evaluated at the TRUE interpolant I_t (uses GT M)
  zero_xt  : free-running with x_t zeroed          -- old gate-3 probe

Diagnosis:
  teacher >> free  -> model fit the on-path drift but cannot reproduce it from its
                      own running state. The bridge noise (sigma) is too small, so
                      x_t = I_t + sigma*sqrt(t(1-t))*eps leaks the target M.
  free   <  static -> trained sampler is worse than the analytic prior (FAIL).

GATE (proposed): require (teacher - free) < ~3 dB AND free >= static - 0.5 dB.

Usage:
  python exps/diag_rollout.py <run_name> [n_samples]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps._config import load_config
from exps._inference import build_model_from_cfg, _load_ckpt
from data import FlyingThings3D
from bridges.reference import make_reference
from bridges import interpolant
from metrics.image_metrics import psnr as psnr_fn

NUM_STEPS = 16
T_EVAL = 0.5


@torch.no_grad()
def rollout(model, reference, batch, device):
    xA = batch["x_A"].to(device)
    xB = batch["x_B"].to(device)
    xM = batch["x_M"].to(device)
    fAB = batch["flow_AB"].to(device)
    b = xA.shape[0]
    t_grid = torch.linspace(0.0, 1.0, NUM_STEPS + 1, device=device)
    zeros = torch.zeros(b, device=device)

    def drift(x, t_scalar):
        t_b = torch.full((b,), t_scalar, device=device)
        u_ref = reference.velocity(xA, xB, fAB, t_b)
        return u_ref + model(x, t_b, xA, xB, u_ref)

    x_free = reference.mean(xA, xB, fAB, zeros)
    x_zero = reference.mean(xA, xB, fAB, zeros)
    out = {}
    for k in range(NUM_STEPS):
        t = float(t_grid[k]); t_next = float(t_grid[k + 1]); dt = t_next - t
        t_b = torch.full((b,), t, device=device)
        I_t = interpolant.interpolate(xA, xM, xB, t_b)

        d_free = drift(x_free, t)
        d_tf = drift(I_t, t)
        u_ref = reference.velocity(xA, xB, fAB, t_b)
        d_zero = u_ref + model(torch.zeros_like(x_zero), t_b, xA, xB, u_ref)

        if t <= T_EVAL <= t_next:
            a = (T_EVAL - t) / max(dt, 1e-8)
            out["free"] = x_free + d_free * (a * dt)
            out["teacher"] = I_t + d_tf * (a * dt)
            out["zero_xt"] = x_zero + d_zero * (a * dt)
            break
        x_free = x_free + d_free * dt
        x_zero = x_zero + d_zero * dt

    out["static"] = reference.mean(xA, xB, fAB, torch.full((b,), T_EVAL, device=device))
    out["xM"] = xM
    return out


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "physics_v2_s0"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48

    cfg = load_config("configs/base.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path("runs") / run_name
    run_cfg = json.loads((run_dir / "config_used.yaml").read_text())
    reference = make_reference(run_cfg["bridge"]["reference"])
    model = build_model_from_cfg(run_cfg, run_cfg["data"]["size"]).to(device)
    ckpt = run_dir / "ckpt_final.pt"
    if not ckpt.exists():
        ckpt = sorted(run_dir.glob("ckpt_*.pt"))[-1]
    _load_ckpt(model, str(ckpt), device)

    ds = FlyingThings3D(root=cfg["data"]["root"], split="TEST",
                        size=cfg["data"]["size"], min_flow=cfg["data"]["min_flow"],
                        max_clips=n, pass_name=cfg["data"]["pass"], seed=0)
    dl = DataLoader(ds, batch_size=8, num_workers=2)

    acc = {k: [] for k in ["static", "free", "teacher", "zero_xt"]}
    for batch in dl:
        r = rollout(model, reference, batch, device)
        for k in acc:
            acc[k].append(psnr_fn(r[k], r["xM"]).cpu().numpy())
    m = {k: float(np.concatenate(v).mean()) for k, v in acc.items()}

    print(f"\n=== {run_name}  ckpt={ckpt.name}  sigma={run_cfg['bridge']['sigma']}  n={n} ===")
    for k in ["static", "free", "teacher", "zero_xt"]:
        print(f"  {k:9s} PSNR = {m[k]:6.2f} dB")
    gap = m["teacher"] - m["free"]
    print(f"\n  teacher - free = {gap:5.2f} dB   (large => x_t leaks M / exposure bias)")
    print(f"  free   - static = {m['free'] - m['static']:5.2f} dB   (negative => worse than no-train prior)")
    passed = gap < 3.0 and m["free"] >= m["static"] - 0.5
    print(f"  GATE: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
