#!/usr/bin/env python3
"""
Static-prior evaluation for Stage 2 Stage-A gates (A1 / A2).

No training. Loads a small TEST set WITH half-flows (F_{A->M}, F_{M->B}) and
compares the midpoint PSNR of several physics-reference variants against x_M:

  physics      (V1) -- Stage-1 constant-velocity warp
  physics_v2   (V2) -- acceleration-aware warp           (A1)
  physics_v3   (V3) -- occlusion-aware blend on top of V2 (A2)

Reports aggregate + per-stratum PSNR for each, and pairwise deltas. Caches the
loaded tensors (incl. half-flows) to runs/_fastcache so re-runs are instant.

Usage:
  python3 exps/eval_static.py --size 128 --n_test 200 --refs physics physics_v2
  python3 exps/eval_static.py --size 128 --n_test 200 --refs physics physics_v2 physics_v3 \
      --gate_against physics_v2 --gate_ref physics_v3 --min_delta 0.0 --min_abs 21.0
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import FlyingThings3D, stratify_by_motion
from bridges.reference import make_reference
from metrics.image_metrics import psnr, ssim

CACHE_DIR = Path("runs/_fastcache")
CACHE_LOAD_SIZE = 256          # match fast_cycle: load native 256, then downsample
T_MID = 0.5
EDGES = [40, 60, 100, 1000000000]
import torch.nn.functional as F


def _downsample_hf(blob, size):
    """Downsample images + flows from CACHE_LOAD_SIZE to `size`, scaling flow
    magnitudes by the same factor. Mirrors exps/fast_cycle._downsample so the
    physics-prior PSNR matches the rest of the system (V1 ~ 20.3 dB @128)."""
    if size == CACHE_LOAD_SIZE:
        return blob
    sc = size / CACHE_LOAD_SIZE
    for k in ("x_A", "x_M", "x_B"):
        blob[k] = F.interpolate(blob[k], size=(size, size), mode="bilinear",
                                align_corners=False)
    for k in ("flow_AB", "flow_AM", "flow_MB"):
        blob[k] = F.interpolate(blob[k], size=(size, size), mode="bilinear",
                                align_corners=False) * sc
    return blob


def build_static_cache(size, n_test, min_flow, root, pass_name):
    """Load n_test TEST clips WITH half-flows at 256, downsample to `size`."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = CACHE_DIR / f"static_hf_{size}_{n_test}_{int(min_flow)}.pt"
    if key.exists():
        return torch.load(key)
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ds = FlyingThings3D(root=root, split="TEST", size=CACHE_LOAD_SIZE,
                        min_flow=min_flow, max_clips=n_test, pass_name=pass_name,
                        seed=0, return_half_flows=True)
    keys = ["x_A", "x_M", "x_B", "flow_AB", "flow_AM", "flow_MB"]
    acc = {k: [] for k in keys}
    dl = DataLoader(ds, batch_size=8, num_workers=8, shuffle=False)
    for b in tqdm(dl, desc=f"static cache TEST@{CACHE_LOAD_SIZE}", total=(len(ds) + 7) // 8):
        for k in keys:
            acc[k].append(b[k])
    blob = {k: torch.cat(v) for k, v in acc.items()}
    blob = _downsample_hf(blob, size)
    blob["strata"] = stratify_by_motion(ds, EDGES)
    blob["edges"] = EDGES
    torch.save(blob, key)
    return blob


@torch.no_grad()
def eval_reference(ref_name, te, device, bs=64):
    reference = make_reference(ref_name)
    xA, xB, xM = te["x_A"], te["x_B"], te["x_M"]
    fAB, fAM, fMB = te["flow_AB"], te["flow_AM"], te["flow_MB"]
    n = xA.shape[0]
    psnrs, ssims = [], []
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        t_b = torch.full((xA[sl].shape[0],), T_MID, device=device)
        mid = reference.mean(xA[sl], xB[sl], fAB[sl], t_b,
                             flow_AM=fAM[sl], flow_MB=fMB[sl])
        psnrs.append(psnr(mid, xM[sl]))
        ssims.append(ssim(mid, xM[sl]))
    return torch.cat(psnrs), torch.cat(ssims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--min_flow", type=float, default=40.0)
    ap.add_argument("--root", default="/home/ubuntu/datasets/FlyingThings3D")
    ap.add_argument("--pass_name", default="cleanpass")
    ap.add_argument("--refs", nargs="+", default=["physics", "physics_v2"])
    ap.add_argument("--gate_ref", default=None, help="reference that must win")
    ap.add_argument("--gate_against", default=None, help="reference it must beat")
    ap.add_argument("--min_delta", type=float, default=0.0,
                    help="gate_ref PSNR - gate_against PSNR must exceed this")
    ap.add_argument("--min_abs", type=float, default=None,
                    help="gate_ref absolute PSNR must exceed this")
    ap.add_argument("--tag", default="A?", help="label for the RESULTS.md line")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    blob = build_static_cache(args.size, args.n_test, args.min_flow,
                              args.root, args.pass_name)
    te = {k: v.to(device) for k, v in blob.items() if torch.is_tensor(v)}
    strata, edges = blob["strata"], blob["edges"]
    n = te["x_A"].shape[0]
    print(f"[static] loaded {n} clips @ {args.size}px ({time.time()-t0:.1f}s)")

    results = {}
    for ref in args.refs:
        p, s = eval_reference(ref, te, device)
        results[ref] = p
        hi = float("nan")
        for b_i, idx in enumerate(strata):
            if idx and b_i == len(strata) - 1:
                hi = p[torch.tensor(idx, device=device)].mean().item()
        print(f"  {ref:14s} PSNR={p.mean().item():.3f}  SSIM={s.mean().item():.4f}  "
              f"hi[100,inf)={hi:.3f}")
        for b_i, idx in enumerate(strata):
            if not idx:
                continue
            rng = (f"[{int(edges[b_i])},{int(edges[b_i+1])})"
                   if b_i < len(edges) - 2 else f"[{int(edges[b_i])},inf)")
            it = torch.tensor(idx, device=device)
            print(f"      {rng:12s} n={len(idx):4d}  PSNR={p[it].mean().item():.3f}")

    # pairwise deltas
    refs = args.refs
    print("\n  pairwise mean PSNR delta (row - col):")
    for r in refs:
        row = "  ".join(f"{(results[r].mean()-results[c].mean()).item():+.3f}" for c in refs)
        print(f"    {r:14s} {row}")

    verdict = "n/a"
    if args.gate_ref and args.gate_against:
        d = (results[args.gate_ref].mean() - results[args.gate_against].mean()).item()
        abs_ok = True
        abs_val = results[args.gate_ref].mean().item()
        if args.min_abs is not None:
            abs_ok = abs_val > args.min_abs
        delta_ok = d > args.min_delta
        passed = delta_ok and abs_ok
        verdict = "PASS" if passed else "FAIL"
        cond = f"{args.gate_ref}-{args.gate_against}={d:+.3f}>{args.min_delta}"
        if args.min_abs is not None:
            cond += f" AND abs={abs_val:.3f}>{args.min_abs}"
        print(f"\n  GATE [{args.tag}] {cond}  -> {verdict}")
        print(f"STATIC_GATE tag={args.tag} ref={args.gate_ref} vs={args.gate_against} "
              f"delta={d:+.3f} abs={abs_val:.3f} min_delta={args.min_delta} "
              f"min_abs={args.min_abs} verdict={verdict}")
    return verdict


if __name__ == "__main__":
    main()
