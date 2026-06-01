"""
Full-scale fair paired H2 (refine_resid): prereg metrics on full test set.

Same fairness guarantees as fair_pair.py, plus:
  * n_test=1000 (default), all four metrics, per-stratum tables
  * JSON artifact under runs/fair_full/
  * FULL_PAIRED_RESULT line for cycles.log / summary scripts

Usage:
  python exps/fair_pair_full.py --size 256 --steps 12000 --seed 0
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridges.reference import make_reference
from exps.fair_pair import train_one, eval_ref
from exps.fast_cycle import build_cache, precompute_ref, _make_model
from metrics.image_metrics import psnr, ssim, lpips_metric
from metrics.flow_metrics import epe_against_gt_flow

T_MID = 0.5
OUT_ROOT = Path("runs/fair_full")
STRATA_EDGES = [40, 60, 100, 1000000000]


@torch.no_grad()
def eval_full_metrics(model, te, te_mid, te_uref, resid, device, eval_batch=16):
    """Per-clip PSNR/SSIM/LPIPS/EPE for trained and static prior."""
    model.eval()
    xA, xB, xM = te["x_A"], te["x_B"], te["x_M"]
    n = xA.shape[0]
    keys = ("psnr", "ssim", "lpips", "epe")
    tr = {k: [] for k in keys}
    st = {k: [] for k in keys}
    for s in range(0, n, eval_batch):
        sl = slice(s, min(s + eval_batch, n))
        t_b = torch.full((xA[sl].shape[0],), T_MID, device=device)
        out = model(te_mid[sl], t_b, xA[sl], xB[sl], te_uref[sl])
        pred = te_mid[sl] + out if resid else out
        ref_mid = te_mid[sl]
        tr["psnr"].append(psnr(pred, xM[sl]))
        tr["ssim"].append(ssim(pred, xM[sl]))
        tr["lpips"].append(lpips_metric(pred, xM[sl]))
        tr["epe"].append(epe_against_gt_flow(pred, xA[sl], te["flow_AB"][sl]))
        st["psnr"].append(psnr(ref_mid, xM[sl]))
        st["ssim"].append(ssim(ref_mid, xM[sl]))
        st["lpips"].append(lpips_metric(ref_mid, xM[sl]))
        st["epe"].append(epe_against_gt_flow(ref_mid, xA[sl], te["flow_AB"][sl]))
    trained = {k: torch.cat(v) for k, v in tr.items()}
    static = {k: torch.cat(v) for k, v in st.items()}
    return trained, static


def _agg(tensors, idx=None):
    out = {}
    for k, v in tensors.items():
        x = v if idx is None else v[idx]
        out[k] = float(x.mean().item())
    return out


def _strata_report(trained, static, strata, edges, device):
    rep = {}
    for b_i, idx in enumerate(strata):
        if not idx:
            continue
        it = torch.tensor(idx, device=device)
        rng = (f"[{int(edges[b_i])},{int(edges[b_i+1])})"
               if b_i < len(edges) - 2 else f"[{int(edges[b_i])},inf)")
        rep[f"stratum_{b_i}"] = {
            "range_px": rng,
            "n": len(idx),
            "trained": _agg(trained, it),
            "static": _agg(static, it),
        }
    return rep


def prereg_verdict(phys_tr, brwn_tr, mean_delta):
    """H2-pilot gate on refine fair pair (PSNR primary + direction checks)."""
    reasons = []
    if mean_delta < 0.3:
        reasons.append(f"PSNR mean_delta {mean_delta:+.3f} < 0.3 dB")
    for m, higher_better in [("ssim", True), ("lpips", False), ("epe", False)]:
        dp = phys_tr[m] - brwn_tr[m]
        ok = dp > 0 if higher_better else dp < 0
        if not ok:
            reasons.append(f"{m} wrong direction (delta={dp:+.4f})")
    passed = len(reasons) == 0
    return {"pass": passed, "reasons": reasons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="refine_resid")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--input_noise", type=float, default=0.0)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/home/ubuntu/datasets/FlyingThings3D")
    ap.add_argument("--pass_name", default="cleanpass")
    ap.add_argument("--min_flow", type=float, default=40.0)
    ap.add_argument("--loss", default="charb", choices=["mse", "l1", "charb"])
    ap.add_argument("--lr_schedule", default="cosine", choices=["none", "cosine"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    args.reference = "physics"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resid = args.variant == "refine_resid"

    tag = (f"s{args.size}_st{args.steps}_ntr{args.n_train}_"
           f"wd{args.weight_decay}_drop{args.dropout}_{args.loss}_seed{args.seed}")
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[full] building cache size={args.size} train={args.n_train} test={args.n_test}...")
    blob = build_cache(args.size, args.n_train, args.n_test, args.min_flow,
                       args.root, args.pass_name)
    tr = {k: v.to(device) for k, v in blob["train"].items() if torch.is_tensor(v)}
    te = {k: v.to(device) for k, v in blob["test"].items() if torch.is_tensor(v)}
    strata, edges = blob["strata"], blob["edges"]

    ref_p, ref_b = make_reference("physics"), make_reference("brownian")
    tr_mid_p, tr_uref_p = precompute_ref(ref_p, tr["x_A"], tr["x_B"], tr["flow_AB"])
    tr_mid_b, tr_uref_b = precompute_ref(ref_b, tr["x_A"], tr["x_B"], tr["flow_AB"])
    te_mid_p, te_uref_p = precompute_ref(ref_p, te["x_A"], te["x_B"], te["flow_AB"])
    te_mid_b, te_uref_b = precompute_ref(ref_b, te["x_A"], te["x_B"], te["flow_AB"])
    print(f"[full] cache ok ({time.time()-t0:.1f}s)")

    base = _make_model(args, device)
    init_state = copy.deepcopy(base.state_dict())
    n = tr["x_A"].shape[0]
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    train_idx = [torch.randint(0, n, (args.batch,), generator=gen, device=device)
                 for _ in range(args.steps)]

    print("[full] training physics (paired batches)...")
    t1 = time.time()
    m_p = train_one(args, "physics", tr, tr_mid_p, tr_uref_p, device, init_state, train_idx)
    print(f"[full] training brownian... ({time.time()-t1:.0f}s)")
    m_b = train_one(args, "brownian", tr, tr_mid_b, tr_uref_b, device, init_state, train_idx)

    print("[full] full metrics eval (physics)...")
    p_tr, p_st = eval_full_metrics(m_p, te, te_mid_p, te_uref_p, resid, device, args.eval_batch)
    print("[full] full metrics eval (brownian)...")
    b_tr, b_st = eval_full_metrics(m_b, te, te_mid_b, te_uref_b, resid, device, args.eval_batch)

    delta_psnr = p_tr["psnr"] - b_tr["psnr"]
    wins = int((delta_psnr > 0).sum().item())
    mean_d = delta_psnr.mean().item()

    hi_delta = float("nan")
    for b_i, idx in enumerate(strata):
        if idx and b_i == len(strata) - 1:
            it = torch.tensor(idx, device=device)
            hi_delta = (p_tr["psnr"][it].mean() - b_tr["psnr"][it].mean()).item()
            break

    phys_agg = _agg(p_tr)
    brwn_agg = _agg(b_tr)
    h2 = "physics" if mean_d > 0 else "brownian"
    gate = prereg_verdict(phys_agg, brwn_agg, mean_d)

    result = {
        "config": vars(args),
        "physics": {
            "aggregate": {"trained": phys_agg, "static": _agg(p_st)},
            "strata": _strata_report(p_tr, p_st, strata, edges, device),
        },
        "brownian": {
            "aggregate": {"trained": brwn_agg, "static": _agg(b_st)},
            "strata": _strata_report(b_tr, b_st, strata, edges, device),
        },
        "paired": {
            "mean_delta_psnr": mean_d,
            "hi_delta_psnr": hi_delta,
            "physics_wins": wins,
            "n_test": int(delta_psnr.numel()),
            "verdict": h2,
        },
        "prereg_h2_refine": gate,
    }
    json_path = out_dir / "fair_full_eval.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"\n=== FULL FAIR PAIRED  size={args.size} seed={args.seed} steps={args.steps} ===")
    print(f"  physics  PSNR={phys_agg['psnr']:.3f} SSIM={phys_agg['ssim']:.4f} "
          f"LPIPS={phys_agg['lpips']:.4f} EPE={phys_agg['epe']:.3f}")
    print(f"  brownian PSNR={brwn_agg['psnr']:.3f} SSIM={brwn_agg['ssim']:.4f} "
          f"LPIPS={brwn_agg['lpips']:.4f} EPE={brwn_agg['epe']:.3f}")
    print(f"  PAIRED mean_delta={mean_d:+.3f} dB  hi_delta={hi_delta:+.3f}  "
          f"wins {wins}/{delta_psnr.numel()}")
    print(f"  prereg H2 refine: {'PASS' if gate['pass'] else 'FAIL'}  {gate['reasons']}")
    print(f"  saved {json_path}")

    # Lines for cycles.log grep
    print(f"PAIRED_RESULT size={args.size} steps={args.steps} n_train={args.n_train} "
          f"wd={args.weight_decay} drop={args.dropout} loss={args.loss} lr={args.lr_schedule} | "
          f"phys_tr={phys_agg['psnr']:.3f} brwn_tr={brwn_agg['psnr']:.3f} "
          f"mean_delta={mean_d:+.3f} hi_delta={hi_delta:+.3f} "
          f"phys_wins={wins}/{delta_psnr.numel()} verdict={h2}")
    print(f"FULL_PAIRED_RESULT size={args.size} seed={args.seed} steps={args.steps} "
          f"n_test={args.n_test} | mean_delta={mean_d:+.3f} hi_delta={hi_delta:+.3f} "
          f"phys_psnr={phys_agg['psnr']:.3f} brwn_psnr={brwn_agg['psnr']:.3f} "
          f"phys_ssim={phys_agg['ssim']:.4f} brwn_ssim={brwn_agg['ssim']:.4f} "
          f"phys_lpips={phys_agg['lpips']:.4f} brwn_lpips={brwn_agg['lpips']:.4f} "
          f"phys_epe={phys_agg['epe']:.3f} brwn_epe={brwn_agg['epe']:.3f} "
          f"wins={wins}/{delta_psnr.numel()} prereg={'PASS' if gate['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
