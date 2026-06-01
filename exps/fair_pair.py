"""
Fair paired H2 comparison: physics vs brownian with IDENTICAL conditions.

Guarantees:
  * Same train/test triplets (shared cache, same seed)
  * Same UNet architecture and hyperparameters
  * Same weight initialization (brownian starts from physics init copy)
  * Same training batch indices each step (paired SGD)
  * Eval on same test clips; report per-sample paired PSNR delta

Only intentional difference: reference path (mid, u_ref, residual target).

Usage:
  python exps/fair_pair.py --size 128 --steps 8000 --n_train 6000 \\
      --weight_decay 0.1 --dropout 0.2 --loss charb --lr_schedule cosine
"""
from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridges.reference import make_reference
from exps.fast_cycle import (
    T_MID, build_cache, precompute_ref, _make_model, _loss_fn, EMA,
)
from metrics.image_metrics import psnr


def train_one(args, reference_name, tr, tr_mid, tr_uref, device,
              init_state, train_idx):
    """Train one model; returns eval-ready model (EMA weights if enabled)."""
    args_ref = copy.copy(args)
    args_ref.reference = reference_name
    resid = args.variant == "refine_resid"
    reference = make_reference(reference_name)

    model = _make_model(args, device)
    model.load_state_dict(init_state, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.05)
        if args.lr_schedule == "cosine" else None)
    loss_fn = _loss_fn(args.loss)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    model.train()
    for idx in train_idx:
        xA, xB, xM = tr["x_A"][idx], tr["x_B"][idx], tr["x_M"][idx]
        mid, u = tr_mid[idx], tr_uref[idx]
        x_in = mid + args.input_noise * torch.randn_like(mid) if args.input_noise > 0 else mid
        target = (xM - mid) if resid else xM
        t_b = torch.full((args.batch,), T_MID, device=device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred = model(x_in, t_b, xA, xB, u)
            loss = loss_fn(pred, target)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        if ema is not None:
            ema.update(model)
    if ema is not None:
        model.load_state_dict(ema.shadow, strict=True)
    return model


def default_eval_batch(size: int) -> int:
    """Eval micro-batch: 64@64px is fine; 256@256px needs small batches."""
    if size <= 64:
        return 64
    if size <= 128:
        return 16
    return 4


@torch.no_grad()
def eval_ref(model, te, te_mid, te_uref, resid, device, eval_batch: int = 64):
    model.eval()
    xA, xB, xM = te["x_A"], te["x_B"], te["x_M"]
    preds = []
    bs = max(1, eval_batch)
    for s in range(0, xA.shape[0], bs):
        sl = slice(s, s + bs)
        t_b = torch.full((xA[sl].shape[0],), T_MID, device=device)
        out = model(te_mid[sl], t_b, xA[sl], xB[sl], te_uref[sl])
        preds.append(te_mid[sl] + out if resid else out)
    pred = torch.cat(preds)
    return psnr(pred, xM), psnr(te_mid, xM), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="refine_resid")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=None,
                    help="eval micro-batch (default: 64@64px, 16@128, 4@256)")
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--input_noise", type=float, default=0.0)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/home/ubuntu/datasets/FlyingThings3D")
    ap.add_argument("--pass_name", default="cleanpass")
    ap.add_argument("--min_flow", type=float, default=40.0)
    ap.add_argument("--loss", default="charb", choices=["mse", "l1", "charb"])
    ap.add_argument("--lr_schedule", default="cosine", choices=["none", "cosine"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    args = ap.parse_args()
    args.reference = "physics"  # placeholder for _make_model
    if args.eval_batch is None:
        args.eval_batch = default_eval_batch(args.size)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resid = args.variant == "refine_resid"

    t0 = time.time()
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
    print(f"[fair] cache ok train={tr['x_A'].shape[0]} test={len(te['x_A'])} "
          f"size={args.size} ({time.time()-t0:.1f}s)")

    # Shared init + identical batch order for both models
    base = _make_model(args, device)
    init_state = copy.deepcopy(base.state_dict())
    n = tr["x_A"].shape[0]
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    train_idx = [torch.randint(0, n, (args.batch,), generator=gen, device=device)
                 for _ in range(args.steps)]

    eb = args.eval_batch
    print(f"[fair] train_batch={args.batch} eval_batch={eb}")

    print("[fair] training physics (paired batches)...")
    t1 = time.time()
    m_p = train_one(args, "physics", tr, tr_mid_p, tr_uref_p, device, init_state, train_idx)
    p_tr_p, p_st_p, _ = eval_ref(m_p, te, te_mid_p, te_uref_p, resid, device, eb)
    del m_p
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[fair] training brownian (same init, same batches)... ({time.time()-t1:.0f}s)")
    m_b = train_one(args, "brownian", tr, tr_mid_b, tr_uref_b, device, init_state, train_idx)
    p_tr_b, p_st_b, _ = eval_ref(m_b, te, te_mid_b, te_uref_b, resid, device, eb)
    del m_b
    if device == "cuda":
        torch.cuda.empty_cache()
    delta = p_tr_p - p_tr_b
    wins = int((delta > 0).sum().item())
    n_test = len(delta)

    hi_delta = float("nan")
    for b_i, idx in enumerate(strata):
        if idx and b_i == len(strata) - 1:
            it = torch.tensor(idx, device=device)
            hi_delta = (p_tr_p[it].mean() - p_tr_b[it].mean()).item()
            break

    mean_d = delta.mean().item()
    h2 = "physics" if mean_d > 0 else "brownian"
    print(f"\n=== FAIR PAIRED H2  size={args.size} steps={args.steps} n_train={args.n_train} "
          f"wd={args.weight_decay} drop={args.dropout} loss={args.loss} ===")
    print(f"  physics  trained={p_tr_p.mean():.3f}  static={p_st_p.mean():.3f}  "
          f"gain={(p_tr_p.mean()-p_st_p.mean()).item():+.3f}")
    print(f"  brownian trained={p_tr_b.mean():.3f}  static={p_st_b.mean():.3f}  "
          f"gain={(p_tr_b.mean()-p_st_b.mean()).item():+.3f}")
    print(f"  PAIRED mean_delta(physics-brownian)={mean_d:+.3f} dB  "
          f"physics_wins {wins}/{n_test} ({100*wins/n_test:.1f}%)  hi_delta={hi_delta:+.3f} dB")
    print(f"  H2 fair verdict: {h2} WINS on absolute PSNR")
    print(f"PAIRED_RESULT size={args.size} steps={args.steps} n_train={args.n_train} "
          f"wd={args.weight_decay} drop={args.dropout} loss={args.loss} lr={args.lr_schedule} | "
          f"phys_tr={p_tr_p.mean():.3f} brwn_tr={p_tr_b.mean():.3f} "
          f"mean_delta={mean_d:+.3f} hi_delta={hi_delta:+.3f} "
          f"phys_wins={wins}/{n_test} verdict={h2}")


if __name__ == "__main__":
    main()
