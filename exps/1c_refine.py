"""
Experiment 1c — direct midpoint refinement (robust alternative to the bridge).

Motivation (see prereg.md CORRECTION 3): the bridge-matching model underperforms
the static prior because open-loop integration visits off-path states the model
never saw in training (exposure bias from tiny sigma). This arm sidesteps that
entirely:

    M_hat = f( reference.mean(0.5), x_A, x_B, u_ref(0.5) )         # ONE forward pass
    loss  = MSE(M_hat, x_M)

Train and inference feed the SAME input (the reference's midpoint estimate, which
does NOT encode the true M), so there is no leak and no compounding. The network
learns the residual from the static warped midpoint to the true midpoint. Worst
case it learns the identity and matches the static prior; it cannot diverge below
it the way the integrated bridge does.

The two conditions differ ONLY in `reference` (physics vs brownian) -- the same
H2 question, in a setting where training is well-posed.

Usage:
    python exps/1c_refine.py --config configs/physics_refine.yaml --seed 0 \
        --run_name physics_refine
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps._config import load_config
from exps._inference import build_model_from_cfg, _load_ckpt
from data import FlyingThings3D, stratify_by_motion
from bridges.reference import make_reference
from model import count_parameters
from metrics.image_metrics import psnr, ssim, lpips_metric
from metrics.flow_metrics import epe_against_gt_flow

T_MID = 0.5


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def refine_predict(model, reference, xA, xB, fAB, cond_ref, input_noise=0.0):
    """One-shot midpoint prediction. Input = reference midpoint estimate."""
    b = xA.shape[0]
    t_b = torch.full((b,), T_MID, device=xA.device)
    ref_mid = reference.mean(xA, xB, fAB, t_b)
    if input_noise > 0.0:
        ref_mid = ref_mid + input_noise * torch.randn_like(ref_mid)
    u_ref = reference.velocity(xA, xB, fAB, t_b)
    return model(ref_mid, t_b, xA, xB, u_ref if cond_ref else None)


def refine_loss(model, reference, batch, cfg, device):
    xA = batch["x_A"].to(device, non_blocking=True)
    xM = batch["x_M"].to(device, non_blocking=True)
    xB = batch["x_B"].to(device, non_blocking=True)
    fAB = batch["flow_AB"].to(device, non_blocking=True)
    cond_ref = cfg["bridge"].get("condition_on_reference", True)
    input_noise = float(cfg.get("refine", {}).get("input_noise", 0.0))
    pred = refine_predict(model, reference, xA, xB, fAB, cond_ref, input_noise)
    return F.mse_loss(pred, xM)


@torch.no_grad()
def evaluate(model, reference, cfg, device, n_clips, cond_ref):
    ds = FlyingThings3D(root=cfg["data"]["root"], split="TEST",
                        size=cfg["data"]["size"], min_flow=cfg["data"]["min_flow"],
                        max_clips=n_clips, pass_name=cfg["data"]["pass"], seed=0)
    strata = stratify_by_motion(ds, cfg["eval"]["strata_edges"])
    edges = cfg["eval"]["strata_edges"]
    dl = DataLoader(ds, batch_size=8, num_workers=cfg["data"]["num_workers"])

    met = defaultdict(list)       # trained
    base = defaultdict(list)      # static prior
    for batch in dl:
        xA = batch["x_A"].to(device); xB = batch["x_B"].to(device)
        xM = batch["x_M"].to(device); fAB = batch["flow_AB"].to(device)
        b = xA.shape[0]
        t_b = torch.full((b,), T_MID, device=device)
        ref_mid = reference.mean(xA, xB, fAB, t_b)
        pred = refine_predict(model, reference, xA, xB, fAB, cond_ref, 0.0)
        for tag, est in [("t", pred), ("s", ref_mid)]:
            d = met if tag == "t" else base
            d["psnr"].append(psnr(est, xM).cpu().numpy())
            d["ssim"].append(ssim(est, xM).cpu().numpy())
            d["lpips"].append(lpips_metric(est, xM).cpu().numpy())
            d["epe"].append(epe_against_gt_flow(est, xA, fAB).cpu().numpy())
    met = {k: np.concatenate(v) for k, v in met.items()}
    base = {k: np.concatenate(v) for k, v in base.items()}

    def agg(d, idx=None):
        idx = np.arange(len(d["psnr"])) if idx is None else np.array(idx)
        return {k: float(v[idx].mean()) for k, v in d.items()}

    out = {"aggregate": {"trained": agg(met), "static": agg(base)}, "strata": {}}
    for b_i, idx in enumerate(strata):
        if not idx:
            continue
        rng = (f"[{int(edges[b_i])},{int(edges[b_i+1])})"
               if b_i < len(edges) - 2 else f"[{int(edges[b_i])},inf)")
        out["strata"][f"stratum_{b_i}"] = {"range_px": rng, "n": len(idx),
                                           "trained": agg(met, idx), "static": agg(base, idx)}
    return out


def print_eval(name, ev):
    a = ev["aggregate"]
    print(f"\n=== {name}: midpoint refinement eval ===")
    print(f"  AGGREGATE  trained PSNR={a['trained']['psnr']:.2f}  "
          f"static PSNR={a['static']['psnr']:.2f}  "
          f"gain={a['trained']['psnr']-a['static']['psnr']:+.2f} dB")
    for key in sorted(ev["strata"]):
        s = ev["strata"][key]
        print(f"  {s['range_px']:12s} n={s['n']:4d}  "
              f"trained {s['trained']['psnr']:.2f}  static {s['static']['psnr']:.2f}  "
              f"gain {s['trained']['psnr']-s['static']['psnr']:+.2f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["seed"] = args.seed
    run_name = args.run_name or f"{cfg.get('run_name', 'refine')}_s{args.seed}"
    out_dir = Path(cfg["paths"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond_ref = cfg["bridge"].get("condition_on_reference", True)

    train_ds = FlyingThings3D(root=cfg["data"]["root"], split="TRAIN",
                              size=cfg["data"]["size"], min_flow=cfg["data"]["min_flow"],
                              max_clips=cfg["data"]["train_clips"],
                              pass_name=cfg["data"]["pass"], seed=args.seed)
    print(f"train {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], drop_last=True, pin_memory=True)

    model = build_model_from_cfg(cfg, cfg["data"]["size"]).to(device)
    print(f"params: {count_parameters(model)/1e6:.2f} M")
    reference = make_reference(cfg["bridge"]["reference"])
    ema = EMA(model, decay=cfg["train"]["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["mixed_precision"])

    def lr_lambda(step):
        warm = cfg["train"]["warmup_steps"]
        return step / max(warm, 1) if step < warm else 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["state_dict"]); ema.shadow = ck["ema_state_dict"]
        opt.load_state_dict(ck["opt_state_dict"]); start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    tb = SummaryWriter(out_dir / "tb")
    total_steps = cfg["train"]["steps"]
    step = start_step
    model.train()
    iter_train = iter(train_loader)
    pbar = tqdm(total=total_steps, initial=start_step, dynamic_ncols=True,
                desc=run_name, unit="step")
    while step < total_steps:
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader); batch = next(iter_train)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg["train"]["mixed_precision"]):
            loss = refine_loss(model, reference, batch, cfg, device)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step(); ema.update(model)

        if step % cfg["train"]["log_every"] == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            tb.add_scalar("train/loss", loss.item(), step)
        if step > 0 and step % cfg["train"]["ckpt_every"] == 0:
            torch.save(dict(step=step, state_dict=model.state_dict(),
                            ema_state_dict=ema.shadow, opt_state_dict=opt.state_dict()),
                       out_dir / f"ckpt_{step:07d}.pt")
        step += 1; pbar.update(1)
    pbar.close()
    torch.save(dict(step=step - 1, state_dict=model.state_dict(),
                    ema_state_dict=ema.shadow, opt_state_dict=opt.state_dict()),
               out_dir / "ckpt_final.pt")
    tb.close()

    # eval with EMA weights
    eval_model = build_model_from_cfg(cfg, cfg["data"]["size"]).to(device)
    _load_ckpt(eval_model, str(out_dir / "ckpt_final.pt"), device)
    ev = evaluate(eval_model, reference, cfg, device,
                  n_clips=cfg["data"]["test_clips"], cond_ref=cond_ref)
    (out_dir / "refine_eval.json").write_text(json.dumps(ev, indent=2))
    print_eval(run_name, ev)
    print(f"\ndone. ckpt + refine_eval.json in {out_dir}")


if __name__ == "__main__":
    main()
