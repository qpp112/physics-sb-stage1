"""
Experiment 1b (v3) — bridge matching training with reference-mean conditioning.

Same as exps/1b_train.py except the UNet is additionally conditioned on the
reference's aligned mean frame:

    u_ref    = reference.velocity(x_A, x_B, flow_AB, t)     # v2 conditioning
    ref_mean = reference.mean(x_A, x_B, flow_AB, t)         # v3 conditioning (NEW)
    pred     = unet_v3(x_t, t, x_A, x_B, u_ref, ref_mean)
    target_v = dot{I}_t - u_ref                             # unchanged

Flags (configs/*_v3.yaml):
    bridge.condition_on_reference:      feed u_ref       (default True)
    bridge.condition_on_reference_mean: feed ref_mean    (default True)

Usage:
    python exps/1b_train_v3.py --config configs/physics_v3.yaml --seed 0 --run_name physics_v3_s0
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps._config import load_config
from data import FlyingThings3D
from bridges.reference import make_reference
from bridges import interpolant
from bridges.conditioning_v3c import reference_occlusion_map
from model.unet_v3 import UNetV3
from model import count_parameters


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def bridge_matching_loss(model, reference, batch, cfg, device):
    xA = batch["x_A"].to(device, non_blocking=True)
    xM = batch["x_M"].to(device, non_blocking=True)
    xB = batch["x_B"].to(device, non_blocking=True)
    fAB = batch["flow_AB"].to(device, non_blocking=True)

    t = interpolant.sample_t(
        xA.shape[0],
        mode=cfg["bridge"]["t_sampling"],
        concentration=cfg["bridge"]["beta_concentration"],
        device=device,
    ).clamp(0.02, 0.98)

    I_t = interpolant.interpolate(xA, xM, xB, t)
    dot_I = interpolant.velocity(xA, xM, xB, t)
    u_ref = reference.velocity(xA, xB, fAB, t)

    sigma = float(cfg["bridge"]["sigma"])
    eps = torch.randn_like(I_t)
    std = sigma * (t * (1 - t)).clamp_min(0).sqrt().view(-1, 1, 1, 1)
    x_t = I_t + std * eps

    target_v = dot_I - u_ref
    cond_ref = cfg["bridge"].get("condition_on_reference", True)
    cond_mean = cfg["bridge"].get("condition_on_reference_mean", True)
    cond_occ = cfg["bridge"].get("condition_on_occlusion", False)
    ref_mean = reference.mean(xA, xB, fAB, t) if cond_mean else None
    occ = reference_occlusion_map(reference, xA, xB, fAB, t) if cond_occ else None
    pred_v = model(x_t, t, xA, xB, u_ref if cond_ref else None, ref_mean, occ)
    return F.mse_loss(pred_v, target_v), {
        "loss": None,
        "target_norm": target_v.pow(2).mean().detach().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["seed"] = args.seed
    run_name = args.run_name or f"{cfg.get('run_name', 'run')}_s{args.seed}"
    out_dir = Path(cfg["paths"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = FlyingThings3D(
        root=cfg["data"]["root"], split="TRAIN",
        size=cfg["data"]["size"], min_flow=cfg["data"]["min_flow"],
        max_clips=cfg["data"]["train_clips"],
        pass_name=cfg["data"]["pass"], seed=args.seed,
    )
    val_ds = FlyingThings3D(
        root=cfg["data"]["root"], split="TEST",
        size=cfg["data"]["size"], min_flow=cfg["data"]["min_flow"],
        max_clips=cfg["data"]["val_clips"],
        pass_name=cfg["data"]["pass"], seed=args.seed + 9999,
    )
    print(f"train {len(train_ds)} | val {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    image_size = cfg["data"]["size"]
    cond_ref = cfg["bridge"].get("condition_on_reference", True)
    cond_mean = cfg["bridge"].get("condition_on_reference_mean", True)
    cond_occ = cfg["bridge"].get("condition_on_occlusion", False)
    in_channels = (9 + (3 if cond_ref else 0) + (3 if cond_mean else 0)
                   + (1 if cond_occ else 0))
    model = UNetV3(
        in_channels=in_channels, out_channels=3,
        base_channels=cfg["model"]["base_channels"],
        channel_mult=tuple(cfg["model"]["channel_mult"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        dropout=cfg["model"]["dropout"],
        image_size=image_size,
    ).to(device)
    print(f"in_channels={in_channels} | params: {count_parameters(model)/1e6:.2f} M")

    reference = make_reference(cfg["bridge"]["reference"])
    ema = EMA(model, decay=cfg["train"]["ema_decay"])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["mixed_precision"])

    def lr_lambda(step):
        warm = cfg["train"]["warmup_steps"]
        if step < warm:
            return step / max(warm, 1)
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["state_dict"])
        ema.shadow = ck["ema_state_dict"]
        opt.load_state_dict(ck["opt_state_dict"])
        start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    tb = SummaryWriter(out_dir / "tb")

    step = start_step
    model.train()
    total_steps = cfg["train"]["steps"]
    t0 = time.time()
    iter_train = iter(train_loader)

    pbar = tqdm(total=total_steps, initial=start_step,
                dynamic_ncols=True, desc=run_name, unit="step")
    while step < total_steps:
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg["train"]["mixed_precision"]):
            loss, aux = bridge_matching_loss(model, reference, batch, cfg, device)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)

        if step % cfg["train"]["log_every"] == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", tgt=f"{aux['target_norm']:.3f}")
            tb.add_scalar("train/loss", loss.item(), step)
            tb.add_scalar("train/target_norm", aux["target_norm"], step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)

        if step > 0 and step % cfg["train"]["val_every"] == 0:
            val_loss = run_val(model, reference, val_loader, cfg, device)
            tqdm.write(f"  [val] step {step}  loss {val_loss:.4f}")
            tb.add_scalar("val/loss", val_loss, step)

        if step > 0 and step % cfg["train"]["ckpt_every"] == 0:
            torch.save(dict(
                step=step, state_dict=model.state_dict(),
                ema_state_dict=ema.shadow,
                opt_state_dict=opt.state_dict(),
            ), out_dir / f"ckpt_{step:07d}.pt")

        step += 1
        pbar.update(1)

    pbar.close()
    torch.save(dict(
        step=step - 1, state_dict=model.state_dict(),
        ema_state_dict=ema.shadow, opt_state_dict=opt.state_dict(),
    ), out_dir / "ckpt_final.pt")
    print(f"done. final ckpt at {out_dir/'ckpt_final.pt'}")
    tb.close()


@torch.no_grad()
def run_val(model, reference, val_loader, cfg, device):
    model.eval()
    losses = []
    for batch in val_loader:
        loss, _ = bridge_matching_loss(model, reference, batch, cfg, device)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


if __name__ == "__main__":
    main()
