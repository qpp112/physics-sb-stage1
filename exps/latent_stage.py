"""
Staged trainer for the Option-3 + cross-attention + latent-bridge model.

  --stage s0 : train SharedAE (faithful codec). Gate: recon PSNR >= --gate_psnr.
               Saves weights to runs/latent/ae_{size}_b{base}_f{fdim}.pt
  --stage s1 : train LatentPrior (cross-attn + f_interp + residual decode).
               Optionally --ae_init <path> to warm-start (and --freeze_enc).
               Gate: midpoint PSNR > physics-V1 floor.

Reuses arch_cycle's half-flow cache + helpers so numbers are comparable.

  python3 exps/latent_stage.py --stage s0 --size 64 --steps 6000
  python3 exps/latent_stage.py --stage s1 --size 64 --steps 8000 \
      --ae_init runs/latent/ae_64_b48_f128.pt
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.latent_bridge import make_latent
from model.synth import make_arch
from model import count_parameters
from metrics.image_metrics import psnr, ssim
from exps.arch_cycle import (build_cache_hf, EMA, _charb, _eval_mid, _hi_mean)

LOG = Path("arch_cycles.log")
CKPT = Path("runs/latent")


def _edge_l1(pred, target):
    def grad(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy
    px, py = grad(pred)
    tx, ty = grad(target)
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def _recon_loss(pred, target):
    return F.l1_loss(pred, target) + 0.25 * F.mse_loss(pred, target) + \
        0.5 * _edge_l1(pred, target)


@torch.no_grad()
def _eval_recon(ae, frames, device, bs=64):
    ps, ss = [], []
    for s in range(0, frames.shape[0], bs):
        x = frames[s:s + bs]
        xr = ae(x)
        ps.append(psnr(xr, x))
        ss.append(ssim(xr, x))
    return torch.cat(ps).mean().item(), torch.cat(ss).mean().item()


def run_s0(args, tr, te, device):
    ae = make_latent("s0", base=args.base_ch, fdim=args.fdim,
                     dropout=args.dropout).to(device)
    tr_frames = torch.cat([tr["x_A"], tr["x_M"], tr["x_B"]], 0)
    te_frames = torch.cat([te["x_A"], te["x_M"], te["x_B"]], 0)
    n = tr_frames.shape[0]
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(ae, args.ema_decay) if args.ema_decay > 0 else None
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    ae.train()
    t1 = time.time()
    last = float("nan")
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), generator=gen, device=device)
        x = tr_frames[idx]
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = _recon_loss(ae(x), x)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ae.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            ema.update(ae)
        last = loss.item()
    train_s = time.time() - t1
    if ema is not None:
        ae.load_state_dict(ema.shadow, strict=True)
    ae.eval()
    rp, rs = _eval_recon(ae, te_frames, device)
    verdict = "PASS" if rp >= args.gate_psnr else "FAIL"
    CKPT.mkdir(parents=True, exist_ok=True)
    path = CKPT / f"ae_{args.size}_b{args.base_ch}_f{args.fdim}.pt"
    torch.save(ae.state_dict(), path)
    line = (f"LATENT_S0 [{verdict}] size={args.size} steps={args.steps} "
            f"base_ch={args.base_ch} fdim={args.fdim} params={count_parameters(ae)/1e6:.3f}M "
            f"recon_loss={last:.5f} | recon_PSNR={rp:.3f} SSIM={rs:.4f} "
            f"gate>={args.gate_psnr} -> {verdict} | saved={path} (train {train_s:.1f}s)")
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    return verdict


def run_s1(args, tr, te, strata, device, v1_psnr, v1_hi):
    model = make_latent("s1", base=args.base_ch, fdim=args.fdim,
                        dropout=args.dropout).to(device)
    if args.ae_init:
        sd = torch.load(args.ae_init, map_location=device)
        missing, unexpected = model.ae.load_state_dict(sd, strict=False)
        # re-zero residual head so we still start exactly at the warp floor
        torch.nn.init.zeros_(model.ae.head[-1].weight)
        torch.nn.init.zeros_(model.ae.head[-1].bias)
        print(f"[s1] warm-start AE from {args.ae_init} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    if args.freeze_enc:
        for m in (model.ae.c0, model.ae.d1, model.ae.d2, model.ae.d3):
            for p in m.parameters():
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    n = tr["x_A"].shape[0]
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    model.train()
    t1 = time.time()
    last = float("nan")
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), generator=gen, device=device)
        xA, xB, xM, fAB = (tr["x_A"][idx], tr["x_B"][idx],
                           tr["x_M"][idx], tr["flow_AB"][idx])
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            mid = model(xA, xB, fAB)["mid"]
            loss = _charb(mid, xM)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            ema.update(model)
        last = loss.item()
    train_s = time.time() - t1
    if ema is not None:
        model.load_state_dict(ema.shadow, strict=True)
    model.eval()
    p, s = _eval_mid(model, te, device, bs=32)
    trained = p.mean().item()
    hi = _hi_mean(p, strata, device)
    d_v1 = trained - v1_psnr
    verdict = "WIN" if d_v1 > 0.05 else ("tie" if d_v1 > -0.05 else "lose")
    tag = "s1_frozen" if args.freeze_enc else ("s1_warm" if args.ae_init else "s1_scratch")
    line = (f"LATENT_S1 [{verdict}] tag={tag} size={args.size} steps={args.steps} "
            f"base_ch={args.base_ch} fdim={args.fdim} params={count_parameters(model)/1e6:.3f}M "
            f"train_loss={last:.5f} | PSNR={trained:.3f} SSIM={s.mean().item():.4f} hi={hi:.3f} | "
            f"vs_v1={d_v1:+.3f}dB hi_vs_v1={hi-v1_hi:+.3f}dB (train {train_s:.1f}s)")
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT / f"s1_{tag}_{args.size}_b{args.base_ch}_f{args.fdim}.pt")
    return verdict


def run_s2(args, tr, te, strata, device, v1_psnr, v1_hi):
    model = make_latent("s2", base=args.base_ch, fdim=args.fdim,
                        dropout=args.dropout, freeze_prior=not args.no_freeze_prior).to(device)
    sd = torch.load(args.s1_init, map_location=device)
    missing, unexpected = model.prior.load_state_dict(sd, strict=False)
    print(f"[s2] loaded S1 prior from {args.s1_init} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")

    # one-shot prior baseline (the C-GATE bar) on the SAME test clips
    model.prior.eval()
    p_prior, _ = _eval_mid(model.prior, te, device, bs=32)
    prior_psnr, prior_hi = p_prior.mean().item(), _hi_mean(p_prior, strata, device)
    print(f"[s2] one-shot S1 prior PSNR={prior_psnr:.3f} hi={prior_hi:.3f} (C-GATE bar)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    n = tr["x_A"].shape[0]
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    model.train()
    t1 = time.time()
    last = float("nan")
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), generator=gen, device=device)
        xA, xB, xM, fAB = (tr["x_A"][idx], tr["x_B"][idx],
                           tr["x_M"][idx], tr["flow_AB"][idx])
        # noise level schedule: sigma ~ U(0, sigma_max); t encodes the level
        sig = torch.rand(1, device=device).item() * args.sigma_max
        tval = torch.full((args.batch,), sig / max(args.sigma_max, 1e-6), device=device)
        z_M = model.encode_target(xM)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = model(xA, xB, fAB, t=tval, sigma=sig)
            loss = _charb(out["mid"], xM) + args.lat_w * F.mse_loss(out["z_M_hat"], z_M)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            ema.update(model)
        last = loss.item()
    train_s = time.time() - t1
    if ema is not None:
        model.load_state_dict(ema.shadow, strict=True)
    model.eval()

    # deterministic refine at eval (sigma=0, t=0)
    @torch.no_grad()
    def eval_bridge(bs=32):
        xA, xB, xM, fAB = te["x_A"], te["x_B"], te["x_M"], te["flow_AB"]
        ps, ss = [], []
        t0 = torch.zeros(bs, device=device)
        for s in range(0, xA.shape[0], bs):
            sl = slice(s, min(s + bs, xA.shape[0]))
            tt = t0[:xA[sl].shape[0]]
            mid = model(xA[sl], xB[sl], fAB[sl], t=tt, sigma=0.0)["mid"]
            ps.append(psnr(mid, xM[sl]))
            ss.append(ssim(mid, xM[sl]))
        return torch.cat(ps), torch.cat(ss)

    p, s = eval_bridge()
    trained = p.mean().item()
    hi = _hi_mean(p, strata, device)
    d_prior = trained - prior_psnr
    verdict = "PASS" if d_prior > 0.05 else "FAIL"   # C-GATE: must beat one-shot prior
    line = (f"LATENT_S2 [{verdict}] size={args.size} steps={args.steps} "
            f"base_ch={args.base_ch} fdim={args.fdim} sigma_max={args.sigma_max} "
            f"freeze_prior={not args.no_freeze_prior} params={count_parameters(model)/1e6:.3f}M "
            f"train_loss={last:.5f} | PSNR={trained:.3f} SSIM={s.mean().item():.4f} hi={hi:.3f} | "
            f"prior={prior_psnr:.3f} C-GATE_delta={d_prior:+.3f}dB | vs_v1={trained-v1_psnr:+.3f}dB "
            f"(train {train_s:.1f}s)")
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT / f"s2_{args.size}_b{args.base_ch}_f{args.fdim}.pt")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["s0", "s1", "s2"])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--base_ch", type=int, default=48)
    ap.add_argument("--fdim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/home/ubuntu/datasets/FlyingThings3D")
    ap.add_argument("--pass_name", default="cleanpass")
    ap.add_argument("--min_flow", type=float, default=40.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--gate_psnr", type=float, default=32.0)
    ap.add_argument("--ae_init", default=None)
    ap.add_argument("--freeze_enc", action="store_true")
    ap.add_argument("--s1_init", default=None, help="S1 prior checkpoint for S2")
    ap.add_argument("--sigma_max", type=float, default=0.5,
                    help="max latent noise level for the bridge denoiser")
    ap.add_argument("--lat_w", type=float, default=0.1,
                    help="weight on latent MSE(z_M_hat, z_M)")
    ap.add_argument("--no_freeze_prior", action="store_true",
                    help="fine-tune the S1 prior jointly with the bridge")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    blob = build_cache_hf(args.size, args.n_train, args.n_test, args.min_flow,
                          args.root, args.pass_name)
    tr = {k: v.to(device) for k, v in blob["train"].items() if torch.is_tensor(v)}
    te = {k: v.to(device) for k, v in blob["test"].items() if torch.is_tensor(v)}
    strata = blob["strata"]
    print(f"[latent {args.stage}] train={tr['x_A'].shape[0]} test={te['x_A'].shape[0]} "
          f"size={args.size} ({time.time()-t0:.1f}s)")

    if args.stage == "s0":
        run_s0(args, tr, te, device)
    else:
        v1 = make_arch("physics_v1").to(device).eval()
        p_v1, _ = _eval_mid(v1, te, device)
        v1_psnr, v1_hi = p_v1.mean().item(), _hi_mean(p_v1, strata, device)
        print(f"[floor] physics_v1 PSNR={v1_psnr:.3f} hi={v1_hi:.3f}")
        if args.stage == "s1":
            run_s1(args, tr, te, strata, device, v1_psnr, v1_hi)
        else:
            run_s2(args, tr, te, strata, device, v1_psnr, v1_hi)


if __name__ == "__main__":
    main()
