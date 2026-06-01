"""
Architecture-cycling harness for the midpoint prior.

Goal: find the best ARCHITECTURE for predicting the midpoint M from
(x_A, x_B, F_{A->B}). We train each candidate identically and rank by midpoint
PSNR vs ground-truth M, with the physics-V1 static prior as the floor
(delta_v1 = arch_PSNR - physics_v1_PSNR on the SAME clips).

Fast by design: low res (64px default), small data subset cached in GPU memory.
Run several archs in one shot and print a leaderboard:

  python3 exps/arch_cycle.py --archs physics_v1 flow_synth feat_warp feat_flow \
      --size 64 --steps 4000 --n_train 3000 --n_test 200

Each finished arch appends an ARCH_RESULT line to arch_cycles.log.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import FlyingThings3D, stratify_by_motion
from model.synth import make_arch
from model import count_parameters
from metrics.image_metrics import psnr, ssim

CACHE_DIR = Path("runs/_fastcache")
CACHE_LOAD_SIZE = 256
EDGES = [40, 60, 100, 1000000000]
LOG = Path("arch_cycles.log")


def _downsample_hf(blob, size):
    if size == CACHE_LOAD_SIZE:
        return blob
    sc = size / CACHE_LOAD_SIZE
    for k in ("x_A", "x_M", "x_B"):
        blob[k] = F.interpolate(blob[k], size=(size, size), mode="bilinear",
                                align_corners=False)
    for k in ("flow_AB", "flow_AM", "flow_MB"):
        if k in blob:
            blob[k] = F.interpolate(blob[k], size=(size, size), mode="bilinear",
                                    align_corners=False) * sc
    return blob


def build_cache_hf(size, n_train, n_test, min_flow, root, pass_name):
    """Train + test clips WITH half-flows, loaded @256 then downsampled."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = CACHE_DIR / f"arch_hf_{size}_{n_train}_{n_test}_{int(min_flow)}.pt"
    if key.exists():
        return torch.load(key)
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    keys = ["x_A", "x_M", "x_B", "flow_AB", "flow_AM", "flow_MB"]

    def grab(split, n, seed):
        ds = FlyingThings3D(root=root, split=split, size=CACHE_LOAD_SIZE,
                            min_flow=min_flow, max_clips=n, pass_name=pass_name,
                            seed=seed, return_half_flows=True)
        acc = {k: [] for k in keys}
        dl = DataLoader(ds, batch_size=8, num_workers=8, shuffle=False)
        for b in tqdm(dl, desc=f"cache {split}", total=(len(ds) + 7) // 8):
            for k in keys:
                acc[k].append(b[k])
        return _downsample_hf({k: torch.cat(v) for k, v in acc.items()}, size), ds

    tr, _ = grab("TRAIN", n_train, 0)
    te, te_ds = grab("TEST", n_test, 0)
    blob = {"train": tr, "test": te, "size": size,
            "strata": stratify_by_motion(te_ds, EDGES), "edges": EDGES}
    torch.save(blob, key)
    return blob


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


def _charb(pred, target, eps=1e-3):
    return torch.sqrt((pred - target).pow(2) + eps * eps).mean()


@torch.no_grad()
def _eval_mid(model, te, device, bs=64):
    xA, xB, xM, fAB = te["x_A"], te["x_B"], te["x_M"], te["flow_AB"]
    ps, ss = [], []
    for s in range(0, xA.shape[0], bs):
        sl = slice(s, min(s + bs, xA.shape[0]))
        mid = model(xA[sl], xB[sl], fAB[sl])["mid"]
        ps.append(psnr(mid, xM[sl]))
        ss.append(ssim(mid, xM[sl]))
    return torch.cat(ps), torch.cat(ss)


def _hi_mean(p, strata, device):
    for b_i, idx in enumerate(strata):
        if idx and b_i == len(strata) - 1:
            return p[torch.tensor(idx, device=device)].mean().item()
    return float("nan")


def train_arch(name, args, tr, te, strata, device, v1_psnr, v1_hi):
    model = make_arch(name, base_ch=args.base_ch, dropout=args.dropout).to(device)
    n_params = count_parameters(model) / 1e6
    trainable = getattr(type(model), "trainable", True)

    train_s = 0.0
    last_loss = float("nan")
    if trainable and args.steps > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=args.lr * 0.05)
        scaler = torch.cuda.amp.GradScaler()
        ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
        n = tr["x_A"].shape[0]
        gen = torch.Generator(device=device).manual_seed(args.seed + 17)
        model.train()
        t1 = time.time()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            last_loss = loss.item()
        train_s = time.time() - t1
        if ema is not None:
            model.load_state_dict(ema.shadow, strict=True)

    model.eval()
    p, s = _eval_mid(model, te, device)
    trained = p.mean().item()
    hi = _hi_mean(p, strata, device)
    d_v1 = trained - v1_psnr
    d_v1_hi = hi - v1_hi
    verdict = "WIN" if d_v1 > 0.05 else ("tie" if d_v1 > -0.05 else "lose")

    line = (f"ARCH_RESULT [{verdict}] arch={name} size={args.size} steps={args.steps} "
            f"n_train={args.n_train} base_ch={args.base_ch} drop={args.dropout} "
            f"params={n_params:.3f}M train_loss={last_loss:.5f} | "
            f"PSNR={trained:.3f} SSIM={s.mean().item():.4f} hi={hi:.3f} | "
            f"vs_v1={d_v1:+.3f}dB hi_vs_v1={d_v1_hi:+.3f}dB (train {train_s:.1f}s)")
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    return dict(arch=name, psnr=trained, hi=hi, d_v1=d_v1, d_v1_hi=d_v1_hi,
                params=n_params, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["physics_v1", "flow_synth", "feat_warp", "feat_flow"])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--base_ch", type=int, default=48)
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    blob = build_cache_hf(args.size, args.n_train, args.n_test, args.min_flow,
                          args.root, args.pass_name)
    tr = {k: v.to(device) for k, v in blob["train"].items() if torch.is_tensor(v)}
    te = {k: v.to(device) for k, v in blob["test"].items() if torch.is_tensor(v)}
    strata = blob["strata"]
    print(f"[arch_cycle] train={tr['x_A'].shape[0]} test={te['x_A'].shape[0]} "
          f"size={args.size} ({time.time()-t0:.1f}s)  archs={args.archs}")

    v1 = make_arch("physics_v1").to(device).eval()
    p_v1, _ = _eval_mid(v1, te, device)
    v1_psnr, v1_hi = p_v1.mean().item(), _hi_mean(p_v1, strata, device)
    print(f"[floor] physics_v1 PSNR={v1_psnr:.3f}  hi[100,inf)={v1_hi:.3f}")

    rows = []
    for name in args.archs:
        rows.append(train_arch(name, args, tr, te, strata, device, v1_psnr, v1_hi))

    rows.sort(key=lambda r: r["psnr"], reverse=True)
    print("\n=== ARCH LEADERBOARD (size={}, steps={}) ===".format(args.size, args.steps))
    print(f"  {'arch':14s} {'PSNR':>8s} {'hi':>8s} {'vs_v1':>9s} {'hi_vs_v1':>9s} "
          f"{'params':>9s}  verdict")
    for r in rows:
        print(f"  {r['arch']:14s} {r['psnr']:8.3f} {r['hi']:8.3f} {r['d_v1']:+9.3f} "
              f"{r['d_v1_hi']:+9.3f} {r['params']:8.3f}M  {r['verdict']}")


if __name__ == "__main__":
    main()
