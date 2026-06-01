"""
Fast experiment harness for small, tractable cycles.

Goal: find a learned recipe that BEATS the static midpoint prior (reference.mean(0.5)),
at small scale, in ~1 min per run -- then scale only the winner.

Speed tricks:
  * low res (default 64px), small UNet, small data subset, held in GPU memory
  * the reference midpoint + drift at t=0.5 are CONSTANT per sample, so we
    precompute them once (no per-step warp cost)
  * data subset cached to disk (runs/_fastcache), reused across cycles

Variants (all one-shot midpoint predictors, no integration):
  refine        : predict M from reference-midpoint input
  refine_resid  : predict (M - reference-midpoint); output 0 == static (a strict floor)

Usage:
  python exps/fast_cycle.py --variant refine_resid --reference physics \
      --steps 3000 --n_train 3000 --weight_decay 0.05 --dropout 0.1
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import FlyingThings3D, stratify_by_motion
from bridges.reference import make_reference
from model import UNet, count_parameters
from metrics.image_metrics import psnr, ssim

CACHE_DIR = Path("runs/_fastcache")
CACHE_LOAD_SIZE = 256
T_MID = 0.5


def _downsample(blob_t, size):
    if size == CACHE_LOAD_SIZE:
        return blob_t
    sc = size / CACHE_LOAD_SIZE
    for k in ["x_A", "x_M", "x_B"]:
        blob_t[k] = F.interpolate(blob_t[k], size=(size, size), mode="bilinear",
                                  align_corners=False)
    blob_t["flow_AB"] = F.interpolate(blob_t["flow_AB"], size=(size, size),
                                      mode="bilinear", align_corners=False) * sc
    return blob_t


def build_cache(size, n_train, n_test, min_flow, root, pass_name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = CACHE_DIR / f"ft3d_{size}_{n_train}_{n_test}_{int(min_flow)}.pt"
    if key.exists():
        return torch.load(key)
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    def grab(split, n, seed):
        ds = FlyingThings3D(root=root, split=split, size=CACHE_LOAD_SIZE, min_flow=min_flow,
                            max_clips=n, pass_name=pass_name, seed=seed)
        keys = ["x_A", "x_M", "x_B", "flow_AB"]
        acc = {k: [] for k in keys}
        dl = DataLoader(ds, batch_size=16, num_workers=8, shuffle=False)
        for b in tqdm(dl, desc=f"cache {split}", total=(len(ds) + 15) // 16):
            for k in keys:
                acc[k].append(b[k])
        return _downsample({k: torch.cat(v) for k, v in acc.items()}, size), ds

    tr, _ = grab("TRAIN", n_train, 0)
    te, te_ds = grab("TEST", n_test, 0)
    edges = [40, 60, 100, 1000000000]
    blob = {"train": tr, "test": te, "size": size,
            "strata": stratify_by_motion(te_ds, edges), "edges": edges}
    torch.save(blob, key)
    return blob


@torch.no_grad()
def precompute_ref(reference, xA, xB, fAB, bs=128):
    mids, urefs = [], []
    for s in range(0, xA.shape[0], bs):
        sl = slice(s, s + bs)
        t_b = torch.full((xA[sl].shape[0],), T_MID, device=xA.device)
        mids.append(reference.mean(xA[sl], xB[sl], fAB[sl], t_b))
        urefs.append(reference.velocity(xA[sl], xB[sl], fAB[sl], t_b))
    return torch.cat(mids), torch.cat(urefs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="refine_resid")
    ap.add_argument("--reference", default="physics", choices=["physics", "brownian"])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--input_noise", type=float, default=0.0)
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/home/ubuntu/datasets/FlyingThings3D")
    ap.add_argument("--pass_name", default="cleanpass")
    ap.add_argument("--min_flow", type=float, default=40.0)
    ap.add_argument("--loss", default="mse", choices=["mse", "l1", "charb"])
    ap.add_argument("--lr_schedule", default="none", choices=["none", "cosine"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA for eval weights; 0 = disable")
    args = ap.parse_args()
    run_single(args)


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


def _loss_fn(name: str):
    if name == "mse":
        return F.mse_loss
    if name == "l1":
        return F.l1_loss
    if name == "charb":
        def charb(pred, target, eps=1e-3):
            return torch.sqrt((pred - target).pow(2) + eps * eps).mean()
        return charb
    raise ValueError(name)


def _make_model(args, device):
    return UNet(
        in_channels=12, out_channels=3, base_channels=args.base_ch,
        channel_mult=(1, 2, 2), num_res_blocks=1, attn_resolutions=(16,),
        time_emb_dim=128, dropout=args.dropout, image_size=args.size,
    ).to(device)


def run_single(args, init_state=None, train_idx=None):
    """Train + eval one reference. Optional shared init and fixed batch indices."""
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = make_reference(args.reference)
    resid = args.variant == "refine_resid"

    t0 = time.time()
    blob = build_cache(args.size, args.n_train, args.n_test, args.min_flow,
                       args.root, args.pass_name)
    tr = {k: v.to(device) for k, v in blob["train"].items() if torch.is_tensor(v)}
    te = {k: v.to(device) for k, v in blob["test"].items() if torch.is_tensor(v)}
    strata, edges = blob["strata"], blob["edges"]
    tr_mid, tr_uref = precompute_ref(reference, tr["x_A"], tr["x_B"], tr["flow_AB"])
    te_mid, te_uref = precompute_ref(reference, te["x_A"], te["x_B"], te["flow_AB"])
    print(f"[cache] train={tr['x_A'].shape[0]} test={te['x_A'].shape[0]} size={args.size} "
          f"({time.time()-t0:.1f}s)")

    model = _make_model(args, device)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.05)
        if args.lr_schedule == "cosine" else None)
    loss_fn = _loss_fn(args.loss)
    n = tr["x_A"].shape[0]
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    if train_idx is None:
        gen = torch.Generator(device=device).manual_seed(args.seed + 17)
        train_idx = [torch.randint(0, n, (args.batch,), generator=gen, device=device)
                     for _ in range(args.steps)]

    t1 = time.time()
    model.train()
    last_loss = 0.0
    for step, idx in enumerate(train_idx):
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
        last_loss = loss.item()
    train_s = time.time() - t1
    print(f"[train] final loss={last_loss:.5f}")

    if ema is not None:
        model.load_state_dict(ema.shadow, strict=True)
    model.eval()
    with torch.no_grad():
        xA, xB, xM = te["x_A"], te["x_B"], te["x_M"]
        preds = []
        for s in range(0, xA.shape[0], 64):
            sl = slice(s, s + 64)
            t_b = torch.full((xA[sl].shape[0],), T_MID, device=device)
            out = model(te_mid[sl], t_b, xA[sl], xB[sl], te_uref[sl])
            preds.append(te_mid[sl] + out if resid else out)
        pred = torch.cat(preds)
        p_tr, p_st = psnr(pred, xM), psnr(te_mid, xM)

    gain = (p_tr.mean() - p_st.mean()).item()
    trained = p_tr.mean().item()
    hi_gain = hi_trained = float("nan")
    for b_i, idx in enumerate(strata):
        if idx and b_i == len(strata) - 1:
            it = torch.tensor(idx, device=device)
            hi_gain = (p_tr[it].mean() - p_st[it].mean()).item()
            hi_trained = p_tr[it].mean().item()
            break
    verdict = "DOMINATE" if gain >= 1.0 and hi_gain >= 0.3 else ("WIN" if gain > 0 else "lose")
    print(f"\n=== variant={args.variant} ref={args.reference} size={args.size} "
          f"steps={args.steps} params={count_parameters(model)/1e6:.2f}M (train {train_s:.1f}s) ===")
    print(f"RESULT [{verdict}] variant={args.variant} ref={args.reference} size={args.size} "
          f"steps={args.steps} n_train={args.n_train} wd={args.weight_decay} drop={args.dropout} "
          f"noise={args.input_noise} base_ch={args.base_ch} loss_fn={args.loss} lr_sched={args.lr_schedule} "
          f"train_loss={last_loss:.5f} | trained={trained:.3f} static={p_st.mean():.3f} "
          f"gain={gain:+.3f}dB hi_gain={hi_gain:+.3f}dB hi_trained={hi_trained:.3f} "
          f"resid_closed=nan | SSIM n/a")
    return dict(trained=trained, static=p_st.mean().item(), gain=gain, hi_trained=hi_trained,
                p_tr=p_tr, p_st=p_st, strata=strata, edges=edges)


if __name__ == "__main__":
    main()
