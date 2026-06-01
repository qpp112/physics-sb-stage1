"""
Shared inference utilities. Loads a checkpoint + config, runs the bridge
integrator to produce a midpoint prediction at t = 0.5, computes metrics.
"""

from __future__ import annotations
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

from data import FlyingThings3D
from bridges.reference import make_reference
from bridges.integrator import integrate
from model import UNet
from metrics.image_metrics import psnr as psnr_fn


def build_model_from_cfg(cfg: dict, image_size: int) -> UNet:
    m = cfg["model"]
    in_channels = 9 + (3 if cfg["bridge"].get("condition_on_reference", True) else 0)
    return UNet(
        in_channels=in_channels, out_channels=3,
        base_channels=m["base_channels"],
        channel_mult=tuple(m["channel_mult"]),
        num_res_blocks=m["num_res_blocks"],
        attn_resolutions=tuple(m["attn_resolutions"]),
        time_emb_dim=m["time_emb_dim"],
        dropout=0.0,                                   # always 0 at inference
        image_size=image_size,
    )


def _load_ckpt(model: torch.nn.Module, ckpt_path: str, device):
    state = torch.load(ckpt_path, map_location=device)
    sd = state["ema_state_dict"] if "ema_state_dict" in state else state["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_midpoint(model, reference, batch, num_steps: int,
                     sigma: float, t_eval: float = 0.5,
                     xt_corruption: str | None = None,
                     condition_on_reference: bool = True,
                     device="cuda") -> torch.Tensor:
    x_A = batch["x_A"].to(device)
    x_B = batch["x_B"].to(device)
    flow_AB = batch["flow_AB"].to(device)

    # Build integration grid: start at 0, end past t_eval but not at 1 (cleaner).
    # We integrate from 0 to 1 with num_steps and read off at t_eval.
    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    def vfun(x, t_b, xA, xB, u_ref):
        # Corrupt ONLY the bridge state x_t (gate-3 ablation); u_ref is the true
        # reference drift conditioning and must be passed through untouched.
        if xt_corruption == "zero":
            x_in = torch.zeros_like(x)
        elif xt_corruption == "noise":
            x_in = torch.randn_like(x)
        else:
            x_in = x
        return model(x_in, t_b, xA, xB, u_ref if condition_on_reference else None)

    return integrate(
        velocity_fn=vfun, reference=reference,
        x_A=x_A, x_B=x_B, flow_AB=flow_AB,
        t_grid=t_grid, sigma=sigma, t_eval=t_eval,
    )


def run_inference(cfg: dict, ckpt_path: str, n_samples: int,
                  xt_corruption: str | None = None,
                  device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["data"]["size"]
    model = build_model_from_cfg(cfg, image_size).to(device)
    _load_ckpt(model, ckpt_path, device)

    ds = FlyingThings3D(
        root=cfg["data"]["root"], split="TEST", size=image_size,
        min_flow=cfg["data"]["min_flow"], max_clips=n_samples,
        pass_name=cfg["data"]["pass"], seed=0,
    )
    dl = DataLoader(ds, batch_size=4, num_workers=cfg["data"]["num_workers"])

    reference = make_reference(cfg["bridge"]["reference"])
    cond_ref = cfg["bridge"].get("condition_on_reference", True)

    psnrs = []
    for batch in dl:
        x_hat = predict_midpoint(
            model, reference, batch,
            num_steps=cfg["eval"]["num_steps"],
            sigma=0.0, t_eval=0.5,
            xt_corruption=xt_corruption,
            condition_on_reference=cond_ref, device=device,
        )
        psnrs.append(psnr_fn(x_hat, batch["x_M"].to(device)).cpu().numpy())
    return np.concatenate(psnrs)


@torch.no_grad()
def run_inference_identity(cfg: dict, ckpt_path: str, n_samples: int = 8,
                            device: str | None = None) -> float:
    """Synthetic identity test: pick any test sample, replace B with A and zero the flow.
    True midpoint should be A. Returns PSNR(M_hat, A) averaged."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["data"]["size"]
    model = build_model_from_cfg(cfg, image_size).to(device)
    _load_ckpt(model, ckpt_path, device)

    ds = FlyingThings3D(
        root=cfg["data"]["root"], split="TEST", size=image_size,
        min_flow=cfg["data"]["min_flow"], max_clips=n_samples,
        pass_name=cfg["data"]["pass"], seed=0,
    )
    reference = make_reference(cfg["bridge"]["reference"])
    cond_ref = cfg["bridge"].get("condition_on_reference", True)
    psnrs = []
    for i in range(min(n_samples, len(ds))):
        s = ds[i]
        x_A = s["x_A"].unsqueeze(0).to(device)
        batch = dict(
            x_A=x_A, x_B=x_A.clone(),
            flow_AB=torch.zeros_like(s["flow_AB"]).unsqueeze(0).to(device),
        )
        x_hat = predict_midpoint(
            model, reference, batch,
            num_steps=cfg["eval"]["num_steps"], sigma=0.0, t_eval=0.5,
            condition_on_reference=cond_ref, device=device,
        )
        psnrs.append(psnr_fn(x_hat, x_A).item())
    return float(np.mean(psnrs))
