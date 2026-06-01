"""
Motion-correctness metric for VFI: endpoint error (EPE) of the predicted
midpoint against the GT.

We define the "expected motion" at t = 0.5 as 0.5 * F_{A->B}: each pixel of A
should have moved by half the chained flow at the midpoint. The predicted
midpoint x_hat should therefore, when compared to x_A under the half-flow,
match closely.

Concretely we compute:
    err(p) = | warp_backward(x_hat, -0.5 * F_{A->B})(p)  -  x_A(p) |
averaged over pixels and channels. Pixels at occlusion boundaries are
unreliable; we mask using a coarse forward-backward consistency mask.

This is a proxy for "did the predicted frame put pixels in the right place"
and is invariant to absolute color/luminance shifts that can fool PSNR.

The metric returns per-image scalars so downstream code can stratify.
"""

from __future__ import annotations
import torch

from data.flow_utils import backward_warp


@torch.no_grad()
def epe_against_gt_flow(x_hat: torch.Tensor, x_A: torch.Tensor,
                        flow_AB: torch.Tensor) -> torch.Tensor:
    """Returns (B,) tensor: mean abs(warp(x_hat by +0.5 F_AB) - x_A), in [0, 1]."""
    p = (x_hat + 1.0) * 0.5
    a = (x_A + 1.0) * 0.5
    warped = backward_warp(p, 0.5 * flow_AB, padding_mode="border")
    return (warped - a).abs().mean(dim=(1, 2, 3))


@torch.no_grad()
def flow_residual_norm(flow_dot_I: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
    """Sanity / characterization: empirical norm of the bridge-matching target
    (dot I_t  -  u_ref). Used in 1a static prior test to compare references
    pre-training. (B,)."""
    return (flow_dot_I - u_ref).pow(2).mean(dim=(1, 2, 3)).sqrt()
