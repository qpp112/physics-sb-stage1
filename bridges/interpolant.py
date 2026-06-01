"""
Empirical interpolant through observed (x_A, x_M, x_B) triplets.

We use a quadratic Bezier whose control point P is chosen so that the curve
passes through M exactly at t = 0.5:

    I_t  = (1-t)^2 A  +  2 t (1-t) P  +  t^2 B
    P    = 2 M - 0.5 (A + B)               (so that I_0.5 = M)
    dI_t = 2 (1-t) (P - A)  +  2 t (B - P)

Sanity:
    I_0   = A                                    ✓
    I_1   = B                                    ✓
    I_0.5 = 0.25 A + 0.5 P + 0.25 B = M          ✓
    If M = 0.5 (A + B) (linear motion):
        P = A + B - 0.5 A - 0.5 B = 0.5 (A + B)
        I_t = (1-t)^2 A + t(1-t)(A+B) + t^2 B = (1-t) A + t B   ✓ (linear)
        dI_t = B - A   (constant)                                 ✓

The training target in bridge matching is:
        dot{I}_t  -  u^ref(I_t, t)
where u^ref is the reference path's velocity. See bridges/reference.py.
"""

from __future__ import annotations
import torch


def control_point(x_A: torch.Tensor, x_M: torch.Tensor, x_B: torch.Tensor) -> torch.Tensor:
    """P such that quadratic Bezier(A, P, B) passes through M at t=0.5."""
    return 2.0 * x_M - 0.5 * (x_A + x_B)


def interpolate(x_A, x_M, x_B, t):
    """I_t. t: (B,) tensor in [0, 1]."""
    P = control_point(x_A, x_M, x_B)
    tv = t.view(-1, 1, 1, 1)
    return (1 - tv) ** 2 * x_A + 2 * tv * (1 - tv) * P + tv ** 2 * x_B


def velocity(x_A, x_M, x_B, t):
    """dot{I}_t. t: (B,) tensor in [0, 1]."""
    P = control_point(x_A, x_M, x_B)
    tv = t.view(-1, 1, 1, 1)
    return 2 * (1 - tv) * (P - x_A) + 2 * tv * (x_B - P)


def sample_t(batch_size: int, mode: str = "uniform",
             concentration: float = 4.0, device="cpu") -> torch.Tensor:
    """Sample t in [0, 1].

    'uniform'           -> U(0, 1)
    'beta_peak_at_half' -> Beta(c, c) with c = concentration; peak at 0.5, supports the
                           prior that t=0.5 is what we evaluate. We avoid concentrating
                           too much because we still want to learn the full path.
    """
    if mode == "uniform":
        return torch.rand(batch_size, device=device)
    if mode == "beta_peak_at_half":
        m = torch.distributions.Beta(concentration, concentration)
        return m.sample((batch_size,)).to(device)
    raise ValueError(f"Unknown t sampling mode: {mode!r}")
