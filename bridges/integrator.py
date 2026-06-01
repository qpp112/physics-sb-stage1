"""
SDE integrator for bridge inference.

The bridge state evolves as:
    dx_t = (u^ref(x_t, t) + v_theta(x_t, t, x_A, x_B)) dt  +  sigma sqrt(t(1-t)) dW_t

For pilot inference we use the deterministic-mean integrator (no noise) by default --
it's cheap, low-variance, and what we'll report on. We also expose a stochastic
Euler-Maruyama path for ablations and the σ-sweep in the full plan.

This file also includes a 1D toy Brownian bridge analytic check used by
exps/sanity_checks.py:  E[x_t | x_0, x_1] = (1-t) x_0 + t x_1.
"""

from __future__ import annotations
from typing import Callable, Optional
import torch


@torch.no_grad()
def integrate(velocity_fn: Callable,
              reference,
              x_A: torch.Tensor,
              x_B: torch.Tensor,
              flow_AB: torch.Tensor,
              t_grid: torch.Tensor,
              x0: Optional[torch.Tensor] = None,
              sigma: float = 0.0,
              t_eval: Optional[float] = None) -> torch.Tensor:
    """
    Run the SDE from t = t_grid[0] up to t_eval (or t_grid[-1] if None),
    starting at x0 (defaults to reference.mean at t_grid[0]).

    velocity_fn(x_t, t_batched, x_A, x_B, u_ref) -> learned residual drift, same shape as x_t.
    reference: BrownianReference or PhysicsReference (provides u^ref).
    t_grid: (T,) increasing schedule from t0 to t1.
    sigma: noise scale (0 = deterministic).

    Returns x at t_eval (or final).
    """
    device = x_A.device
    b = x_A.shape[0]
    if x0 is None:
        t0 = t_grid[0].view(1).expand(b).to(device)
        x = reference.mean(x_A, x_B, flow_AB, t0)
    else:
        x = x0

    t_eval_v = float(t_eval) if t_eval is not None else float(t_grid[-1])

    for k in range(len(t_grid) - 1):
        t = float(t_grid[k])
        t_next = float(t_grid[k + 1])
        dt = t_next - t

        t_b = torch.full((b,), t, device=device)
        u_ref = reference.velocity(x_A, x_B, flow_AB, t_b)
        # u_ref is t-dependent and recomputed each step; the model is conditioned
        # on this same reference drift so train/sample stay consistent.
        v_res = velocity_fn(x, t_b, x_A, x_B, u_ref)
        drift = u_ref + v_res

        x = x + drift * dt
        if sigma > 0.0 and t > 0.0 and t_next < 1.0:
            std = sigma * (t * (1 - t)) ** 0.5 * (dt ** 0.5)
            x = x + std * torch.randn_like(x)

        # if t_eval lies in [t, t_next], interpolate; otherwise continue
        if t <= t_eval_v <= t_next:
            alpha = (t_eval_v - t) / max(dt, 1e-8)
            x_eval_drift = u_ref + v_res
            x_eval = x - drift * dt + x_eval_drift * (alpha * dt)
            return x_eval

    return x


# ----------------------------- toy 1D test -----------------------------

@torch.no_grad()
def toy_1d_brownian_bridge_test(n_paths: int = 20000, n_steps: int = 200,
                                 sigma: float = 1.0, seed: int = 0):
    """
    Simulate the unconditioned Brownian bridge SDE for x in R, conditioned on
    x_0 = 0, x_1 = 1.

    Analytic: E[x_t] = t,  Var[x_t] = sigma^2 * t * (1 - t).

    Bridge SDE drift (conditioned on endpoint at T=1):
        u(x, t) = (x_1 - x) / (1 - t)
    Integrated with Euler-Maruyama.

    Returns dict of (analytic, empirical) means and variances at t = 0.5.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n_paths)
    dt = 1.0 / n_steps
    target = 1.0

    for k in range(n_steps - 1):
        t = k * dt
        drift = (target - x) / (1 - t)
        noise = sigma * (dt ** 0.5) * torch.randn(n_paths, generator=g)
        x = x + drift * dt + noise
        # snapshot at t closest to 0.5
        if abs((k + 1) * dt - 0.5) < dt / 2:
            x_mid = x.clone()
    mean_emp = x_mid.mean().item()
    var_emp = x_mid.var().item()
    mean_ana = 0.5
    var_ana = sigma ** 2 * 0.5 * 0.5
    return dict(mean_emp=mean_emp, var_emp=var_emp,
                mean_ana=mean_ana, var_ana=var_ana,
                mean_err=abs(mean_emp - mean_ana),
                var_err=abs(var_emp - var_ana))
