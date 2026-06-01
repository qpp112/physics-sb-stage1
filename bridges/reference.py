"""
Reference paths for the Schrödinger Bridge.

Each reference exposes:
    mean(x_A, x_B, flow_AB, t)       -> reference image at time t
    velocity(x_A, x_B, flow_AB, t)   -> d/dt of mean (used as u^ref in bridge matching)

Brownian: linear blend, zero drift conceptually (we still expose velocity
    as the time-derivative of the path mean (B - A), which is what enters
    the bridge matching residual target -- see bridges/interpolant.py).

Physics: bidirectional flow-warped blend of A and B,
        bar(x_t) = (1-t) * warp(A, t * F_AB) + t * warp(B, -(1-t) * F_AB)
    where warp() = backward_warp from flow_utils. The time derivative is
    computed numerically (central differences with small dt).

Both references take and return tensors in [-1, 1] range.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch

from data.flow_utils import backward_warp, chain_flows


class Reference(ABC):
    name: str

    @abstractmethod
    def mean(self, x_A: torch.Tensor, x_B: torch.Tensor,
             flow_AB: torch.Tensor, t: torch.Tensor,
             flow_AM: torch.Tensor | None = None,
             flow_MB: torch.Tensor | None = None) -> torch.Tensor: ...

    def velocity(self, x_A, x_B, flow_AB, t, flow_AM=None, flow_MB=None, dt=1e-3):
        """Numerical time derivative of the reference path mean."""
        t_p = torch.clamp(t + dt, 0.0, 1.0)
        t_m = torch.clamp(t - dt, 0.0, 1.0)
        with torch.no_grad():
            mp = self.mean(x_A, x_B, flow_AB, t_p)
            mm = self.mean(x_A, x_B, flow_AB, t_m)
        return (mp - mm) / (t_p - t_m).view(-1, 1, 1, 1).clamp_min(1e-6)


def _t_view(t, b):
    return t.view(b, 1, 1, 1)


class BrownianReference(Reference):
    name = "brownian"

    def mean(self, x_A, x_B, flow_AB, t, flow_AM=None, flow_MB=None):
        b = x_A.shape[0]
        tv = _t_view(t, b)
        return (1.0 - tv) * x_A + tv * x_B

    def velocity(self, x_A, x_B, flow_AB, t, flow_AM=None, flow_MB=None, dt=None):
        # analytic: d/dt [(1-t)A + tB] = B - A
        return x_B - x_A


class PhysicsReference(Reference):
    """Physics path under the rigid-body / constant-velocity prior.

    Construction:
      forward-warped A   :=  warp_backward(x_A, -t * F_AB)
            -- "where does each pixel of A end up at time t under the flow"
      backward-warped B  :=  warp_backward(x_B,  (1-t) * F_AB)
            -- "where does each pixel of B come from at time t"
      mean(t) = (1-t) * forward(A) + t * backward(B)

    The use of backward_warp() with sign-flipped flow for the forward direction is
    a deliberate choice: backward warping fills the output grid densely, while
    forward splatting leaves holes. We pay for this with slightly imperfect
    physics at disocclusions -- acceptable for Stage 1.
    """
    name = "physics"

    def velocity(self, x_A, x_B, flow_AB, t, flow_AM=None, flow_MB=None, dt=None):
        """frozen-warp velocity: override the base numerical d/dt, whose
        chain rule through the warp picks up a flow-magnitude factor
        (u_ref ~ F * grad(I)), making the target ~100x the interpolant
        scale and unlearnable. We drop that term and return the warped
        analog of Brownian's B - A. In SI/bridge-matching any drift
        defines a valid reference SDE, and pred + u_ref is what sampling
        integrates, so this stays consistent across train and sample."""
        b = x_A.shape[0]
        tv = _t_view(t, b)
        warped_A = backward_warp(x_A, -tv * flow_AB)
        warped_B = backward_warp(x_B, (1.0 - tv) * flow_AB)
        return warped_B - warped_A


    def mean(self, x_A, x_B, flow_AB, t, flow_AM=None, flow_MB=None):
        b = x_A.shape[0]
        tv = _t_view(t, b)
        # Pull A toward "its position at time t": sample A using the flow that
        # maps the t-image grid back to A's grid -> -t * F_AB
        warped_A = backward_warp(x_A, -tv * flow_AB)
        # Pull B back toward "its position at time t": sample B using
        # the flow that maps t-image grid forward to B's grid -> +(1-t) * F_AB
        warped_B = backward_warp(x_B, (1.0 - tv) * flow_AB)
        return (1.0 - tv) * warped_A + tv * warped_B


class PhysicsReferenceV2(PhysicsReference):
    """Acceleration-aware physics prior (Stage 2 A1).

    The Stage-1 physics prior assumes constant velocity: it warps A by
    -0.5*F_AB and B by +0.5*F_AB to reach the midpoint. With access to the
    two half-step flows F_{A->M} and F_{M->B} we can estimate curvature and
    apply a second-order (Taylor) correction to the midpoint displacement.

    Expressed on A's grid:
        f1 = F_{A->M}
        f2 = F_{M->B} sampled at the A->M landing  (== chain - f1)
        v  = f1 + f2                  (== F_AB, full-interval velocity)
        a  = f2 - f1                  (acceleration proxy)
        disp_A = 0.5*v + 0.125*a      (A -> midpoint, x(0.5)=0.5 v + 0.125 a)
        disp_B = v - disp_A           (midpoint -> B)

    When motion is constant (f1 == f2 => a == 0) this reduces EXACTLY to the
    Stage-1 physics midpoint. Falls back to V1 if half-flows are unavailable.
    Only the midpoint (t=0.5) is modeled; that is all the prior is used for.
    """
    name = "physics_v2"

    def _mid_disp(self, flow_AB, flow_AM, flow_MB):
        sampled_MB = chain_flows(flow_AM, flow_MB) - flow_AM   # f2 on A grid
        f1, f2 = flow_AM, sampled_MB
        v = f1 + f2
        a = f2 - f1
        disp_A = 0.5 * v + 0.125 * a
        disp_B = v - disp_A
        return disp_A, disp_B

    def mean(self, x_A, x_B, flow_AB, t, flow_AM=None, flow_MB=None):
        if flow_AM is None or flow_MB is None:
            return super().mean(x_A, x_B, flow_AB, t)
        disp_A, disp_B = self._mid_disp(flow_AB, flow_AM, flow_MB)
        warped_A = backward_warp(x_A, -disp_A)
        warped_B = backward_warp(x_B, disp_B)
        return 0.5 * warped_A + 0.5 * warped_B

    def velocity(self, x_A, x_B, flow_AB, t, flow_AM=None, flow_MB=None, dt=None):
        if flow_AM is None or flow_MB is None:
            return super().velocity(x_A, x_B, flow_AB, t)
        disp_A, disp_B = self._mid_disp(flow_AB, flow_AM, flow_MB)
        warped_A = backward_warp(x_A, -disp_A)
        warped_B = backward_warp(x_B, disp_B)
        return warped_B - warped_A


def make_reference(name: str) -> Reference:
    name = name.lower()
    if name == "brownian":
        return BrownianReference()
    if name == "physics":
        return PhysicsReference()
    if name in ("physics_v2", "physicsv2", "physics2"):
        return PhysicsReferenceV2()
    raise ValueError(f"Unknown reference: {name!r}")
