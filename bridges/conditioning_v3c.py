"""
v3c conditioning: a per-pixel occlusion / warp-reliability map.

We only have the forward chained flow F_{A->B} (no reverse flow), so a classic
forward-backward geometric occlusion check is unavailable. Instead we use the
photometric-consistency flavor of occlusion masking (cf. FlowVid / FB-consistency
methods): the disagreement between the reference's two warped views at time t.

    mask(t) = mean_c | warped_A(t) - warped_B(t) |          # (B, 1, H, W)

High where the two warps disagree -- occlusion, disocclusion, or wrong flow --
i.e. exactly where the reference prior (their blend, reference.mean) is
unreliable and the model should inpaint from the bridge state / endpoints.

This is reference-specific by construction, which keeps the project invariant
"only the reference differs" intact:
  * physics  -> warped views differ at occlusion boundaries (true occlusion signal)
  * brownian -> no warp, so it reduces to |A - B| (motion magnitude only)
so the signal differentially benefits the physics path. It mirrors exactly how
reference.mean / reference.velocity already branch by reference type; nothing in
bridges/reference.py is modified.
"""

from __future__ import annotations
import torch

from data.flow_utils import backward_warp


@torch.no_grad()
def reference_occlusion_map(reference, x_A: torch.Tensor, x_B: torch.Tensor,
                            flow_AB: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) soft occlusion/reliability map at time t. Recompute per step
    (t-dependent), same as u_ref and ref_mean."""
    b = x_A.shape[0]
    tv = t.view(b, 1, 1, 1)
    if getattr(reference, "name", "") == "physics":
        warped_A = backward_warp(x_A, -tv * flow_AB)
        warped_B = backward_warp(x_B, (1.0 - tv) * flow_AB)
    else:                                   # brownian / no-warp references
        warped_A, warped_B = x_A, x_B
    return (warped_A - warped_B).abs().mean(dim=1, keepdim=True)
