"""
v3 UNet: v2 conditioning (u_ref) PLUS the reference's aligned mean frame.

Input  : (B, 15, H, W) = concat([x_t, x_A, x_B, u_ref, ref_mean], dim=1)
   where ref_mean = reference.mean(x_A, x_B, flow_AB, t)  (warped blend for
   physics, linear blend for brownian). Falls back to the v2 (12ch) or v1 (9ch)
   stacks when the corresponding conditioning tensors are None.

Rationale: at large motion the model only saw the velocity-style u_ref and the
raw, spatially-misaligned x_A / x_B. Feeding the motion-aligned reference frame
gives it the warped content directly, so it can refine an aligned prior rather
than reconstruct alignment through a limited conv receptive field.

Architecture/capacity are identical to the v2 UNet except the input projection
conv gains the extra conditioning channels (negligible param change). Both
references run this same path; only what reference.mean / reference.velocity
returns differs.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from .unet import UNet
from .blocks import ResBlock, Upsample


class UNetV3(UNet):
    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                x_A: torch.Tensor, x_B: torch.Tensor,
                u_ref: torch.Tensor | None = None,
                ref_mean: torch.Tensor | None = None,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x_t, x_A, x_B]
        if u_ref is not None:
            parts.append(u_ref)
        if ref_mean is not None:
            parts.append(ref_mean)
        if extra is not None:
            # v3c: extra reference-derived conditioning channel(s),
            # e.g. an occlusion / warp-reliability map.
            parts.append(extra)
        h = torch.cat(parts, dim=1)
        t_emb = self.time_mlp(t)
        h = self.conv_in(h)
        skips = [h]

        for stage in self.downs:
            for blk in stage:
                h = blk(h, t_emb) if isinstance(blk, ResBlock) else blk(h)
            skips.append(h)

        for blk in self.mid:
            h = blk(h, t_emb) if isinstance(blk, ResBlock) else blk(h)

        for stage in self.ups:
            if isinstance(stage[0], Upsample):
                h = stage[0](h)
                continue
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for blk in stage:
                h = blk(h, t_emb) if isinstance(blk, ResBlock) else blk(h)

        h = self.norm_out(h)
        return self.conv_out(F.silu(h))
