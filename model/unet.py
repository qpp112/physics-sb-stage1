"""
UNet for predicting the bridge residual drift v_theta(x_t, t, x_A, x_B).

Input  : (B, 9, H, W)   = concat([x_t, x_A, x_B], dim=1)            all in [-1, 1]
   or    (B, 12, H, W)  = concat([x_t, x_A, x_B, u_ref], dim=1)     when conditioning
                          on the reference drift (condition_on_reference=True).
Output : (B, 3, H, W)  =  residual drift  (in same units as image / unit-time)
Time   : (B,)          in [0, 1]

Defaults:
    base 64, mult (1, 2, 4, 4) -> channels (64, 128, 256, 256)
    2 ResBlocks per stage, attention at the lowest two resolutions
    Time-embedding dim 256
  -> ~30M parameters at 256x256
"""

from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

from .blocks import (TimeMLP, ResBlock, SelfAttention, Downsample, Upsample, _gn)


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 9,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 channel_mult: Sequence[int] = (1, 2, 4, 4),
                 num_res_blocks: int = 2,
                 attn_resolutions: Sequence[int] = (16, 8),
                 time_emb_dim: int = 256,
                 dropout: float = 0.1,
                 image_size: int = 256):
        super().__init__()
        self.time_mlp = TimeMLP(time_emb_dim, time_emb_dim)
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- down path ---
        ch = base_channels
        cur_res = image_size
        skip_channels = [ch]
        self.downs = nn.ModuleList()

        for i, m in enumerate(channel_mult):
            out_ch = base_channels * m
            for _ in range(num_res_blocks):
                blocks = [ResBlock(ch, out_ch, time_emb_dim, dropout)]
                ch = out_ch
                if cur_res in attn_resolutions:
                    blocks.append(SelfAttention(ch))
                self.downs.append(nn.ModuleList(blocks))
                skip_channels.append(ch)
            if i != len(channel_mult) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                cur_res //= 2

        # --- bottleneck ---
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout),
            SelfAttention(ch),
            ResBlock(ch, ch, time_emb_dim, dropout),
        ])

        # --- up path ---
        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * m
            for _ in range(num_res_blocks + 1):
                in_ch = ch + skip_channels.pop()
                blocks = [ResBlock(in_ch, out_ch, time_emb_dim, dropout)]
                ch = out_ch
                if cur_res in attn_resolutions:
                    blocks.append(SelfAttention(ch))
                self.ups.append(nn.ModuleList(blocks))
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(ch)]))
                cur_res *= 2

        # --- output head ---
        self.norm_out = _gn(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                x_A: torch.Tensor, x_B: torch.Tensor,
                u_ref: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x_t, x_A, x_B]
        if u_ref is not None:
            parts.append(u_ref)
        h = torch.cat(parts, dim=1)
        t_emb = self.time_mlp(t)
        h = self.conv_in(h)
        skips = [h]

        for stage in self.downs:
            for blk in stage:
                if isinstance(blk, ResBlock):
                    h = blk(h, t_emb)
                elif isinstance(blk, Downsample):
                    h = blk(h)
                else:                                       # SelfAttention
                    h = blk(h)
            if not isinstance(stage[-1], Downsample) or len(stage) > 1:
                # res-block stages push their final h onto skip stack
                skips.append(h)
            else:
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
                if isinstance(blk, ResBlock):
                    h = blk(h, t_emb)
                else:
                    h = blk(h)

        h = self.norm_out(h)
        return self.conv_out(torch.nn.functional.silu(h))


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
