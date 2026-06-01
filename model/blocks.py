"""
UNet building blocks. DDPM-style: sinusoidal time embedding, residual blocks
conditioned via FiLM, multi-head self-attention at the low-resolution stages.

Kept minimal so the parameter count is tractable and so the comparison
baseline-vs-physics is not confounded by architectural complexity.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- time embedding -----------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) in [0, 1] -> (B, dim).
        We scale t by max_period to span the typical sinusoidal positional range."""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device).float() / half
        )
        args = t.float()[:, None] * freqs[None] * 2 * math.pi
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t):
        return self.net(t)


# ----------------------------- residual block -----------------------------

def _gn(c, num_groups=32):
    # pick largest divisor of c that is <= num_groups (down to 1)
    g = min(num_groups, c)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_c)
        self.norm2 = _gn(out_c)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ----------------------------- attention -----------------------------

class SelfAttention(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        assert c % heads == 0
        self.h = heads
        self.norm = _gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.h, c // self.h, n).unbind(1)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) / (c // self.h) ** 0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


# ----------------------------- up / down -----------------------------

class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)
