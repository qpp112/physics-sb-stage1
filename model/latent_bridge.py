"""
Option 3 + cross-attention + latent bridge.

Staged, evidence-driven build (see RESULTS.md "Architecture cycling"):
  S0  SharedAE         : shared encoder/decoder with pixel skips (faithful codec).
                         Lesson from feat_resid: a decode path that isn't a faithful
                         codec, without pixel skips, overfits and blurs. Gate: recon PSNR.
  S1  LatentPrior      : + cross-attention @ bottleneck + physics embedding +
                         feature-space f_interp, decoded as a RESIDUAL on the
                         pixel-warp floor. Gate: midpoint PSNR > physics-V1.
  S2  LatentBridge     : + latent denoiser; z_t = z_mid + sigma*eps (noised from the
                         PRIOR, never a GT interpolant). Gate: bridge > S1 one-shot.

All tensors in [-1, 1]. Bottleneck is at /8 resolution.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.synth import Res, _gn, _flow_to_scale
from data.flow_utils import backward_warp


# --------------------------------------------------------------------------
# S0: shared autoencoder with pixel skips
# --------------------------------------------------------------------------

class SharedAE(nn.Module):
    """Shared-weight encoder/decoder. encode(x) -> (z@/8, skips=(s0/1, s1/2, s2/4))."""

    def __init__(self, base: int = 48, fdim: int = 128, drop: float = 0.0):
        super().__init__()
        self.base, self.fdim = base, fdim
        self.c0 = nn.Sequential(nn.Conv2d(3, base, 3, 1, 1), Res(base, base, drop))
        self.d1 = nn.Sequential(nn.Conv2d(base, base * 2, 3, 2, 1),
                                Res(base * 2, base * 2, drop))
        self.d2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, 2, 1),
                                Res(base * 4, base * 4, drop))
        self.d3 = nn.Sequential(nn.Conv2d(base * 4, fdim, 3, 2, 1),
                                Res(fdim, fdim, drop))
        self.u2 = nn.ConvTranspose2d(fdim, base * 4, 4, 2, 1)
        self.r2 = Res(base * 4 * 2, base * 4, drop)
        self.u1 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.r1 = Res(base * 2 * 2, base * 2, drop)
        self.u0 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.r0 = Res(base * 2, base, drop)
        self.head = nn.Sequential(_gn(base), nn.SiLU(), nn.Conv2d(base, 3, 3, 1, 1))

    def encode(self, x):
        s0 = self.c0(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        z = self.d3(s2)
        return z, (s0, s1, s2)

    def decode(self, z, skips):
        s0, s1, s2 = skips
        h = self.r2(torch.cat([self.u2(z), s2], 1))
        h = self.r1(torch.cat([self.u1(h), s1], 1))
        h = self.r0(torch.cat([self.u0(h), s0], 1))
        return self.head(h)

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode(z, skips)


# --------------------------------------------------------------------------
# Cross-attention @ bottleneck + physics embedding (for S1)
# --------------------------------------------------------------------------

class CrossAttn(nn.Module):
    """Bidirectional cross-attention between z_A and z_B at bottleneck res.
    Updates z_A using z_B (and vice versa) to establish correspondence."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.h = heads
        self.norm = _gn(dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _attend(self, x_from, x_to):
        b, c, h, w = x_from.shape
        q = self.q(self.norm(x_from)).reshape(b, self.h, c // self.h, h * w)
        k = self.k(self.norm(x_to)).reshape(b, self.h, c // self.h, h * w)
        v = self.v(self.norm(x_to)).reshape(b, self.h, c // self.h, h * w)
        attn = torch.softmax((q.transpose(-1, -2) @ k) / (c // self.h) ** 0.5, dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(b, c, h, w)
        return self.proj(out)

    def forward(self, zA, zB):
        return zA + self._attend(zA, zB), zB + self._attend(zB, zA)


class PhysicsEmbed(nn.Module):
    """Map flow statistics -> per-sample embedding -> FiLM (scale, shift) on features."""

    def __init__(self, dim: int, edim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, edim), nn.SiLU(),
                                 nn.Linear(edim, edim), nn.SiLU())
        self.film = nn.Linear(edim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, flow_AB):
        mag = flow_AB.pow(2).sum(1, keepdim=True).sqrt()
        feat = torch.stack([mag.mean((1, 2, 3)), mag.amax((1, 2, 3)),
                            flow_AB[:, 0].mean((1, 2)), flow_AB[:, 1].mean((1, 2))], dim=1)
        e = self.mlp(feat)
        scale, shift = self.film(e).chunk(2, dim=1)
        return scale, shift, e


# --------------------------------------------------------------------------
# S1: latent prior (residual on the pixel-warp floor)
# --------------------------------------------------------------------------

class LatentPrior(nn.Module):
    """Encode A,B -> cross-attend -> warp+blend bottleneck by flow (FiLM by e_phys)
    -> decode a RESIDUAL on the pixel-warp floor, with warped-blended pixel skips."""
    trainable = True
    SCALE = 0.125          # bottleneck is /8

    def __init__(self, base: int = 48, fdim: int = 128, drop: float = 0.0):
        super().__init__()
        self.ae = SharedAE(base, fdim, drop)
        self.xattn = CrossAttn(fdim, heads=4)
        self.phys = PhysicsEmbed(fdim)
        self.refine = Res(fdim, fdim, drop)
        nn.init.zeros_(self.ae.head[-1].weight)   # residual head -> floor at init
        nn.init.zeros_(self.ae.head[-1].bias)

    @staticmethod
    def _warp_blend(a, b, flow_scaled):
        return 0.5 * backward_warp(a, -0.5 * flow_scaled) + \
               0.5 * backward_warp(b, 0.5 * flow_scaled)

    def _floor(self, x_A, x_B, flow_AB):
        return self._warp_blend(x_A, x_B, flow_AB)

    def encode_prior(self, x_A, x_B, flow_AB):
        """Produce the latent prior z_mid + everything S2's denoiser needs."""
        zA, sA = self.ae.encode(x_A)
        zB, sB = self.ae.encode(x_B)
        zA, zB = self.xattn(zA, zB)
        scale, shift, e = self.phys(flow_AB)
        s = scale.view(-1, scale.shape[1], 1, 1)
        sh = shift.view(-1, shift.shape[1], 1, 1)
        f8 = _flow_to_scale(flow_AB, self.SCALE)
        z_mid = self.refine(self._warp_blend(zA, zB, f8) * (1 + s) + sh)
        skips = []
        for lvl, (a, b) in enumerate(zip(sA, sB)):
            scl = 1.0 / (2 ** lvl)          # s0:/1, s1:/2, s2:/4
            skips.append(self._warp_blend(a, b, _flow_to_scale(flow_AB, scl)))
        floor = self._floor(x_A, x_B, flow_AB)
        return {"z_mid": z_mid, "zA": zA, "zB": zB, "e_phys": e,
                "skips": tuple(skips), "floor": floor}

    def decode_mid(self, z, skips, floor):
        return floor + self.ae.decode(z, skips)

    def forward(self, x_A, x_B, flow_AB):
        p = self.encode_prior(x_A, x_B, flow_AB)
        mid = self.decode_mid(p["z_mid"], p["skips"], p["floor"])
        return {"mid": mid, "z_mid": p["z_mid"], "skips": p["skips"],
                "floor": p["floor"], "flow_AM": None}


# --------------------------------------------------------------------------
# S2: latent bridge (denoise z_mid -> z_M in latent space)
# --------------------------------------------------------------------------

def _time_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) *
                      (torch.log(torch.tensor(10000.0)) / max(half - 1, 1)))
    a = t.float().view(-1, 1) * freqs.view(1, -1)
    return torch.cat([a.sin(), a.cos()], dim=1)


class LatentDenoiser(nn.Module):
    """Predict the residual (z_M - z_mid) from z_t = z_mid + sigma*eps.

    Conditioned on (z_A, z_B), the physics embedding e_phys (FiLM) and time t.
    Zero-init output head => residual 0 => z_M_hat = z_mid => exactly the S1
    prior (strict floor). The bridge can only improve on the one-shot prior."""

    def __init__(self, dim: int, edim: int = 128, n_blocks: int = 4, drop: float = 0.0):
        super().__init__()
        self.tdim = edim
        self.tproj = nn.Sequential(nn.Linear(edim, edim), nn.SiLU(),
                                   nn.Linear(edim, dim))
        self.inp = nn.Conv2d(3 * dim, dim, 3, 1, 1)
        self.blocks = nn.ModuleList([Res(dim, dim, drop) for _ in range(n_blocks)])
        self.film = nn.Linear(edim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.norm = _gn(dim)
        self.out = nn.Conv2d(dim, dim, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_t, t, z_A, z_B, e_phys):
        h = self.inp(torch.cat([z_t, z_A, z_B], 1))
        temb = self.tproj(_time_emb(t, self.tdim)).view(h.shape[0], -1, 1, 1)
        h = h + temb
        scale, shift = self.film(e_phys).chunk(2, dim=1)
        h = h * (1 + scale.view(h.shape[0], -1, 1, 1)) + shift.view(h.shape[0], -1, 1, 1)
        for b in self.blocks:
            h = b(h)
        return self.out(F.silu(self.norm(h)))


class LatentBridge(nn.Module):
    """S1 prior + latent denoiser. Trains the bridge to refine z_mid -> z_M;
    the prior can be frozen (default) so the C-GATE isolates the bridge's value."""
    trainable = True

    def __init__(self, base: int = 48, fdim: int = 128, drop: float = 0.0,
                 freeze_prior: bool = True):
        super().__init__()
        self.prior = LatentPrior(base, fdim, drop)
        self.denoiser = LatentDenoiser(fdim, edim=128, n_blocks=4, drop=drop)
        self.freeze_prior = freeze_prior
        if freeze_prior:
            for p in self.prior.parameters():
                p.requires_grad = False

    def encode_target(self, x_M):
        with torch.no_grad():
            z_M, _ = self.prior.ae.encode(x_M)
        return z_M

    def forward(self, x_A, x_B, flow_AB, t=None, sigma=0.0):
        ctx = torch.no_grad() if self.freeze_prior else torch.enable_grad()
        with ctx:
            p = self.prior.encode_prior(x_A, x_B, flow_AB)
        z_mid = p["z_mid"]
        b = z_mid.shape[0]
        if t is None:
            t = torch.ones(b, device=z_mid.device)
        z_t = z_mid + sigma * torch.randn_like(z_mid) if sigma > 0 else z_mid
        r = self.denoiser(z_t, t, p["zA"], p["zB"], p["e_phys"])
        z_M_hat = z_mid + r
        mid = self.prior.decode_mid(z_M_hat, p["skips"], p["floor"])
        return {"mid": mid, "z_mid": z_mid, "z_M_hat": z_M_hat,
                "skips": p["skips"], "floor": p["floor"], "flow_AM": None}


def make_latent(stage: str, base: int = 48, fdim: int = 128, dropout: float = 0.0,
                freeze_prior: bool = True):
    stage = stage.lower()
    if stage == "s0":
        return SharedAE(base, fdim, dropout)
    if stage == "s1":
        return LatentPrior(base, fdim, dropout)
    if stage == "s2":
        return LatentBridge(base, fdim, dropout, freeze_prior=freeze_prior)
    raise ValueError(f"Unknown latent stage {stage!r}")
