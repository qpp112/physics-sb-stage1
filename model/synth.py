"""
Architecture-cycling building blocks for the midpoint prior.

We cycle several candidate architectures for predicting the midpoint frame
M from (x_A, x_B, F_{A->B}). Each architecture exposes the SAME interface:

    forward(x_A, x_B, flow_AB) -> dict(mid=Tensor[B,3,H,W], flow_AM=Optional)

so they can be trained + ranked identically by exps/arch_cycle.py against the
physics-V1 static prior (20.31 dB @128, the Stage-1 floor).

Registered architectures
-------------------------
  physics_v1  : Stage-1 constant-velocity warp+blend.   (no params, the floor)
  flow_synth  : Option 2. Learn the intermediate displacement as a RESIDUAL
                around the constant-velocity field (0.5*F_AB), then warp+blend.
                Zero-init head => EXACTLY the physics-V1 midpoint, so it can
                only improve on the floor (same trick that made refine_resid work).
  feat_warp   : Option 3. Encode A,B -> features, warp/blend features by the
                (downscaled) flow, decode to RGB. Robust to occlusion because
                features are semantically smooth. Pure latent path (Stage 2C scaffold).
  feat_flow   : Option 2+3. Encode, predict a feature-space residual flow,
                warp features by it, decode.

All tensors are in [-1, 1].
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.flow_utils import backward_warp


def _gn(c: int) -> nn.GroupNorm:
    g = 8
    while c % g:
        g //= 2
    return nn.GroupNorm(g, c)


class Res(nn.Module):
    def __init__(self, i: int, o: int, drop: float = 0.0):
        super().__init__()
        self.b1 = nn.Sequential(_gn(i), nn.SiLU(), nn.Conv2d(i, o, 3, 1, 1))
        self.b2 = nn.Sequential(_gn(o), nn.SiLU(), nn.Dropout(drop),
                                nn.Conv2d(o, o, 3, 1, 1))
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x):
        return self.b2(self.b1(x)) + self.skip(x)


class TinyUNet(nn.Module):
    """Compact 3-level UNet (no temporal embedding) -> out_ch channels."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 48,
                 drop: float = 0.0, zero_out: bool = True):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.cin = nn.Conv2d(in_ch, c1, 3, 1, 1)
        self.e1 = Res(c1, c1, drop)
        self.d1 = nn.Conv2d(c1, c1, 3, 2, 1)
        self.e2 = Res(c1, c2, drop)
        self.d2 = nn.Conv2d(c2, c2, 3, 2, 1)
        self.mid = Res(c2, c3, drop)
        self.u2 = nn.ConvTranspose2d(c3, c2, 4, 2, 1)
        self.r2 = Res(c2 + c2, c2, drop)
        self.u1 = nn.ConvTranspose2d(c2, c1, 4, 2, 1)
        self.r1 = Res(c1 + c1, c1, drop)
        self.norm = _gn(c1)
        self.cout = nn.Conv2d(c1, out_ch, 3, 1, 1)
        if zero_out:
            nn.init.zeros_(self.cout.weight)
            nn.init.zeros_(self.cout.bias)

    def forward(self, x):
        h1 = self.e1(self.cin(x))
        h2 = self.e2(self.d1(h1))
        hm = self.mid(self.d2(h2))
        u2 = self.r2(torch.cat([self.u2(hm), h2], 1))
        u1 = self.r1(torch.cat([self.u1(u2), h1], 1))
        return self.cout(F.silu(self.norm(u1)))


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------

class PhysicsV1(nn.Module):
    """Static constant-velocity warp+blend (the floor). No parameters."""
    trainable = False

    def forward(self, x_A, x_B, flow_AB):
        warped_A = backward_warp(x_A, -0.5 * flow_AB)
        warped_B = backward_warp(x_B, 0.5 * flow_AB)
        return {"mid": 0.5 * warped_A + 0.5 * warped_B, "flow_AM": None}


class FlowSynth(nn.Module):
    """Option 2. Predict A->M displacement as a residual around 0.5*F_AB.

    d = 0.5 * F_AB + net(x_A, x_B, F_AB)
    mid = 0.5 * warp(x_A, -d) + 0.5 * warp(x_B, F_AB - d)

    Zero-init head => d = 0.5*F_AB => EXACTLY physics-V1 (strict floor)."""
    trainable = True

    def __init__(self, base: int = 48, drop: float = 0.0):
        super().__init__()
        self.net = TinyUNet(in_ch=3 + 3 + 2, out_ch=2, base=base, drop=drop,
                            zero_out=True)

    def forward(self, x_A, x_B, flow_AB):
        d = 0.5 * flow_AB + self.net(torch.cat([x_A, x_B, flow_AB], 1))
        warped_A = backward_warp(x_A, -d)
        warped_B = backward_warp(x_B, flow_AB - d)
        return {"mid": 0.5 * warped_A + 0.5 * warped_B, "flow_AM": d}


class _Enc(nn.Module):
    def __init__(self, in_ch: int, fdim: int, base: int = 48, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1), Res(base, base, drop),
            nn.Conv2d(base, base * 2, 3, 2, 1), Res(base * 2, base * 2, drop),
            nn.Conv2d(base * 2, fdim, 3, 2, 1), Res(fdim, fdim, drop),
        )

    def forward(self, x):
        return self.net(x)


class _Dec(nn.Module):
    def __init__(self, fdim: int, out_ch: int, base: int = 48, drop: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            Res(fdim, fdim, drop),
            nn.ConvTranspose2d(fdim, base * 2, 4, 2, 1), Res(base * 2, base * 2, drop),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), Res(base, base, drop),
        )
        self.head = nn.Sequential(_gn(base), nn.SiLU(),
                                  nn.Conv2d(base, out_ch, 3, 1, 1))

    def forward(self, z):
        return self.head(self.body(z))


def _flow_to_scale(flow_AB: torch.Tensor, scale: float) -> torch.Tensor:
    """Downscale a pixel-unit flow field to a coarser grid (features)."""
    f = F.interpolate(flow_AB, scale_factor=scale, mode="bilinear",
                      align_corners=False)
    return f * scale


class FeatWarp(nn.Module):
    """Option 3. Warp/blend in feature space, decode to RGB. /4 feature grid."""
    trainable = True
    SCALE = 0.25

    def __init__(self, base: int = 48, fdim: int = 96, drop: float = 0.0):
        super().__init__()
        self.enc = _Enc(3, fdim, base, drop)
        self.dec = _Dec(fdim, 3, base, drop)

    def forward(self, x_A, x_B, flow_AB):
        zA, zB = self.enc(x_A), self.enc(x_B)
        f = _flow_to_scale(flow_AB, self.SCALE)
        zA_m = backward_warp(zA, -0.5 * f)
        zB_m = backward_warp(zB, 0.5 * f)
        z_mid = 0.5 * zA_m + 0.5 * zB_m
        return {"mid": self.dec(z_mid), "flow_AM": None}


class FeatFlow(nn.Module):
    """Option 2+3. Predict a feature-space residual flow, warp features, decode."""
    trainable = True
    SCALE = 0.25

    def __init__(self, base: int = 48, fdim: int = 96, drop: float = 0.0):
        super().__init__()
        self.enc = _Enc(3, fdim, base, drop)
        self.flow_net = TinyUNet(in_ch=2 * fdim + 2, out_ch=2, base=base,
                                 drop=drop, zero_out=True)
        self.dec = _Dec(fdim, 3, base, drop)

    def forward(self, x_A, x_B, flow_AB):
        zA, zB = self.enc(x_A), self.enc(x_B)
        f = _flow_to_scale(flow_AB, self.SCALE)
        d = 0.5 * f + self.flow_net(torch.cat([zA, zB, f], 1))
        zA_m = backward_warp(zA, -d)
        zB_m = backward_warp(zB, f - d)
        z_mid = 0.5 * zA_m + 0.5 * zB_m
        return {"mid": self.dec(z_mid), "flow_AM": None}


class FlowSynthOcc(nn.Module):
    """Option 2 + occlusion. Learn the residual flow AND a per-pixel blend mask.

    o = net(x_A, x_B, F_AB)              -> 3 channels (2 flow residual, 1 logit)
    d = 0.5*F_AB + o[:, :2]
    a = sigmoid(o[:, 2:3])               -> zero-init => 0.5
    mid = a * warp(x_A, -d) + (1-a) * warp(x_B, F_AB - d)

    Zero-init head => d=0.5*F_AB, a=0.5 => EXACTLY physics-V1 (strict floor).
    The mask lets the model pick A or B where one side is occluded."""
    trainable = True

    def __init__(self, base: int = 48, drop: float = 0.0):
        super().__init__()
        self.net = TinyUNet(in_ch=3 + 3 + 2, out_ch=3, base=base, drop=drop,
                            zero_out=True)

    def forward(self, x_A, x_B, flow_AB):
        o = self.net(torch.cat([x_A, x_B, flow_AB], 1))
        d = 0.5 * flow_AB + o[:, :2]
        a = torch.sigmoid(o[:, 2:3])
        warped_A = backward_warp(x_A, -d)
        warped_B = backward_warp(x_B, flow_AB - d)
        return {"mid": a * warped_A + (1.0 - a) * warped_B, "flow_AM": d}


class FeatResid(nn.Module):
    """Option 3 done right. Feature-space correction as a residual on the V1 floor.

    mid = physics_v1(x_A, x_B, F_AB) + Decoder(feature_warp(z_A, z_B, F_AB))

    Zero-init decoder head => residual 0 => EXACTLY physics-V1. Keeps V1's
    easy-region fidelity while letting the semantically-smooth feature path add
    corrections where pixel-warp tears (occlusion / large motion)."""
    trainable = True
    SCALE = 0.25

    def __init__(self, base: int = 48, fdim: int = 96, drop: float = 0.0):
        super().__init__()
        self.enc = _Enc(3, fdim, base, drop)
        self.dec = _Dec(fdim, 3, base, drop)
        nn.init.zeros_(self.dec.head[-1].weight)
        nn.init.zeros_(self.dec.head[-1].bias)

    def forward(self, x_A, x_B, flow_AB):
        v1 = 0.5 * backward_warp(x_A, -0.5 * flow_AB) + \
             0.5 * backward_warp(x_B, 0.5 * flow_AB)
        zA, zB = self.enc(x_A), self.enc(x_B)
        f = _flow_to_scale(flow_AB, self.SCALE)
        z_mid = 0.5 * backward_warp(zA, -0.5 * f) + 0.5 * backward_warp(zB, 0.5 * f)
        return {"mid": v1 + self.dec(z_mid), "flow_AM": None}


_REGISTRY = {
    "physics_v1": PhysicsV1,
    "flow_synth": FlowSynth,
    "flow_synth_occ": FlowSynthOcc,
    "feat_warp": FeatWarp,
    "feat_flow": FeatFlow,
    "feat_resid": FeatResid,
}


def make_arch(name: str, base_ch: int = 48, dropout: float = 0.0) -> nn.Module:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown arch {name!r}. Have: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    if cls is PhysicsV1:
        return cls()
    return cls(base=base_ch, drop=dropout)


def arch_names():
    return list(_REGISTRY)
