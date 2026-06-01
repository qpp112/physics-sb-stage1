"""
Optical flow utilities for FlyingThings3D.

The flow files ship in PFM format, single-step (frame i -> frame i+1).
For a triplet (A=i, M=i+1, B=i+2) we need flow A->B, obtained by chaining
F_{A->M} with F_{M->B}.
"""

from __future__ import annotations
import re
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------- PFM reader -----------------------------

def read_pfm(path: str) -> np.ndarray:
    """Read a PFM file. FlyingThings3D ships .pfm flow as 3-channel
    (u, v, mask-or-unused). We return (H, W, 2) -- u, v only."""
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")

        # tolerate optional comment lines
        while True:
            line = f.readline().decode("latin-1").strip()
            if line and not line.startswith("#"):
                break
        m = re.match(r"^(\d+)\s+(\d+)\s*$", line)
        if not m:
            raise ValueError(f"Malformed PFM dimensions: {line!r}")
        w, h = int(m.group(1)), int(m.group(2))

        scale_line = f.readline().decode("latin-1").strip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        ch = 3 if color else 1
        data = np.frombuffer(f.read(), dtype=endian + "f").reshape(h, w, ch)
        data = np.flipud(data).copy()                # PFM origin is bottom-left
        if scale != 1.0:
            data *= scale

    if color:
        data = data[..., :2]                          # drop padding channel
    return data.astype(np.float32)


def read_flo(path: str) -> np.ndarray:
    """Read a Middlebury .flo file -> (H, W, 2)."""
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, 1)[0]
        if abs(magic - 202021.25) > 1e-3:
            raise ValueError(f"Not a .flo file: {path}")
        w = int(np.fromfile(f, np.int32, 1)[0])
        h = int(np.fromfile(f, np.int32, 1)[0])
        data = np.fromfile(f, np.float32, 2 * h * w).reshape(h, w, 2)
    return data.astype(np.float32)


# ----------------------------- warping -----------------------------

def _build_base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Return identity sampling grid in pixel coords, shape (1, H, W, 2),
    last dim = (x, y) in absolute pixel coordinates."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).unsqueeze(0)


def backward_warp(img: torch.Tensor, flow: torch.Tensor,
                  padding_mode: str = "border") -> torch.Tensor:
    """Sample `img` at locations `p + flow(p)`. Use this when `flow` is the
    motion vector from the TARGET frame back to the source frame -- i.e.,
    flow_{target->source}. This is the standard "warp via grid_sample" idiom.

    img:  (B, C, H, W)
    flow: (B, 2, H, W)  -- channel 0 = dx, channel 1 = dy, pixel units
    """
    b, _, h, w = img.shape
    base = _build_base_grid(h, w, img.device, img.dtype).expand(b, -1, -1, -1)
    target = base + flow.permute(0, 2, 3, 1)
    # normalize to [-1, 1]
    norm_x = 2.0 * target[..., 0] / max(w - 1, 1) - 1.0
    norm_y = 2.0 * target[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


def forward_splat(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Naive forward-splat: for each source pixel, splat its value to the
    target location using a 2x2 bilinear footprint. Z-buffering by accumulation;
    final pixel = sum(value * weight) / sum(weight). Holes filled with 0.

    For Stage 1 we mostly use bidirectional backward-warping (flow A->B AND
    flow B->A), but this is here for sanity checks and visualizations.

    img:  (B, C, H, W)
    flow: (B, 2, H, W)
    """
    b, c, h, w = img.shape
    device = img.device
    dtype = img.dtype

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    tx = xs.unsqueeze(0) + flow[:, 0]               # (B, H, W)
    ty = ys.unsqueeze(0) + flow[:, 1]
    out = torch.zeros_like(img)
    wgt = torch.zeros(b, 1, h, w, device=device, dtype=dtype)

    x0 = torch.floor(tx).long(); x1 = x0 + 1
    y0 = torch.floor(ty).long(); y1 = y0 + 1
    wx1 = (tx - x0.float()).unsqueeze(1)
    wy1 = (ty - y0.float()).unsqueeze(1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    for x_idx, y_idx, ww in (
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    ):
        valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        ww = ww * valid.unsqueeze(1).float()
        # clamp and scatter
        x_idx = x_idx.clamp(0, w - 1)
        y_idx = y_idx.clamp(0, h - 1)
        lin = (y_idx * w + x_idx).unsqueeze(1).expand(-1, c, -1, -1)   # (B,C,H,W)
        out_flat = out.view(b, c, -1)
        out_flat.scatter_add_(2, lin.reshape(b, c, -1),
                              (img * ww).reshape(b, c, -1))
        wgt_flat = wgt.view(b, 1, -1)
        wgt_flat.scatter_add_(2, lin[:, :1].reshape(b, 1, -1),
                              ww.reshape(b, 1, -1))

    out = out / wgt.clamp_min(1e-6)
    return out


# ----------------------------- flow chaining -----------------------------

def chain_flows(flow_AM: torch.Tensor, flow_MB: torch.Tensor) -> torch.Tensor:
    """Chain F_{A->M} with F_{M->B} to produce F_{A->B}.

    F_{A->B}(p) = F_{A->M}(p) + F_{M->B}(p + F_{A->M}(p))

    The second term requires sampling F_{M->B} at the landing point in M,
    so we use bilinear sampling via grid_sample.

    flow_AM, flow_MB: (B, 2, H, W) in pixel units
    """
    b, _, h, w = flow_AM.shape
    device = flow_AM.device
    dtype = flow_AM.dtype

    base = _build_base_grid(h, w, device, dtype).expand(b, -1, -1, -1)
    landing = base + flow_AM.permute(0, 2, 3, 1)            # (B, H, W, 2)
    norm_x = 2.0 * landing[..., 0] / max(w - 1, 1) - 1.0
    norm_y = 2.0 * landing[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)
    sampled = F.grid_sample(flow_MB, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    return flow_AM + sampled


# ----------------------------- sanity check 1 -----------------------------

def warp_consistency_error(x_A: torch.Tensor, x_B: torch.Tensor,
                            flow_AB: torch.Tensor) -> torch.Tensor:
    """Sanity check: warp x_A forward using flow_AB and compare to x_B.

    Implementation note: we use BACKWARD warping with -flow_AB, which approximates
    forward warping for the purposes of this check (it gives the right answer on
    non-occluded, non-disocclusion pixels, which is what the check needs).

    Returns per-image mean L1 error in [0, 1] range, shape (B,).
    """
    # bring inputs into [0, 1]
    x_A01 = (x_A + 1.0) * 0.5
    x_B01 = (x_B + 1.0) * 0.5
    # backward-warp x_B using flow_AB so that warp(x_B)[p] = x_B[p + F_AB(p)] ≈ x_A[p]
    # then compare warp(x_B) vs x_A on non-occluded pixels
    warped = backward_warp(x_B01, flow_AB, padding_mode="border")
    err = (warped - x_A01).abs().mean(dim=(1, 2, 3))
    return err
