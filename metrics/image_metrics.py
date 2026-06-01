"""
PSNR / SSIM / LPIPS.

All metrics operate on tensors in [-1, 1]. Convert internally to [0, 1] for
the standard PSNR/SSIM formulas. LPIPS uses the AlexNet backbone (smaller and
more stable than VGG for VFI).

Reductions: each metric returns one number per batch element by default,
to enable downstream stratification and statistical aggregation.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


# ----------------------------- PSNR -----------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """pred, target: (B, C, H, W) in [-1, 1]. Returns (B,) tensor in dB."""
    p = (pred + 1.0) * 0.5
    t = (target + 1.0) * 0.5
    mse = ((p - t) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10((max_val ** 2) / mse)


# ----------------------------- SSIM -----------------------------

def _gaussian_kernel(window_size: int, sigma: float, device, dtype):
    half = window_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = g[:, None] * g[None, :]
    return k2d


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5,
         max_val: float = 1.0) -> torch.Tensor:
    """Mean SSIM per image. (B, C, H, W) in [-1, 1] -> (B,)."""
    p = (pred + 1.0) * 0.5
    t = (target + 1.0) * 0.5
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    c = p.shape[1]
    k2d = _gaussian_kernel(window_size, sigma, p.device, p.dtype)
    kernel = k2d[None, None].expand(c, 1, window_size, window_size)
    pad = window_size // 2

    def _conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=c)

    mu_p = _conv(p)
    mu_t = _conv(t)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = _conv(p * p) - mu_pp
    sigma_tt = _conv(t * t) - mu_tt
    sigma_pt = _conv(p * t) - mu_pt

    num = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    den = (mu_pp + mu_tt + c1) * (sigma_pp + sigma_tt + c2)
    return (num / den).mean(dim=(1, 2, 3))


# ----------------------------- LPIPS -----------------------------

_LPIPS_INSTANCE = None


def lpips_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Perceptual distance. (B, C, H, W) in [-1, 1] -> (B,). Lower is better.

    Lazily instantiates an LPIPS network (AlexNet backbone) on the first call,
    placed on the same device as `pred`.
    """
    global _LPIPS_INSTANCE
    import lpips                                                 # lazy import
    if _LPIPS_INSTANCE is None:
        _LPIPS_INSTANCE = lpips.LPIPS(net="alex").to(pred.device)
        _LPIPS_INSTANCE.eval()
        for p in _LPIPS_INSTANCE.parameters():
            p.requires_grad_(False)
    with torch.no_grad():
        d = _LPIPS_INSTANCE(pred, target)                        # (B, 1, 1, 1)
    return d.flatten()
