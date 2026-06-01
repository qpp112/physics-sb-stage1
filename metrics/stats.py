"""
Statistical aggregation for the pilot.

Pilot uses 2 seeds per condition, so we don't claim formal significance:
H2-pilot decision rule is "ranges across the 2 seeds do not overlap on PSNR
and direction agrees on the other metrics" (see prereg.md).

The full plan uses paired Wilcoxon with Holm correction across 3 seeds; we
include that path here too so the same code runs both phases.
"""

from __future__ import annotations
from typing import Sequence, Tuple, Dict
import numpy as np


def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Mean and (lo, hi) percentile bootstrap CI."""
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def ranges_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """Do the [min, max] ranges of a and b overlap?"""
    a = np.asarray(list(a))
    b = np.asarray(list(b))
    return not (a.max() < b.min() or b.max() < a.min())


def paired_wilcoxon(diff: Sequence[float]) -> Dict[str, float]:
    """Two-sided paired Wilcoxon signed-rank test on the seed-level differences.
    Used in the full plan, not the pilot."""
    from scipy.stats import wilcoxon
    d = np.asarray(list(diff))
    if np.all(d == 0) or len(d) < 3:
        return {"stat": float("nan"), "p": float("nan"), "n": len(d)}
    stat, p = wilcoxon(d)
    return {"stat": float(stat), "p": float(p), "n": len(d)}


def holm(pvals: Sequence[float], alpha: float = 0.05) -> Sequence[bool]:
    """Holm-Bonferroni step-down. Returns (reject_i) bool list in input order."""
    pvals = list(pvals)
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    reject = [False] * n
    for k, i in enumerate(order):
        if pvals[i] <= alpha / (n - k):
            reject[i] = True
        else:
            break
    return reject


def pilot_decision(psnr_seeds: Dict[str, Sequence[float]],
                   ssim_seeds: Dict[str, Sequence[float]],
                   lpips_seeds: Dict[str, Sequence[float]],
                   epe_seeds: Dict[str, Sequence[float]],
                   psnr_threshold_db: float = 0.3) -> Dict[str, object]:
    """Implements the H2-pilot rule from prereg.md.

    Inputs are dicts keyed by condition name ('brownian', 'physics') -> list of
    per-seed metric means (aggregate across test set).

    Returns a dict reporting all metric deltas, pass/fail per criterion, and
    an overall verdict.
    """
    out = {}
    b_psnr = list(psnr_seeds["brownian"]); p_psnr = list(psnr_seeds["physics"])
    delta_psnr = np.mean(p_psnr) - np.mean(b_psnr)
    no_overlap = (min(p_psnr) > max(b_psnr))
    psnr_pass = (delta_psnr >= psnr_threshold_db) and no_overlap

    def _delta(d):
        return float(np.mean(d["physics"]) - np.mean(d["brownian"]))
    d_ssim  = _delta(ssim_seeds)        # higher is better
    d_lpips = _delta(lpips_seeds)       # lower is better
    d_epe   = _delta(epe_seeds)         # lower is better
    ssim_dir_ok  = d_ssim  >= 0.0
    lpips_dir_ok = d_lpips <= 0.0
    epe_dir_ok   = d_epe   <= 0.0

    verdict = "PASS" if (psnr_pass and ssim_dir_ok and lpips_dir_ok and epe_dir_ok) else "FAIL"
    out.update(
        delta_psnr_db=float(delta_psnr),
        psnr_threshold_db=psnr_threshold_db,
        seed_ranges_disjoint=bool(no_overlap),
        delta_ssim=d_ssim, delta_lpips=d_lpips, delta_epe=d_epe,
        ssim_dir_ok=ssim_dir_ok, lpips_dir_ok=lpips_dir_ok, epe_dir_ok=epe_dir_ok,
        verdict=verdict,
    )
    return out
