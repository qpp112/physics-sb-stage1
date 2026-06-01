"""
Experiment 1b — evaluation of a trained bridge.

Loads one or more (run_dir, condition_name, seed) entries and computes
PSNR / SSIM / LPIPS / EPE on the test set, stratified by motion magnitude.

If multiple seeds per condition are provided, also applies the
pilot_decision rule from metrics.stats (no overlap on PSNR, direction agreement
on the other metrics).

Usage:
  python exps/1b_eval.py --config configs/base.yaml \
      --runs runs/brownian_s0 runs/brownian_s1 runs/physics_s0 runs/physics_s1 \
      --out runs/1b_eval/results.json
"""

from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps._config import load_config
from exps._inference import build_model_from_cfg, predict_midpoint, _load_ckpt
from data import FlyingThings3D, stratify_by_motion
from bridges.reference import make_reference
from metrics.image_metrics import psnr, ssim, lpips_metric
from metrics.flow_metrics import epe_against_gt_flow
from metrics.stats import bootstrap_ci, pilot_decision


def parse_run_dir(p: Path):
    """Run dir name format: '<condition>_s<seed>'."""
    name = p.name
    if "_s" not in name:
        raise ValueError(f"bad run-dir name {name}; expected '<condition>_s<seed>'")
    cond, seed = name.rsplit("_s", 1)
    return cond, int(seed)


@torch.no_grad()
def evaluate_run(run_dir: Path, cfg: dict, ds: FlyingThings3D, device: str):
    cfg_used_path = run_dir / "config_used.yaml"
    if cfg_used_path.exists():
        run_cfg = json.loads(cfg_used_path.read_text())
    else:
        run_cfg = cfg
    ckpt = run_dir / "ckpt_final.pt"
    if not ckpt.exists():
        ckpts = sorted(run_dir.glob("ckpt_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint in {run_dir}")
        ckpt = ckpts[-1]

    model = build_model_from_cfg(run_cfg, run_cfg["data"]["size"]).to(device)
    _load_ckpt(model, str(ckpt), device)
    reference = make_reference(run_cfg["bridge"]["reference"])

    dl = DataLoader(ds, batch_size=4, num_workers=cfg["data"]["num_workers"])
    metrics = defaultdict(list)
    for batch in dl:
        xM = batch["x_M"].to(device)
        xA = batch["x_A"].to(device)
        fAB = batch["flow_AB"].to(device)
        x_hat = predict_midpoint(
            model, reference, batch,
            num_steps=run_cfg["eval"]["num_steps"],
            sigma=0.0, t_eval=0.5,
            condition_on_reference=run_cfg["bridge"].get("condition_on_reference", True),
            device=device,
        )
        metrics["psnr"].append(psnr(x_hat, xM).cpu().numpy())
        metrics["ssim"].append(ssim(x_hat, xM).cpu().numpy())
        metrics["lpips"].append(lpips_metric(x_hat, xM).cpu().numpy())
        metrics["epe"].append(epe_against_gt_flow(x_hat, xA, fAB).cpu().numpy())
    return {k: np.concatenate(v) for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories named '<condition>_s<seed>'")
    ap.add_argument("--out", default="runs/1b_eval/results.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = FlyingThings3D(
        root=cfg["data"]["root"], split="TEST",
        size=cfg["data"]["size"], min_flow=cfg["data"]["min_flow"],
        max_clips=cfg["data"]["test_clips"],
        pass_name=cfg["data"]["pass"], seed=0,
    )
    print(f"Test set: {len(ds)} triplets")
    strata = stratify_by_motion(ds, cfg["eval"]["strata_edges"])

    by_cond = defaultdict(dict)            # by_cond[condition][seed] = {metric: array}
    for rd in args.runs:
        rd = Path(rd)
        cond, seed = parse_run_dir(rd)
        print(f"\nEvaluating {rd.name}: condition={cond} seed={seed}")
        per_sample = evaluate_run(rd, cfg, ds, device)
        for k, v in per_sample.items():
            print(f"  {k:8s}  mean={v.mean():.4f}  std={v.std():.4f}")
        by_cond[cond][seed] = per_sample

    # Aggregate per condition / seed / stratum.
    summary = {}
    seed_aggregates = {"psnr": {}, "ssim": {}, "lpips": {}, "epe": {}}
    for cond, seeds in by_cond.items():
        summary[cond] = {}
        seeds_sorted = sorted(seeds.keys())
        for s in seeds_sorted:
            per = seeds[s]
            entry = {"aggregate": {}, "strata": {}}
            for k, arr in per.items():
                mean, lo, hi = bootstrap_ci(arr)
                entry["aggregate"][k] = dict(mean=mean, ci_lo=lo, ci_hi=hi,
                                              std=float(arr.std()), n=len(arr))
                seed_aggregates[k].setdefault(cond, []).append(mean)
            for b, idx in enumerate(strata):
                if not idx:
                    continue
                entry["strata"][f"stratum_{b}"] = {
                    k: {"mean": float(arr[idx].mean()),
                        "std":  float(arr[idx].std()),
                        "n":    int(len(idx))}
                    for k, arr in per.items()
                }
            summary[cond][f"seed_{s}"] = entry

    summary["__pilot_decision"] = None
    if "brownian" in by_cond and "physics" in by_cond:
        verdict = pilot_decision(
            psnr_seeds  = seed_aggregates["psnr"],
            ssim_seeds  = seed_aggregates["ssim"],
            lpips_seeds = seed_aggregates["lpips"],
            epe_seeds   = seed_aggregates["epe"],
            psnr_threshold_db=0.3,
        )
        summary["__pilot_decision"] = verdict
        print("\n=== H2-pilot ===")
        for k, v in verdict.items():
            print(f"  {k:32s}  {v}")

    # --- per-stratum physics-vs-brownian gaps ---
    if "brownian" in by_cond and "physics" in by_cond:
        edges = cfg["eval"]["strata_edges"]
        stratum_report = {}
        for b, idx in enumerate(strata):
            if not idx:
                continue
            rng = (f"[{int(edges[b])}, {int(edges[b+1])})"
                   if b < len(edges) - 2 else f"[{int(edges[b])}, inf)")
            row = {"range_px": rng, "n": len(idx)}
            for metric in ("psnr", "ssim", "lpips", "epe"):
                def _cm(cond, metric=metric, idx=idx):
                    return float(np.mean([by_cond[cond][sd][metric][idx].mean()
                                          for sd in by_cond[cond]]))
                bm, pm = _cm("brownian"), _cm("physics")
                row[metric] = {"brownian": bm, "physics": pm, "gap": pm - bm}
            stratum_report[f"stratum_{b}"] = row
        summary["__per_stratum"] = stratum_report
        print("")
        print("=== H2 per-stratum (physics - brownian) ===")
        last = f"stratum_{len(strata) - 1}"
        for key in sorted(stratum_report):
            r = stratum_report[key]
            star = "  <-- DECISIVE (large motion)" if key == last else ""
            print(f"  {r['range_px']:12s} n={r['n']:4d}  "
                  f"PSNR {r['psnr']['gap']:+.2f}dB  SSIM {r['ssim']['gap']:+.3f}  "
                  f"LPIPS {r['lpips']['gap']:+.3f}  EPE {r['epe']['gap']:+.4f}{star}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
