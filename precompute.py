"""
Precompute the selected clips into compact tensors so training is GPU-bound
instead of starved by PNG/PFM decoding from slow storage.

For each split it reads the cached triplet index, decodes + resizes each clip
ONCE, and writes:
  <root>/.pilot_cache/precomp_<split>_<key>/frames.npy   (N, 3, 3, size, size) uint8
  <root>/.pilot_cache/precomp_<split>_<key>/flow.npy     (N, 2, size, size)     float16
  <root>/.pilot_cache/precomp_<split>_<key>/valid.npy    (N,)                   bool
  <root>/.pilot_cache/precomp_<split>_<key>/DONE         (flag)

The dataset auto-detects this cache (see flying_things.py) and serves from it.
Working set shrinks ~10x (e.g. 40 GB -> ~4 GB), so it fits in RAM and the GPU
stops waiting. Bad/corrupt files are skipped here (valid=False) rather than
crashing a training run.

Run once per machine:  python3 precompute.py --config configs/base.yaml
Idempotent: re-running skips splits that already have a DONE flag.
"""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm

sys.path.insert(0, ".")
from exps._config import load_config
from data.flying_things import (FlyingThings3D, _cache_key,
                                _img_to_tensor, _flow_to_tensor)
from data.flow_utils import read_pfm, chain_flows


def precompute_split(cfg, split):
    root = Path(cfg["data"]["root"])
    size = cfg["data"]["size"]
    min_flow = cfg["data"]["min_flow"]
    key = _cache_key(root, split, min_flow, size)
    cache_dir = root / ".pilot_cache"
    idx_file = cache_dir / f"index_{split}_{key}.json"
    if not idx_file.exists():
        print(f"  {split}: no index yet (build it first)"); return
    items = json.load(open(idx_file))
    out = cache_dir / f"precomp_{split}_{key}"; out.mkdir(exist_ok=True)
    if (out / "DONE").exists():
        print(f"  {split}: already precomputed ({len(items)} clips)"); return

    N = len(items)
    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+",
                                       dtype=np.uint8, shape=(N, 3, 3, size, size))
    flow = np.lib.format.open_memmap(out / "flow.npy", mode="w+",
                                     dtype=np.float16, shape=(N, 2, size, size))
    maxf = np.zeros(N, np.float32)
    valid = np.zeros(N, bool)

    for i, it in enumerate(tqdm(items, desc=f"precompute {split}", unit="clip")):
        fA, fM, fB, flAM, flMB = it["paths"]
        try:
            xs = [_img_to_tensor(imageio.imread(f), size) for f in (fA, fM, fB)]
            fAM = _flow_to_tensor(read_pfm(flAM), size)
            fMB = _flow_to_tensor(read_pfm(flMB), size)
            fab = chain_flows(fAM.unsqueeze(0), fMB.unsqueeze(0)).squeeze(0)
        except Exception:
            continue                                 # leave valid[i]=False, skip
        for j, x in enumerate(xs):
            frames[i, j] = ((x.clamp(-1, 1) + 1) * 0.5 * 255).round().byte().numpy()
        flow[i] = fab.numpy().astype(np.float16)
        maxf[i] = it["max_flow"]
        valid[i] = True

    frames.flush(); flow.flush()
    np.save(out / "max_flow.npy", maxf)
    np.save(out / "valid.npy", valid)
    (out / "DONE").write_text(f"{int(valid.sum())}/{N} valid")
    print(f"  {split}: {int(valid.sum())}/{N} valid clips cached -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # ensure indices exist (loads cached ones, or builds them)
    FlyingThings3D(root=cfg["data"]["root"], split="TRAIN", size=cfg["data"]["size"],
                   min_flow=cfg["data"]["min_flow"], max_clips=cfg["data"]["train_clips"],
                   pass_name=cfg["data"]["pass"], seed=0)
    FlyingThings3D(root=cfg["data"]["root"], split="TEST", size=cfg["data"]["size"],
                   min_flow=cfg["data"]["min_flow"], max_clips=cfg["data"]["test_clips"],
                   pass_name=cfg["data"]["pass"], seed=0)

    for split in ("TRAIN", "TEST"):
        precompute_split(cfg, split)
    print("\nPrecompute complete. Training will now read from the compact cache.")


if __name__ == "__main__":
    main()
