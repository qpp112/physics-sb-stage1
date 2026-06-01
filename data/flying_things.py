"""
FlyingThings3D triplet loader (refined).

Expected layout (Sceneflow distribution):
  <root>/frames_cleanpass/{TRAIN,TEST}/{A,B,C}/<scene>/left/*.png
  <root>/optical_flow/{TRAIN,TEST}/{A,B,C}/<scene>/into_future/left/OpticalFlowIntoFuture_*_L.pfm

Triplet = (frame i, i+1, i+2); chained flow A->B = F_{i->i+1} composed with F_{i+1->i+2}.

Two performance features (added after a painful debugging session):
  * Filtering: TRAIN selects large-motion clips by SINGLE-STEP magnitude |F_AM|
    (one PFM read, fast) with early-stop; TEST stratifies by CHAINED |F_AB|
    (two reads, correct) since the motion strata define the evaluation.
  * Precompute: if a precomputed cache exists (see precompute.py), __getitem__
    reads compact in-RAM tensors instead of decoding PNG/PFM every step.
"""

from __future__ import annotations
import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .flow_utils import read_pfm, chain_flows


# ----------------------------- helpers -----------------------------

def _img_to_tensor(arr: np.ndarray, size: int) -> torch.Tensor:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = _center_crop_resize(t.unsqueeze(0), size).squeeze(0)
    return t * 2.0 - 1.0


def _flow_to_tensor(arr: np.ndarray, size: int) -> torch.Tensor:
    h, w = arr.shape[:2]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    s = min(h, w)
    scale = size / s
    t = _center_crop_resize(t, size, mode="bilinear")
    return (t * scale).squeeze(0)


def _center_crop_resize(t: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    _, _, h, w = t.shape
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    t = t[:, :, top:top + s, left:left + s]
    return F.interpolate(t, size=(size, size), mode=mode,
                         align_corners=False if mode == "bilinear" else None)


# ----------------------------- index build / cache -----------------------------

_FRAME_RE = re.compile(r"^(\d{4,7})\.png$")


def _list_frames(scene_dir: Path) -> List[str]:
    out = []
    for name in sorted(os.listdir(scene_dir)):
        m = _FRAME_RE.match(name)
        if m:
            out.append(m.group(1))
    return out


def _flow_path(flow_scene_dir: Path, frame_id: str) -> Optional[Path]:
    for c in (flow_scene_dir / f"OpticalFlowIntoFuture_{frame_id}_L.pfm",
              flow_scene_dir / f"{frame_id}.pfm",
              flow_scene_dir / f"{frame_id}.flo"):
        if c.exists():
            return c
    return None


def _scan_split(frame_root: Path, flow_root: Path) -> List[Tuple[str, str, str, str, str]]:
    triplets: List[Tuple[str, str, str, str, str]] = []
    if not frame_root.exists():
        raise FileNotFoundError(f"Frame root not found: {frame_root}")
    for letter in sorted(os.listdir(frame_root)):
        ldir = frame_root / letter
        if not ldir.is_dir():
            continue
        for scene in sorted(os.listdir(ldir)):
            frame_dir = ldir / scene / "left"
            flow_scene = flow_root / letter / scene / "into_future" / "left"
            if not (frame_dir.is_dir() and flow_scene.is_dir()):
                continue
            ids = _list_frames(frame_dir)
            for i in range(len(ids) - 2):
                idA, idM, idB = ids[i], ids[i + 1], ids[i + 2]
                fl_AM = _flow_path(flow_scene, idA)
                fl_MB = _flow_path(flow_scene, idM)
                if fl_AM is None or fl_MB is None:
                    continue
                triplets.append((
                    str(frame_dir / f"{idA}.png"),
                    str(frame_dir / f"{idM}.png"),
                    str(frame_dir / f"{idB}.png"),
                    str(fl_AM), str(fl_MB),
                ))
    return triplets


def _max_flow_for(tup, size, mode):
    """Return chained-vs-single-step max flow magnitude (target-res px), or None on failure.

    mode='single_step': read F_AM only (fast).  mode='chained': read both + chain (correct).
    """
    if mode == "single_step":
        try:
            fl = read_pfm(tup[3])
        except Exception:
            return None
        u = np.nan_to_num(fl[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(fl[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
        h, w = fl.shape[:2]
        return float(np.sqrt(u * u + v * v).max()) * (size / min(h, w))
    else:  # chained
        try:
            fl_AM = read_pfm(tup[3]); fl_MB = read_pfm(tup[4])
        except Exception:
            return None
        t_AM = _flow_to_tensor(fl_AM, size).unsqueeze(0)
        t_MB = _flow_to_tensor(fl_MB, size).unsqueeze(0)
        try:
            t_AB = chain_flows(t_AM, t_MB)
        except Exception:
            return None
        return float(t_AB.norm(dim=1).amax().item())


def _filter_triplets(triplets, min_flow, size, mode, target_keep=None, desc="indexing"):
    """Keep triplets whose max flow magnitude (per `mode`) >= min_flow.
    Stops once target_keep kept (triplets should be pre-shuffled). Progress bar."""
    from tqdm import tqdm
    kept = []
    pbar = tqdm(triplets, desc=desc, unit="trip", dynamic_ncols=True)
    for tup in pbar:
        if target_keep is not None and len(kept) >= target_keep:
            break
        mag = _max_flow_for(tup, size, mode)
        if mag is not None and mag >= min_flow:
            kept.append({"paths": tup, "max_flow": mag})
            pbar.set_postfix(kept=len(kept))
    pbar.close()
    return kept


def _cache_key(root: Path, split: str, min_flow: float, size: int) -> str:
    return hashlib.md5(f"{root}|{split}|{min_flow}|{size}".encode()).hexdigest()[:12]


# ----------------------------- main dataset -----------------------------

class FlyingThings3D(Dataset):
    def __init__(self, root: str, split: str = "TRAIN", size: int = 256,
                 min_flow: float = 40.0, max_clips: Optional[int] = None,
                 pass_name: str = "cleanpass", cache_dir: Optional[str] = None,
                 seed: int = 0, return_paths: bool = False,
                 filter_mode: Optional[str] = None,
                 return_half_flows: bool = False):
        self.root = Path(root)
        self.split = split
        self.size = size
        self.min_flow = min_flow
        self.return_paths = return_paths
        # Stage 2: half-flows (F_{A->M}, F_{M->B}) are needed for the
        # acceleration / occlusion references. The precompute cache only stores
        # the chained F_{A->B}, so when half-flows are requested we bypass it
        # and read raw PFM per clip (fine for small static-eval sets).
        self.return_half_flows = return_half_flows

        # TRAIN: single-step selection (fast). TEST: chained (correct stratification).
        if filter_mode is None:
            filter_mode = "single_step" if split == "TRAIN" else "chained"

        frame_root = self.root / f"frames_{pass_name}" / split
        flow_root = self.root / "optical_flow" / split

        if not self.root.exists():
            raise FileNotFoundError(
                f"FlyingThings3D root not found: {self.root}\n"
                f"  Edit configs/base.yaml -> data.root.")
        if not frame_root.exists():
            raise FileNotFoundError(
                f"Frames dir not found: {frame_root}\n"
                f"  Expected <root>/frames_{pass_name}/{{TRAIN,TEST}}/<letter>/<scene>/left/*.png")
        if not flow_root.exists():
            raise FileNotFoundError(
                f"Optical flow dir not found: {flow_root}\n"
                f"  Did you download/extract the optical_flow archive?")

        cache_dir = Path(cache_dir) if cache_dir else (self.root / ".pilot_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(self.root, split, min_flow, size)
        cache_file = cache_dir / f"index_{split}_{key}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                kept = json.load(f)
        else:
            raw = _scan_split(frame_root, flow_root)
            if filter_mode == "single_step":
                np.random.default_rng(0).shuffle(raw)            # random subset for early-stop
                tk = (max_clips + max_clips // 2) if max_clips else None
            else:
                tk = None                                         # examine all for correct strata
            kept = _filter_triplets(raw, min_flow, size, filter_mode,
                                    target_keep=tk, desc=f"indexing {split}")
            with open(cache_file, "w") as f:
                json.dump(kept, f)

        for _r, _it in enumerate(kept):
            _it["row"] = _r
        rng = np.random.default_rng(seed)
        kept = [kept[i] for i in rng.permutation(len(kept))]
        if max_clips is not None:
            kept = kept[:max_clips]
        self.items = kept

        # precomputed cache (built by precompute.py)
        self._precomp = None
        pc = cache_dir / f"precomp_{split}_{key}"
        if (pc / "DONE").exists() and not self.return_half_flows:
            self._precomp = dict(
                frames=np.load(pc / "frames.npy", mmap_mode="r"),
                flow=np.load(pc / "flow.npy", mmap_mode="r"),
                valid=np.load(pc / "valid.npy"),
            )
            self.items = [it for it in self.items if self._precomp["valid"][it["row"]]]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        if self._precomp is not None:
            r = item["row"]
            fr = self._precomp["frames"][r]
            fl = self._precomp["flow"][r]

            def _ti(arr):
                return torch.from_numpy(np.array(arr)).float().div(255.0).mul(2.0).sub(1.0)

            return dict(
                x_A=_ti(fr[0]), x_M=_ti(fr[1]), x_B=_ti(fr[2]),
                flow_AB=torch.from_numpy(np.array(fl)).float(),
                max_flow=torch.tensor(float(item["max_flow"]), dtype=torch.float32),
            )

        fA, fM, fB, fl_AM, fl_MB = item["paths"]
        import imageio.v2 as imageio
        xA = _img_to_tensor(imageio.imread(fA), self.size)
        xM = _img_to_tensor(imageio.imread(fM), self.size)
        xB = _img_to_tensor(imageio.imread(fB), self.size)
        f_AM = _flow_to_tensor(read_pfm(fl_AM), self.size)
        f_MB = _flow_to_tensor(read_pfm(fl_MB), self.size)
        flow_AB = chain_flows(f_AM.unsqueeze(0), f_MB.unsqueeze(0)).squeeze(0)
        sample = dict(x_A=xA, x_M=xM, x_B=xB, flow_AB=flow_AB,
                      max_flow=torch.tensor(item["max_flow"], dtype=torch.float32))
        if self.return_half_flows:
            sample["flow_AM"] = f_AM
            sample["flow_MB"] = f_MB
        if self.return_paths:
            sample["paths"] = (fA, fM, fB)
        return sample


def stratify_by_motion(test_set: "FlyingThings3D", edges):
    edges = [float(e) for e in edges]
    bins = [[] for _ in range(len(edges) - 1)]
    for i, item in enumerate(test_set.items):
        m = item["max_flow"]
        for b in range(len(edges) - 1):
            if edges[b] <= m < edges[b + 1]:
                bins[b].append(i)
                break
    return bins
