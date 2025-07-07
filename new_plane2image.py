#!/usr/bin/env python
"""
Convert optimized Tri‑Plane feature volumes into 2‑D image streams that can be fed to a
standard video codec.  Four packing strategies are supported and **two different
quantisation modes** (global vs. per‑channel), giving 8 baselines in total.

Packing strategies
------------------
1. **tiling**      – TeTriRF's original flatten‑and‑tile layout
2. **separate**    – one greyscale image per feature channel
3. **grouped**     – fixed RGB triplets (0‑2, 3‑5, 6‑8, 9‑11)
4. **correlation** – *new*: greedy correlation‑based clustering of the most
                     correlated 3‑channel groups

Quantisation modes
------------------
* **global   (default)**  – fixed numeric range [low_bound, high_bound]
* **per_channel**         – individual min/max range for each channel (frame‑wise)

Usage example
-------------
```
python triplane_packer.py \
    --logdir logs/out_triplane/flame_steak_old \
    --numframe 20 \
    --strategy correlation \
    --qmode per_channel
```
"""

import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

to8b  = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
_to16 = lambda x: ((2 ** 16 - 1) * np.clip(x, 0, 1)).astype(np.uint16)


def tile_maker(feat_plane: torch.Tensor, h: int = 2560, w: int = 2560) -> torch.Tensor:
    """Pack C feature maps of size H×W into a single canvas of size *h×w*.
    The canvas is filled row‑wise.  It is the caller's responsibility to ensure that
    the canvas is large enough.
    """
    canvas = torch.zeros(h, w, device=feat_plane.device, dtype=feat_plane.dtype)
    C, H, W = feat_plane.shape[1:]
    x, y = 0, 0
    for c in range(C):
        if y + W > canvas.shape[1]:
            y = 0
            x += H
        assert x + H <= canvas.shape[0], "tile_maker: canvas too small"
        canvas[x : x + H, y : y + W] = feat_plane[0, c]
        y += W
    return canvas


def density_quantize(density: torch.Tensor, nbits: int) -> torch.Tensor:
    """Simple hard‑clipped linear quantiser for density volumes."""
    levels = 2 ** nbits - 1
    data = density.clone()
    data.clamp_(-5, 30)
    data = (data + 5) / 35.0
    data = torch.round(data * levels) / levels
    return data


def make_density_image(density_grid: torch.Tensor, nbits: int, h: int = 2560, w: int = 4096) -> torch.Tensor:
    data = density_quantize(density_grid, nbits)
    return tile_maker(data[0], h=h, w=w)

# -----------------------------------------------------------------------------
# correlation‑based grouping
# -----------------------------------------------------------------------------

def _channel_correlation(feat: torch.Tensor) -> np.ndarray:
    """Return absolute Pearson correlation matrix of shape (C, C)."""
    C = feat.shape[1]
    flat = feat.view(C, -1).float()
    flat -= flat.mean(dim=1, keepdim=True)
    cov = flat @ flat.t()
    var = torch.diag(cov)
    std = torch.sqrt(var + 1e-8)
    corr = cov / (std[:, None] * std[None, :] + 1e-8)
    return corr.abs().cpu().numpy()


def correlation_groups(feat: torch.Tensor) -> List[List[int]]:
    """Greedy grouping of channels into triplets with the highest internal correlation."""
    C = feat.shape[1]
    assert C % 3 == 0, "correlation strategy expects C to be a multiple of 3"

    corr = _channel_correlation(feat)
    remaining = set(range(C))
    groups: List[List[int]] = []

    # Pre‑compute pair list sorted by correlation
    pairs = sorted(((corr[i, j], i, j) for i in range(C) for j in range(i + 1, C)), reverse=True)

    used = set()
    for _, i, j in pairs:
        if i in used or j in used:
            continue
        remaining.discard(i)
        remaining.discard(j)
        # pick k that maximises combined correlation with i and j
        best_k, best_val = None, -1.0
        for k in remaining:
            v = corr[i, k] + corr[j, k]
            if v > best_val:
                best_k, best_val = k, v
        if best_k is None:
            break
        groups.append([i, j, best_k])
        used.update([i, j, best_k])
        remaining.discard(best_k)
        if len(groups) == C // 3:
            break

    # Safety: assign any leftovers round‑robin
    for idx, k in enumerate(sorted(remaining)):
        groups[idx % len(groups)].append(k)

    return [g[:3] for g in groups]  # ensure size‑3 groups

# -----------------------------------------------------------------------------
# main script
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--logdir", required=True, help="Path to Tri‑plane checkpoints (.tar files)")
    p.add_argument("--numframe", type=int, default=20, help="number of frames to convert")
    p.add_argument("--startframe", type=int, default=0, help="start frame id (inclusive)")
    p.add_argument("--strategy", type=str, default="tiling",
                   choices=["tiling", "separate", "grouped", "correlation"],
                   help="packing strategy")
    p.add_argument("--qmode", type=str, default="global", choices=["global", "per_channel"],
                   help="quantisation mode")
    p.add_argument("--codec", type=str, default="h265", help="placeholder – not used here")
    return p.parse_args()


# Default global min/max used in TeTriRF
GLOBAL_LOW, GLOBAL_HIGH = -20.0, 20.0
NBITS = 2 ** 16 - 1  # 16‑bit PNG


def quantise(feat: torch.Tensor, qmode: str, bounds_out: List[Tuple[float, float]]) -> torch.Tensor:
    """Quantise *feat* (shape [1, C, H, W]) either globally or per‑channel."""
    if qmode == "global":
        low, high = GLOBAL_LOW, GLOBAL_HIGH
        bounds_out.extend([(low, high)] * feat.shape[1])
        norm = (feat - low) / (high - low)
        q = torch.round(norm * NBITS) / NBITS
        return q.clamp_(0, 1)

    # per‑channel mode
    q_ch = []
    C = feat.shape[1]
    for c in range(C):
        ch = feat[0, c]
        low, high = float(ch.min()), float(ch.max())
        if abs(high - low) < 1e-6:
            high = low + 1e-6  # avoid divide‑by‑zero
        bounds_out.append((low, high))
        norm = (ch - low) / (high - low)
        q_ch.append(torch.round(norm * NBITS) / NBITS)
    return torch.stack(q_ch, dim=0).unsqueeze(0).clamp_(0, 1)



def main():
    args = parse_args()

    logdir = args.logdir.rstrip("/")
    name = os.path.basename(logdir)
    out_root = os.path.join(logdir, f"planeimg_{args.startframe:02d}_{args.startframe + args.numframe - 1:02d}_{args.strategy}_{args.qmode}")
    os.makedirs(out_root, exist_ok=True)
    print(f"[INFO] Saving plane images to {out_root}")

    # Meta‑info to be stored per run
    meta_plane_bounds: Dict[str, List[Tuple[float, float]]] = {}

    for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
        ckpt_path = os.path.join(logdir, f"fine_last_{frameid}.tar")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # ---------------- density grid → mask ----------------
        density = ckpt["model_state_dict"]["density.grid"].clone()
        voxel_size_ratio = ckpt["model_kwargs"]["voxel_size_ratio"]

        masks = None
        if "act_shift" in ckpt["model_state_dict"]:
            alpha = 1 - (torch.exp(density + ckpt["model_state_dict"]["act_shift"]) + 1) ** (-voxel_size_ratio)
            alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)
            mask = alpha < 1e-4
            density[mask] = -5
            feature_alpha = F.interpolate(alpha, size=tuple(np.array(density.shape[-3:]) * 3), mode="trilinear", align_corners=True)
            mask_fg = feature_alpha >= 1e-4
            masks = {
                "xy": mask_fg.sum(axis=4),
                "xz": mask_fg.sum(axis=3),
                "yz": mask_fg.sum(axis=2),
            }

        # ---------------- feature planes ----------------
        planes = {k.split(".")[-1]: v.clone() for k, v in ckpt["model_state_dict"].items()
                  if "k0" in k and "plane" in k and "residual" not in k}

        for plane_name, feat in planes.items():   # feat: [1, C, H, W]
            if plane_name not in meta_plane_bounds:
                meta_plane_bounds[plane_name] = []

            if masks is not None:
                m = masks[plane_name].unsqueeze(1).repeat(1, feat.shape[1], 1, 1)
                feat = feat.masked_fill(m == 0, 0)

            bounds_this_plane: List[Tuple[float, float]] = []
            feat_q = quantise(feat, args.qmode, bounds_this_plane)
            meta_plane_bounds[plane_name].append(bounds_this_plane)  # list of frames

            C, H, W = feat_q.shape[1:]
            base_dir = os.path.join(out_root, plane_name)

            if args.strategy == "tiling":
                os.makedirs(base_dir, exist_ok=True)
                img = tile_maker(feat_q)
                cv2.imwrite(os.path.join(base_dir, f"im{frameid + 1:05d}.png"), _to16(img.cpu().numpy()))

            elif args.strategy == "separate":
                for c in range(C):
                    ch_dir = os.path.join(base_dir, f"c{c}")
                    os.makedirs(ch_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(ch_dir, f"im{frameid + 1:05d}.png"),
                                _to16(feat_q[0, c].cpu().numpy()))

            elif args.strategy == "grouped":
                for gi in range(0, C, 3):
                    stream_dir = os.path.join(base_dir, f"stream{gi // 3}")
                    os.makedirs(stream_dir, exist_ok=True)
                    triplet = feat_q[0, gi : gi + 3]
                    bgr = np.stack([triplet[2], triplet[1], triplet[0]], axis=-1)
                    cv2.imwrite(os.path.join(stream_dir, f"im{frameid + 1:05d}.png"), _to16(bgr))

            elif args.strategy == "correlation":
                for gi, idxs in enumerate(correlation_groups(feat)):
                    stream_dir = os.path.join(base_dir, f"stream{gi}")
                    os.makedirs(stream_dir, exist_ok=True)
                    triplet = feat_q[0, idxs]
                    bgr = np.stack([triplet[2], triplet[1], triplet[0]], axis=-1)
                    cv2.imwrite(os.path.join(stream_dir, f"im{frameid + 1:05d}.png"), _to16(bgr))

        # ---------------- density plane ----------------
        dens_img = make_density_image(density, 16)
        dens_dir = os.path.join(out_root, "density")
        os.makedirs(dens_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dens_dir, f"im{frameid + 1:05d}.png"), _to16(dens_img.cpu().numpy()))

    # ---------------------------------------------------------------------
    # persist meta information once after the loop
    # ---------------------------------------------------------------------
    torch.save({
        "qmode": args.qmode,
        "bounds": meta_plane_bounds,        # list‑of‑frames × list‑of‑channels per plane
        "nbits": NBITS,
    }, os.path.join(out_root, "planes_frame_meta.nf"))

    print("[DONE] Conversion finished.")


if __name__ == "__main__":
    main()
