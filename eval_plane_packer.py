#!/usr/bin/env python3
import os
import argparse
from typing import List, Tuple, Dict
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image

# ***** EXACT same utilities as training *****
from src.models.model_utils import (
    DCVC_ALIGN,
    pack_planes_to_rgb,
    pack_density_to_rgb,
)

"""
python eval_plane_packer.py     --logdir logs/nerf_synthetic/lego_image \
            --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global
python eval_plane_packer.py     --logdir logs/dynerf_flame_steak/av1_qp20 \
            --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global
"""

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _to16(x: np.ndarray) -> np.ndarray:
    """float01 -> uint16"""
    return (np.clip(x, 0, 1) * (2**16 - 1) + 0.5).astype(np.uint16)

def save_png16_rgb(path: str, arr_u16_rgb: np.ndarray) -> None:
    """
    Save 16-bit PNG with correct on-disk RGB order.
    We must pass BGR to cv2.imwrite (OpenCV’s convention).
    """
    if arr_u16_rgb.dtype != np.uint16 or arr_u16_rgb.ndim != 3 or arr_u16_rgb.shape[2] != 3:
        raise ValueError(f"save_png16_rgb expects HxWx3 uint16, got {arr_u16_rgb.shape} {arr_u16_rgb.dtype}")
    # CRITICAL: swap RGB -> BGR for OpenCV
    arr_u16_bgr = arr_u16_rgb[..., ::-1].copy()
    if not cv2.imwrite(path, arr_u16_bgr):
        raise RuntimeError(f"Failed to write PNG: {path}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--logdir", required=True, help="Path to Tri-plane checkpoints (.tar files)")
    p.add_argument("--numframe", type=int, default=20, help="number of frames to convert")
    p.add_argument("--startframe", type=int, default=0, help="start frame id (inclusive)")
    p.add_argument("--qmode", type=str, default="global", choices=["global", "per_channel"],
                   help="quantisation mode")
    p.add_argument("--orient", choices=["yx", "yz", "xz"], default="xz",
                   help="(kept for compatibility; not used by packer)")
    p.add_argument("--plane_packing_mode", choices=["flatten", "mosaic", "flat4"], default="flatten",
                   help="Packing mode for feature planes (xy/xz/yz)")
    p.add_argument("--grid_packing_mode", choices=["flatten", "mosaic", "flat4"], default="flatten",
                   help="Packing mode for density grid")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Quantisation (unchanged)
# -----------------------------------------------------------------------------

GLOBAL_LOW, GLOBAL_HIGH = -20.0, 20.0
NBITS = 2 ** 16 - 1  # 16-bit PNG
PLANE_BOUNDS: Dict[str, List[Tuple[float, float]]] = {}  # for per-channel qmode pre-scan

def quantise(feat: torch.Tensor, qmode: str, bounds_out: List[Tuple[float, float]], plane_name=None) -> torch.Tensor:
    """Quantise *feat* (shape [1, C, H, W]) to [0,1] and round to 16-bit grid."""
    if qmode == "global":
        low, high = GLOBAL_LOW, GLOBAL_HIGH
        bounds_out.extend([(low, high)] * feat.shape[1])
        norm = (feat - low) / (high - low)
        return torch.round(norm.clamp(0, 1) * NBITS) / NBITS

    # per-channel
    C = feat.shape[1]
    q_ch = []
    for c in range(C):
        low, high = PLANE_BOUNDS[plane_name][c]
        ch        = feat[0, c]
        bounds_out.append((low, high))
        norm = (ch.clamp(low, high) - low) / (high - low + 1e-8)
        q_ch.append(torch.round(norm * NBITS) / NBITS)
    return torch.stack(q_ch, dim=0).unsqueeze(0).clamp_(0, 1)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    logdir = args.logdir.rstrip("/")

    # ---------- per-channel bounds pre-scan ----------
    if args.qmode == "per_channel":
        print("[INFO] Pre-scanning frames to compute per-channel bounds…")
        ch_min_max_per_plane: Dict[str, Tuple[List[float], List[float]]] = {}

        for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
            ckpt_path = os.path.join(logdir, f"fine_last_{frameid}.tar")
            ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            density = ckpt["model_state_dict"]["density.grid"].clone()
            voxel_size_ratio = ckpt["model_kwargs"]["voxel_size_ratio"]

            masks = None
            if "act_shift" in ckpt["model_state_dict"]:
                alpha = 1 - (torch.exp(density + ckpt["model_state_dict"]["act_shift"]) + 1) ** (-voxel_size_ratio)
                alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)
                feature_alpha = F.interpolate(alpha, size=tuple(np.array(density.shape[-3:]) * 3),
                                              mode="trilinear", align_corners=True)
                mask_fg = feature_alpha >= 1e-4
                masks = {"xy": mask_fg.sum(axis=4),
                         "xz": mask_fg.sum(axis=3),
                         "yz": mask_fg.sum(axis=2)}
                masks['xy_plane'] = masks['xy']
                masks['xz_plane'] = masks['xz']
                masks['yz_plane'] = masks['yz']

            planes_scan = {k.split(".")[-1]: v.clone()
                           for k, v in ckpt["model_state_dict"].items()
                           if "k0" in k and "plane" in k and "residual" not in k}

            for plane, feat_scan in planes_scan.items():
                if masks is not None:
                    m = masks[plane].unsqueeze(1).repeat(1, feat_scan.shape[1], 1, 1)
                    feat_scan = feat_scan.masked_fill(m == 0, 0)

                C_scan = feat_scan.shape[1]
                if plane not in ch_min_max_per_plane:
                    ch_min_max_per_plane[plane] = ([float("inf")] * C_scan, [float("-inf")] * C_scan)

                ch_min, ch_max = ch_min_max_per_plane[plane]
                for c in range(C_scan):
                    ch         = feat_scan[0, c]
                    ch_min[c]  = min(ch_min[c],  ch.min().item())
                    ch_max[c]  = max(ch_max[c],  ch.max().item())

        for plane, (mn, mx) in ch_min_max_per_plane.items():
            PLANE_BOUNDS[plane] = list(zip(mn, mx))

        print("[INFO] Per-channel bounds (segment-wide, per plane):")
        for p, lst in PLANE_BOUNDS.items():
            for idx, (lo, hi) in enumerate(lst):
                print(f"  {p:>2s}-chan{idx:02d}: [{lo:.4f}, {hi:.4f}]")

    # --- new folder pattern with two modes ---
    out_root = os.path.join(
        logdir,
        f"planeimg_{args.startframe:02d}_{args.startframe + args.numframe - 1:02d}_{args.plane_packing_mode}_{args.grid_packing_mode}_{args.qmode}"
    )
    os.makedirs(out_root, exist_ok=True)
    print(f"[INFO] Saving plane images to {out_root}")

    meta_plane_bounds: Dict[str, List[List[Tuple[float, float]]]] = {}
    orig_sizes_map: Dict[str, List[Tuple[int,int]]] = {}
    plane_size: Dict[str, Tuple[int, int, int, int]] = {}

    for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
        ckpt_path = os.path.join(logdir, f"fine_last_{frameid}.tar")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # ---- optional mask via density (unchanged) ----
        density = ckpt["model_state_dict"]["density.grid"].clone()  # [1,1,Dy,Dx,Dz]
        voxel_size_ratio = ckpt["model_kwargs"]["voxel_size_ratio"]
        masks = None
        # if "act_shift" in ckpt["model_state_dict"]:
        #     alpha = 1 - (torch.exp(density + ckpt["model_state_dict"]["act_shift"]) + 1) ** (-voxel_size_ratio)
        #     alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)
        #     mask = alpha < 1e-4
        #     density[mask] = -5
        #     feature_alpha = F.interpolate(alpha, size=tuple(np.array(density.shape[-3:]) * 3),
        #                                   mode="trilinear", align_corners=True)
        #     mask_fg = feature_alpha >= 1e-4
        #     masks = {
        #         "xy": mask_fg.sum(axis=4),
        #         "xz": mask_fg.sum(axis=3),
        #         "yz": mask_fg.sum(axis=2),
        #     }
        #     masks['xy_plane'] = masks['xy']
        #     masks['xz_plane'] = masks['xz']
        #     masks['yz_plane'] = masks['yz']

        # ---- feature planes ----
        planes = {k.split(".")[-1]: v.clone() for k, v in ckpt["model_state_dict"].items()
                  if "k0" in k and "plane" in k and "residual" not in k}

        for plane_name, feat in planes.items():   # feat: [1, C, H, W]
            if plane_name not in meta_plane_bounds:
                meta_plane_bounds[plane_name] = []
                orig_sizes_map[plane_name] = []
                plane_size[plane_name] = tuple(feat.shape)

            if masks is not None:
                m = masks[plane_name].unsqueeze(1).repeat(1, feat.shape[1], 1, 1)
                feat = feat.masked_fill(m == 0, 0)

            # quantise -> [0,1]
            bounds_this_plane: List[Tuple[float, float]] = []
            feat_q = quantise(feat, args.qmode, bounds_this_plane, plane_name=plane_name)
            meta_plane_bounds[plane_name].append(bounds_this_plane)

            # pack (training-consistent) with plane mode
            y_pad, orig_hw = pack_planes_to_rgb(feat_q, align=DCVC_ALIGN, mode=args.plane_packing_mode)
            orig_sizes_map[plane_name].append(tuple(orig_hw))

            # save 16-bit PNG (RGB order on disk)
            base_dir = os.path.join(out_root, plane_name)
            os.makedirs(base_dir, exist_ok=True)
            y_u16 = _to16(y_pad[0].permute(1,2,0).cpu().numpy())  # HxWx3 uint16, RGB
            save_png16_rgb(os.path.join(base_dir, f"im{frameid + 1:05d}.png"), y_u16)

        # ---- density canvas (uses its OWN mode) ----
        dens_y_pad, dens_orig = pack_density_to_rgb(density, align=DCVC_ALIGN, mode=args.grid_packing_mode)
        dens_u16 = _to16(dens_y_pad[0].permute(1,2,0).cpu().numpy())  # HxWx3 uint16, RGB
        dens_dir = os.path.join(out_root, "density")
        os.makedirs(dens_dir, exist_ok=True)
        save_png16_rgb(os.path.join(dens_dir, f"im{frameid + 1:05d}.png"), dens_u16)

        # store original size used before padding (for exact crop on unpack)
        if "density_orig" not in plane_size:
            plane_size["density_orig"] = []
        plane_size["density_orig"].append(tuple(dens_orig))

    # ---- persist meta ----
    torch.save({
        "qmode": args.qmode,
        "bounds": meta_plane_bounds,
        "nbits": NBITS,
        "plane_size": plane_size,         # includes per-plane shape + "density_orig" list
        "orig_sizes": orig_sizes_map,     # per-plane list[(H2,W2)] for feature planes
        "plane_packing_mode": args.plane_packing_mode,
        "grid_packing_mode":  args.grid_packing_mode,
    }, os.path.join(out_root, "planes_frame_meta.nf"))

    print("[DONE] Conversion finished.")

if __name__ == "__main__":
    main()