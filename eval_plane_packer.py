#!/usr/bin/env python3
import os
import argparse
from typing import List, Tuple, Dict
import copy

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# ***** Use the SAME utilities as training *****
from src.models.model_utils import (
    DCVC_ALIGN,
    pack_planes_to_rgb, unpack_rgb_to_planes,  # (unpack only used for sanity if needed)
    dens_to01, dens_from01,
    tile_1xCHW, untile_to_1xCHW,
    pad_to_align, crop_from_align,
)

"""
python eval_plane_packer.py \
    --logdir logs/dynerf_sear_steak/dcvc_qp60_gradbpp \
        --numframe 1 --packing_mode flatten  --qmode global
"""

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

_to16 = lambda x: ((2 ** 16 - 1) * np.clip(x, 0, 1)).astype(np.uint16)

def make_density_rgb_image(density_grid: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """
    Training-consistent density packing:
      [1,1,Dy,Dx,Dz] --dens_to01--> view as [1,C=Dy,H=Dx,W=Dz]
      --tile_1xCHW--> mono canvas [Hc,Wc] --repeat 3ch--> [1,3,Hc,Wc]
      --pad_to_align--> [1,3,Hp,Wp]
    Returns:
      y_pad : [1,3,Hp,Wp] float in [0,1]
      orig  : (Hc, Wc)    (for cropping back before un-tiling)
    """
    assert density_grid.dim() == 5 and density_grid.shape[:2] == (1,1)
    _, _, Dy, Dx, Dz = density_grid.shape
    d01 = dens_to01(density_grid)                                 # [1,1,Dy,Dx,Dz]
    d01_chw = d01.view(1, Dy, Dx, Dz)                             # [1,C=Dy,H=Dx,W=Dz]
    mono, (Hc, Wc) = tile_1xCHW(d01_chw)                          # [Hc,Wc]
    y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)            # [1,3,Hc,Wc]
    y_pad, orig = pad_to_align(y, align=DCVC_ALIGN)               # [1,3,Hp,Wp], (Hc,Wc)
    return y_pad, orig


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
    p.add_argument("--packing_mode", choices=["flatten", "mosaic"], default="flatten",
                   help="Feature packing mode (match training cfg.dcvc.packing_mode)")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Quantisation (kept as in your original script)
# -----------------------------------------------------------------------------

GLOBAL_LOW, GLOBAL_HIGH = -20.0, 20.0
NBITS = 2 ** 16 - 1  # 16-bit PNG
PLANE_BOUNDS: Dict[str, List[Tuple[float, float]]] = {}  # used only during pre-scan

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
        low, high = PLANE_BOUNDS[plane_name][c]         # segment-wide bounds per plane/channel
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

    # ---------------------------------------------------------------------
    # Pre-scan: collect per-channel bounds across the segment (per plane)
    # ---------------------------------------------------------------------
    if args.qmode == "per_channel":
        print("[INFO] Pre-scanning frames to compute per-channel bounds…")
        ch_min_max_per_plane: Dict[str, Tuple[List[float], List[float]]] = {}

        for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
            ckpt_path = os.path.join(logdir, f"fine_last_{frameid}.tar")
            ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # optional foreground masking (same logic as your original)
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

            planes_scan = {k.split(".")[-1]: v.clone()
                           for k, v in ckpt["model_state_dict"].items()
                           if "k0" in k and "plane" in k and "residual" not in k}

            for plane, feat_scan in planes_scan.items():  # [1,C,H,W]
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

        # store bounds per plane
        for plane, (mn, mx) in ch_min_max_per_plane.items():
            PLANE_BOUNDS[plane] = list(zip(mn, mx))

        print("[INFO] Per-channel bounds (segment-wide, per plane):")
        for p, lst in PLANE_BOUNDS.items():
            for idx, (lo, hi) in enumerate(lst):
                print(f"  {p:>2s}-chan{idx:02d}: [{lo:.4f}, {hi:.4f}]")

    name = os.path.basename(logdir)
    out_root = os.path.join(
        logdir, f"planeimg_{args.startframe:02d}_{args.startframe + args.numframe - 1:02d}_{args.packing_mode}_{args.qmode}"
    )
    os.makedirs(out_root, exist_ok=True)
    print(f"[INFO] Saving plane images to {out_root}")

    # Meta-info to be stored per run
    meta_plane_bounds: Dict[str, List[List[Tuple[float, float]]]] = {}
    orig_sizes_map: Dict[str, List[Tuple[int,int]]] = {}  # per-plane list of (H2,W2)
    plane_size: Dict[str, Tuple[int, int, int, int]] = {}

    # ---------------------------------------------------------------------
    # iterate frames
    # ---------------------------------------------------------------------
    for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
        ckpt_path = os.path.join(logdir, f"fine_last_{frameid}.tar")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # ---------------- density grid → mask (same as before) ----------------
        density = ckpt["model_state_dict"]["density.grid"].clone()  # [1,1,Dy,Dx,Dz]
        voxel_size_ratio = ckpt["model_kwargs"]["voxel_size_ratio"]

        masks = None
        if "act_shift" in ckpt["model_state_dict"]:
            alpha = 1 - (torch.exp(density + ckpt["model_state_dict"]["act_shift"]) + 1) ** (-voxel_size_ratio)
            alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)
            mask = alpha < 1e-4
            density[mask] = -5
            feature_alpha = F.interpolate(alpha, size=tuple(np.array(density.shape[-3:]) * 3),
                                          mode="trilinear", align_corners=True)
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
                orig_sizes_map[plane_name] = []
                plane_size[plane_name] = tuple(feat.shape)

            if masks is not None:
                m = masks[plane_name].unsqueeze(1).repeat(1, feat.shape[1], 1, 1)
                feat = feat.masked_fill(m == 0, 0)

            # quantise -> [0,1] on a 16-bit grid
            bounds_this_plane: List[Tuple[float, float]] = []
            feat_q = quantise(feat, args.qmode, bounds_this_plane, plane_name=plane_name)
            meta_plane_bounds[plane_name].append(bounds_this_plane)

            # Pack (training-consistent): just call pack_planes_to_rgb
            y_pad, orig_hw = pack_planes_to_rgb(feat_q, align=DCVC_ALIGN, mode=args.packing_mode)  # [1,3,Hp,Wp], (H2,W2)
            orig_sizes_map[plane_name].append(tuple(orig_hw))

            # Save as 16-bit PNG (OpenCV expects BGR order but we just dump numeric values)
            base_dir = os.path.join(out_root, plane_name)
            os.makedirs(base_dir, exist_ok=True)
            y_u16 = _to16(y_pad[0].permute(1,2,0).cpu().numpy())   # H×W×3 uint16
            cv2.imwrite(os.path.join(base_dir, f"im{frameid + 1:05d}.png"), y_u16)

        # ---------------- density canvas ----------------
        dens_y_pad, dens_orig = make_density_rgb_image(density)               # [1,3,Hp,Wp], (Hc,Wc)
        dens_u16 = _to16(dens_y_pad[0].permute(1,2,0).cpu().numpy())          # H×W×3 uint16
        dens_dir = os.path.join(out_root, "density")
        os.makedirs(dens_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dens_dir, f"im{frameid + 1:05d}.png"), dens_u16)

        # Save orig mono-canvas size per frame for exact crop on unpack
        # store once per frame, in index order
        # (list[(Hc,Wc)] aligned to frame id range)
        # we’ll stuff this into the meta file below
        if "density_orig" not in plane_size:
            plane_size["density_orig"] = []
        plane_size["density_orig"].append(tuple(dens_orig))  # reuse in meta payload

    # ---------------------------------------------------------------------
    # persist meta
    # ---------------------------------------------------------------------
    torch.save({
        "qmode": args.qmode,
        "bounds": meta_plane_bounds,                  # dict[plane][frame][channel] -> (low, hi)
        "nbits": NBITS,
        "plane_size": plane_size,                    # shapes per plane; also "density_orig" list here
        "orig_sizes": orig_sizes_map,                # dict[plane][frame] -> (H2,W2)
        "packing_mode": args.packing_mode,
    }, os.path.join(out_root, "planes_frame_meta.nf"))

    print("[DONE] Conversion finished.")


if __name__ == "__main__":
    main()
