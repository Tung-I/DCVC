#!/usr/bin/env python3
import os
import copy
import argparse
from typing import Tuple, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ***** Use the SAME utilities as training *****
from src.models.model_utils import (
    unpack_rgb_to_planes,
    crop_from_align,
    unpack_density_from_rgb,  # uses grid mode
    dens_from01,
)

"""
Usage:
    python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/lego_image \
    --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global \
    --qp 20 --codec jpeg
"""

def _load_rgb01(path: str) -> torch.Tensor:
    """Load PNG/JPEG/etc. to [1,3,H,W] float32 in [0,1] (CPU), RGB order."""
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    den = 65535.0 if arr.dtype == np.uint16 else 255.0
    arr = arr.astype(np.float32) / den
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def dequantise_channel(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return t * (high - low) + low

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True, help="root of ckpt")
    p.add_argument("--model_template", default="fine_last_0.tar", help="template ckpt for metadata")
    p.add_argument("--numframe", type=int, default=20)
    p.add_argument("--startframe", type=int, default=0)
    p.add_argument("--plane_packing_mode", choices=["flatten", "mosaic", "flat4"], required=True,
                   help="Must match the packer plane mode used for the data")
    p.add_argument("--grid_packing_mode", choices=["flatten", "mosaic", "flat4"], required=True,
                   help="Must match the packer grid mode used for the data")
    p.add_argument("--qmode", choices=["global", "per_channel"], required=True)
    p.add_argument("--orient", choices=["yx", "yz", "xz"], default="xz",
                   help="(kept for compatibility; not used by unpacker)")
    p.add_argument("--qp", type=int, default=40, help="(for codec folder name if using jpeg)")
    p.add_argument("--codec", choices=["dcvc", "jpeg"], default="dcvc",
                   help="Choose folder containing reconstructed images")
    return p.parse_args()

def main():
    args = parse_args()
    root = args.root_dir
    S, N = args.startframe, args.startframe + args.numframe - 1

    # meta produced by the packer
    meta_dir = os.path.join(
        root,
        f"planeimg_{S:02d}_{N:02d}_{args.plane_packing_mode}_{args.grid_packing_mode}_{args.qmode}"
    )
    # reconstructed images folder (e.g., *_jpeg_qp80)
    rec_img_dir = os.path.join(
        root,
        f"planeimg_{S:02d}_{N:02d}_{args.plane_packing_mode}_{args.grid_packing_mode}_{args.qmode}_{args.codec}_qp{args.qp}"
    )

    print(f"[INFO] Loaded meta data from {meta_dir}")
    print(f"[INFO] Load reconstructed images from {rec_img_dir}")

    # ---- load meta + template ckpt ----
    meta = torch.load(os.path.join(meta_dir, "planes_frame_meta.nf"))
    bounds            = meta["bounds"]                         # dict[plane][frame][channel] -> (low, hi)
    qmode_saved       = meta["qmode"]
    assert qmode_saved == args.qmode, "mismatched qmode between data and flag"
    plane_mode_saved  = meta.get("plane_packing_mode", "flatten")
    grid_mode_saved   = meta.get("grid_packing_mode",  "flatten")

    # sanity: CLI must match what was packed
    assert plane_mode_saved == args.plane_packing_mode, f"plane_packing_mode mismatch: {plane_mode_saved} vs {args.plane_packing_mode}"
    assert grid_mode_saved  == args.grid_packing_mode,  f"grid_packing_mode mismatch: {grid_mode_saved} vs {args.grid_packing_mode}"

    nbits             = meta["nbits"]
    orig_sizes_map    = meta.get("orig_sizes", {})             # dict[plane] -> list[(H2,W2)]
    plane_size        = meta["plane_size"]                     # dict[plane] -> shape, plus "density_orig" list
    density_orig_list = plane_size.get("density_orig", [])

    ckpt_template = torch.load(os.path.join(root, args.model_template), map_location="cpu", weights_only=False)
    dens_shape = ckpt_template["model_state_dict"]["density.grid"].shape  # [1,1,Dy,Dx,Dz]
    Dy, Dx, Dz = dens_shape[2], dens_shape[3], dens_shape[4]

    # Cleanup the template to avoid confusion
    for plane_name, feat_shape in plane_size.items():
        if plane_name == "density_orig":
            continue
        ckpt_template["model_state_dict"][f"k0.{plane_name}"] = torch.zeros(feat_shape)
    ckpt_template["model_state_dict"]["density.grid"] = torch.zeros(1,1,Dy,Dx,Dz)

    # ---- iterate over frames ----
    for frame_idx in tqdm(range(args.numframe)):
        fid = args.startframe + frame_idx
        ckpt_cur = copy.deepcopy(ckpt_template)

        # -------- feature planes --------
        for plane_name, feat_shape in plane_size.items():
            if plane_name == "density_orig":
                continue

            C, H, W = feat_shape[1], feat_shape[2], feat_shape[3]
            folder  = os.path.join(rec_img_dir, plane_name)
            path    = os.path.join(folder, f"im{fid + 1:05d}.png")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")

            # crop to *pre-pad* size saved by the packer
            if plane_name not in orig_sizes_map:
                raise RuntimeError(f"Missing orig_sizes for plane {plane_name}")
            H2, W2 = orig_sizes_map[plane_name][frame_idx]

            y_pad = _load_rgb01(path)                 # [1,3,Hp,Wp], RGB
            y     = crop_from_align(y_pad, (H2, W2))  # [1,3,H2,W2]

            # invert pack → [1,C,H,W] in [0,1]
            feat01 = unpack_rgb_to_planes(y, C, (H2, W2), mode=plane_mode_saved)

            # de-quantise
            if args.qmode == "global":
                low, hi = bounds[plane_name][frame_idx][0]
                feat = dequantise_channel(feat01, low, hi)
            elif args.qmode == "per_channel":
                feat = torch.zeros_like(feat01)
                for c in range(C):
                    low, hi = bounds[plane_name][frame_idx][c]
                    feat[0, c] = dequantise_channel(feat01[0, c], low, hi)
            else:
                raise ValueError("Unknown quantisation mode")

            ckpt_cur["model_state_dict"][f"k0.{plane_name}"] = feat.clone()

        # -------- density grid --------
        dens_path = os.path.join(rec_img_dir, "density", f"im{fid + 1:05d}.png")
        if not os.path.isfile(dens_path):
            raise FileNotFoundError(f"Image not found: {dens_path}")

        if not density_orig_list:
            raise RuntimeError("Missing density_orig list in meta (per-frame orig crop size).")
        H2d, W2d = density_orig_list[frame_idx]      # tuple returned by pack_density_to_rgb

        y_pad = _load_rgb01(dens_path)               # [1,3,Hp,Wp]
        y     = crop_from_align(y_pad, (H2d, W2d))   # [1,3,H2d,W2d]

        # invert density pack (returns d01 in [0,1] as [1,1,Dy,Dx,Dz])
        d01_5 = unpack_density_from_rgb(y, Dy, Dx, Dz, (H2d, W2d), mode=grid_mode_saved)
        # map back to raw density range [-5,30]
        d5 = dens_from01(d01_5)
        ckpt_cur["model_state_dict"]["density.grid"] = d5.clone()

        # save restored checkpoint
        torch.save(ckpt_cur, os.path.join(rec_img_dir, f"fine_last_{fid}.tar"))

    print("[DONE] Restored checkpoints written to", rec_img_dir)

if __name__ == "__main__":
    main()
