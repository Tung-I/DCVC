#!/usr/bin/env python3
import os
import copy
import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm

# ***** Use the SAME utilities as training *****
from src.models.model_utils import (
    unpack_rgb_to_planes,
    crop_from_align,
    untile_to_1xCHW,
    dens_from01,
)

def _load_rgb01(path: str) -> torch.Tensor:
    """
    Robust loader for packed canvas PNG:
      - Always returns [1,3,H,W] float32 in [0,1] (CPU).
      - Works for 8-bit (uint8) and 16-bit (uint16) PNGs.
      - Forces RGB channel order.
    """
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    if arr.dtype == np.uint16:
        den = float(65535.0)
    else:
        arr = arr.astype(np.uint8, copy=False)
        den = float(255.0)
    t01 = torch.from_numpy(arr.astype(np.float32) / den).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return t01

def dequantise_channel(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return t * (high - low) + low

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True, help="root of ckpt")
    p.add_argument("--rec_dir", required=True, help="root of reconstructed images (after codec)")
    p.add_argument("--model_template", default="fine_last_0.tar", help="template ckpt for metadata")
    p.add_argument("--numframe", type=int, default=20)
    p.add_argument("--startframe", type=int, default=0)
    p.add_argument("--packing_mode", choices=["flatten", "mosaic"], default="flatten",
                   help="Feature packing mode (match packer/training)")
    p.add_argument("--qmode", choices=["global", "per_channel"], required=True)
    p.add_argument("--orient", choices=["yx", "yz", "xz"], default="xz",
                   help="(kept for compatibility; not used by unpacker)")
    return p.parse_args()

def main():
    args = parse_args()
    root = args.root_dir
    S, N = args.startframe, args.startframe + args.numframe - 1

    meta_dir = os.path.join(root, f"planeimg_{S:02d}_{N:02d}_{args.packing_mode}_{args.qmode}")
    rec_img_dir = args.rec_dir
    print(f"[INFO] Loaded meta data from {meta_dir}")
    print(f"[INFO] Loaded ckpt template from {root}")
    print(f"[INFO] Load reconstructed images from {rec_img_dir}")
    print(f"[INFO] Writing unpacked tensor planes to {rec_img_dir}")

    # ---------------------------------------------------------------------
    # load meta + template ckpt
    # ---------------------------------------------------------------------
    meta = torch.load(os.path.join(meta_dir, "planes_frame_meta.nf"))
    bounds = meta["bounds"]                         # dict[plane][frame][channel] -> (low, hi)
    qmode_saved = meta["qmode"]
    assert qmode_saved == args.qmode, "mismatched qmode between data and flag"
    nbits = meta["nbits"]
    packing_mode = meta.get("packing_mode", "flatten")
    orig_sizes_map = meta.get("orig_sizes", {})     # dict[plane] -> list[(H2,W2)]

    plane_size = meta["plane_size"]                 # dict[plane] -> shape, plus "density_orig" list
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

    # ---------------------------------------------------------------------
    # iterate over frames
    # ---------------------------------------------------------------------
    for frame_idx in tqdm(range(args.numframe)):
        fid = args.startframe + frame_idx
        ckpt_cur = copy.deepcopy(ckpt_template)

        # ------------------- restore feature planes -------------------
        for plane_name, feat_shape in plane_size.items():
            if plane_name == "density_orig":
                continue

            C, H, W = feat_shape[1], feat_shape[2], feat_shape[3]
            folder = os.path.join(rec_img_dir, plane_name)
            path = os.path.join(folder, f"im{fid + 1:05d}.png")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")

            # Load [1,3,Hp,Wp] float01 and crop to the *pre-pad* size saved by the packer
            if plane_name not in orig_sizes_map:
                raise RuntimeError(f"Missing orig_sizes for plane {plane_name}")
            H2, W2 = orig_sizes_map[plane_name][frame_idx]  # (H2,W2) stored by packer

            y_pad = _load_rgb01(path)                       # [1,3,Hp,Wp] float01
            y = crop_from_align(y_pad, (H2, W2))            # [1,3,H2,W2]

            # Invert packer → [1,C,H,W] in [0,1]
            feat01 = unpack_rgb_to_planes(y, C, (H2, W2), mode=packing_mode)

            # De-quantise
            if args.qmode == "global":
                low, hi = bounds[plane_name][frame_idx][0]  # same for all channels
                feat = dequantise_channel(feat01, low, hi)
            elif args.qmode == "per_channel":
                feat = torch.zeros_like(feat01)
                for c in range(C):
                    low, hi = bounds[plane_name][frame_idx][c]
                    feat[0, c] = dequantise_channel(feat01[0, c], low, hi)
            else:
                raise ValueError("Unknown quantisation mode")

            ckpt_cur["model_state_dict"][f"k0.{plane_name}"] = feat.clone()

        # -------------------- density grid ---------------------
        dens_folder = os.path.join(rec_img_dir, "density")
        path = os.path.join(dens_folder, f"im{fid + 1:05d}.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")

        y_pad = _load_rgb01(path)  # [1,3,Hp,Wp]

        if not density_orig_list:
            raise RuntimeError("Missing density_orig list in meta (mono canvas Hc,Wc per frame).")
        Hc, Wc = density_orig_list[frame_idx]
        y = crop_from_align(y_pad, (Hc, Wc))             # [1,3,Hc,Wc]

        # take mono canvas (any channel), un-tile to [1,C=Dy,H=Dx,W=Dz]
        mono = y[:, 0]                                   # [1,Hc,Wc]
        d01_chw = untile_to_1xCHW(mono.squeeze(0), Dy, Dx, Dz)  # [1,Dy,Dx,Dz]

        # map back to raw density and view as [1,1,Dy,Dx,Dz]
        d5 = dens_from01(d01_chw).view(1,1,Dy,Dx,Dz)
        ckpt_cur["model_state_dict"]["density.grid"] = d5.clone()

        # save restored checkpoint
        torch.save(ckpt_cur, os.path.join(rec_img_dir, f"fine_last_{fid}.tar"))

    print("[DONE] Restored checkpoints written to", rec_img_dir)


if __name__ == "__main__":
    main()
