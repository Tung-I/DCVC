#!/usr/bin/env python
"""
Reconstruct Tri‑Plane checkpoints from decoded PNG frames produced by *triplane_packer.py*.
Supports the four packing strategies (**tiling / separate / grouped / correlation**)
plus the two quantisation modes (global, per_channel).

Example
-------
python image2plane.py \
    --logdir logs/out_triplane/flame_steak_old \
    --numframe 20 \
    --qp 22 \
    --strategy correlation \
    --qmode per_channel

python triplane_unpacker.py   --logdir logs/out_triplane/flame_steak_image   --numframe 1 --startframe 0   --qp 40   --strategy tiling   --qmode global
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm.auto import tqdm
from einops import rearrange

# -----------------------------------------------------------------------------

NBITS = 2 ** 16 - 1

def feat4d_to_dens5d(x: torch.Tensor, orient: str = "xz") -> torch.Tensor:
    """
    x: [1,C,H,W] per dens5d_to_feat4d; returns [1,1,Dy,Dx,Dz]
    """
    assert x.ndim == 5 and x.shape[0] == 1
    if orient == "yx":
        return rearrange(x, "b 1 Dz Dy Dx -> b 1 Dy Dx Dz")
    elif orient == "yz":
        return rearrange(x, "b 1 Dx Dy Dz -> b 1 Dy Dx Dz")
    elif orient == "xz":
        return rearrange(x, "b 1 Dy Dx Dz -> b 1 Dy Dx Dz")
    else:
        raise ValueError(f"Unknown orient={orient!r}")

def untile_image(image: np.ndarray, h: int, w: int, ndim: int) -> torch.Tensor:
    """Inverse of tile_maker in the packer."""
    feat = torch.zeros(1, ndim, h, w)
    x = y = 0
    for i in range(ndim):
        if y + w > image.shape[1]:
            y = 0
            x += h
        feat[0, i] = torch.from_numpy(image[x : x + h, y : y + w])
        y += w
    return feat

# -----------------------------------------------------------------------------


def dequantise_channel(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return t * (high - low) + low

# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True, help="root of planeimg and checkpoints")
    p.add_argument("--model_template", default="fine_last_0.tar", help="template ckpt for metadata")
    p.add_argument("--numframe", type=int, default=20)
    p.add_argument("--startframe", type=int, default=0)
    p.add_argument("--qp", type=int, default=20)
    p.add_argument("--strategy", choices=["tiling", "separate", "grouped", "correlation", "flatfour"], required=True)
    p.add_argument("--qmode", choices=["global", "per_channel"], required=True)
    p.add_argument("--dcvc", action="store_true", help="use compressed DCVC data or not")
    p.add_argument("--orient", choices=["yx", "yz", "xz"], default="xz", help="orientation of the feature planes")
    return p.parse_args()

# -----------------------------------------------------------------------------


def main():
    args = parse_args()

    S, N = args.startframe, args.startframe + args.numframe - 1

    root = args.logdir.rstrip("/")
    meta_root = os.path.join(root, f"planeimg_{S:02d}_{N:02d}_{args.strategy}_{args.qmode}")
    out_root = os.path.join(root, f"planeimg_{S:02d}_{N:02d}_{args.strategy}_{args.qmode}_qp{args.qp}")

    # ---------------------------------------------------------------------
    # load meta + template ckpt
    # ---------------------------------------------------------------------
    meta = torch.load(os.path.join(meta_root, "planes_frame_meta.nf"))
    bounds = meta["bounds"]            # dict[plane][frame][channel] -> (low, hi)
    group_map = meta.get("groups", None)
    qmode_saved = meta["qmode"]
    assert qmode_saved == args.qmode, "mismatched qmode between data and flag"
    nbits = meta["nbits"]

    ckpt_template = torch.load(os.path.join(root, args.model_template), map_location="cpu", weights_only=False)
    dens_shape = ckpt_template["model_state_dict"]["density.grid"].shape  # [1,1,Dy,Dx,Dz]
    Dy, Dx, Dz = dens_shape[2], dens_shape[3], dens_shape[4]

    # ---------------------------------------------------------------------
    # iterate over frames
    # ---------------------------------------------------------------------
    for frame_idx in tqdm(range(args.numframe)):
        fid = args.startframe + frame_idx
        ckpt_cur = {**ckpt_template}

        # ------------------- restore feature planes -------------------
        for plane_name, feat_shape in meta["plane_size"].items():
            C, H, W = feat_shape[1], feat_shape[2], feat_shape[3]
            feat = torch.zeros(1, C, H, W)

            if args.strategy == "tiling":
                folder = os.path.join(out_root, plane_name)
                img = cv2.imread(os.path.join(folder, f"im{fid + 1:05d}.png"), -1)
                if img is None:
                    raise FileNotFoundError(f"Image not found: {folder}/im{fid + 1:05d}.png")
                feat = untile_image(img.astype(np.float32) / nbits, H, W, C)

            elif args.strategy == "separate":
                # base = os.path.join(dec_root, plane_name)
                # for c in range(C):
                #     sub = os.path.join(base, ("dcvc_" if args.dcvc else "") + f"c{c}_qp{args.qp}")
                #     img_name = f"im{fid + 1:05d}.png" if args.dcvc else f"im{fid + 1:05d}_decoded.png"
                #     img = cv2.imread(os.path.join(sub, img_name), -1)
                #     feat[0, c] = torch.from_numpy(img.astype(np.float32) / nbits)
                raise NotImplementedError("Separate strategy not implemented")

            elif args.strategy in ("grouped", "correlation"):
                # base = os.path.join(dec_root, plane_name)
                # groups = (C + 2) // 3
                # if args.strategy == "grouped":
                #     group_indices: List[List[int]] = [list(range(3 * g, min(3 * g + 3, C))) for g in range(groups)]
                # else:  # correlation – use mapping stored in meta
                #     assert group_map is not None, "correlation groups not found in metadata"
                #     group_indices: List[List[int]] = group_map[plane_name][frame_idx]

                # for g, idxs in enumerate(group_indices):
                #     sub = os.path.join(base, ("dcvc_" if args.dcvc else "") + f"stream{g}_qp{args.qp}")
                #     img_name = f"im{fid + 1:05d}.png" if args.dcvc else f"im{fid + 1:05d}_decoded.png"
                #     arr = cv2.imread(os.path.join(sub, img_name), -1)
                #     if len(idxs) == 3:
                #         b, g_, r = cv2.split(arr)
                #         for k, ch in enumerate([r, g_, b]):
                #             feat[0, idxs[k]] = torch.from_numpy(ch.astype(np.float32) / nbits)
                #     else:
                #         feat[0, idxs[0]] = torch.from_numpy(arr.astype(np.float32) / nbits)
                raise NotImplementedError("Grouped/Correlation strategy not implemented")
            elif args.strategy == "flatfour":
                # # -----------------------------------------------------------
                # #  flat-4 unpacking
                # #    • one BGR image per plane & frame
                # #    • each colour channel contains a 2×2 tiling of four
                # #      mono feature maps -> recover 12 channels total
                # # -----------------------------------------------------------
                # assert C == 12 and C % 4 == 0, "flatfour expects exactly 12 channels"

                # # Locate the decoded (or DCVC) image
                # folder = os.path.join(dec_root, plane_name)
                # folder = f"{folder}_qp{args.qp}" if os.path.isdir(f"{folder}_qp{args.qp}") else folder
                # img_name = f"im{fid + 1:05d}.png" if args.dcvc else f"im{fid + 1:05d}_decoded.png"
                # arr = cv2.imread(os.path.join(folder, img_name), -1)          # (H*2, W*2, 3)

                # H2, W2 = arr.shape[:2]
                # h_orig, w_orig = H2 // 2, W2 // 2        # original per-channel dims

                # # OpenCV = BGR, but packer wrote BGR = [block2, block1, block0]
                # colour_blocks = [arr[:, :, 2],  # R  -> channels 0-3
                #                  arr[:, :, 1],  # G  -> channels 4-7
                #                  arr[:, :, 0]]  # B  -> channels 8-11

                # for block_idx, mono in enumerate(colour_blocks):
                #     mono_norm = mono.astype(np.float32) / NBITS
                #     # untile back to the 4 individual channels
                #     mono_feat = untile_image(mono_norm, h_orig, w_orig, 4)   # [1,4,H,W]
                #     start = block_idx * 4
                #     feat[0, start : start + 4] = mono_feat[0]
                raise NotImplementedError("Flat-four unpacking not implemented")
            else:
                raise ValueError("Unknown strategy")

            # ---------------- de‑quantisation -------------------
            if args.qmode == "global":
                low, hi = bounds[plane_name][frame_idx][0]  # same for all channels
                feat = dequantise_channel(feat, low, hi)
            elif args.qmode == "per_channel":  # per‑channel
                for c in range(C):
                    low, hi = bounds[plane_name][frame_idx][c]
                    feat[0, c] = dequantise_channel(feat[0, c], low, hi)
            else:
                raise ValueError("Unknown quantisation mode")

            ckpt_cur["model_state_dict"][f"k0.{plane_name}"] = feat.clone()

        # -------------------- density grid ---------------------
        dens_folder = os.path.join(out_root, "density")
        img_name = f"im{fid + 1:05d}.png"
        img = cv2.imread(os.path.join(dens_folder, img_name), -1)
        feat = untile_image(img.astype(np.float32) / NBITS, h=Dy, w=Dx, ndim=Dz)  # [1,Dz,Dy,Dx]
        feat = feat.unsqueeze(0)
        # undo density quantisation
        feat = feat * 35.0 - 5.0
        d5 = feat4d_to_dens5d(feat, orient=args.orient)     
        ckpt_cur["model_state_dict"]["density.grid"] = d5.clone()

        # save restored checkpoint
        torch.save(ckpt_cur, os.path.join(out_root, f"fine_last_{fid}.tar"))

    print("[DONE] Restored checkpoints written to", out_root)


if __name__ == "__main__":
    main()
