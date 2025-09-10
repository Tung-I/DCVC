import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import copy
from tqdm.auto import tqdm
from einops import rearrange
from PIL import Image

from TeTriRF.lib.dcvc_wrapper import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    _dens_to01, _dens_from01,
    _tile_1xCHW, _untile_to_1xCHW,
    _pad_to_align, _crop_from_align,
)

"""
Example
-------
python posthoc_plane_unpacker.py \
    --logdir logs/out_triplane/flame_steak_image_jpeg_qp10 \
    --numframe 1 \
    --qp 10 \
    --packing_mode flatten \
    --qmode global \
    --codec jpeg

python eval_plane_unpacker.py \
    --root_dir logs/dynerf_flame_steak/flame_steak_video_ds3 \
    --rec_dir logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_crf28_g10_yuv444p \
    --numframe 10 \
    --packing_mode flatten \
    --qmode global 

python eval_plane_unpacker.py \
        --root_dir logs/dynerf_flame_steak/flame_steak_video_ds3 \
        --rec_dir logs/dynerf_flame_steak/av1_crf44/compressed_av1_crf44_g10_yuv444p \
        --numframe 10 \
        --packing_mode flatten \
        --qmode global
"""

# -----------------------------------------------------------------------------

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

def _load_rgb01(path: str) -> torch.Tensor:
    """
    Robust loader for packed canvas PNG:
      - Always returns [1,3,H,W] float32 in [0,1] (CPU).
      - Works for 8-bit (uint8) and 16-bit (uint16) PNGs.
      - Forces RGB channel order.
    """
    im = Image.open(path)
    if im.mode == "I;16" or im.mode == "I;16B":
        # 16-bit grayscale not expected here; but handle generally
        im = im.convert("RGB")
    else:
        im = im.convert("RGB")
    arr = np.array(im)
    # dtype-aware scaling
    if arr.dtype == np.uint16:
        den = float(65535.0)
    else:
        # most PNGs from the video eval are 8-bit
        arr = arr.astype(np.uint8, copy=False)
        den = float(255.0)
    t01 = torch.from_numpy(arr.astype(np.float32) / den).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return t01

# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True, help="root of ckpt")
    p.add_argument("--rec_dir", required=True, help="root of reconstructed images")
    p.add_argument("--model_template", default="fine_last_0.tar", help="template ckpt for metadata")
    p.add_argument("--numframe", type=int, default=20)
    p.add_argument("--startframe", type=int, default=0)
    p.add_argument("--packing_mode", choices=["flatten", "separate", "grouped", "correlation", "mosaic"], default="flatten",
               help="Feature packing mode (match training cfg.dcvc.packing_mode)")
    p.add_argument("--qmode", choices=["global", "per_channel"], required=True)
    p.add_argument("--dcvc", action="store_true", help="use compressed DCVC data or not")
    p.add_argument("--orient", choices=["yx", "yz", "xz"], default="xz", help="orientation of the feature planes")
    return p.parse_args()

# -----------------------------------------------------------------------------

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
    bounds = meta["bounds"]            # dict[plane][frame][channel] -> (low, hi)
    group_map = meta.get("groups", None)
    qmode_saved = meta["qmode"]
    assert qmode_saved == args.qmode, "mismatched qmode between data and flag"
    nbits = meta["nbits"]
    packing_mode = meta.get("packing_mode", "flatten")
    orig_sizes_map = meta.get("orig_sizes", {})         # dict[plane] -> list[(H2,W2)]
    density_orig_list = meta.get("density_orig", [])    # list[(Hc,Wc)]

    ckpt_template = torch.load(os.path.join(root, args.model_template), map_location="cpu", weights_only=False)
    dens_shape = ckpt_template["model_state_dict"]["density.grid"].shape  # [1,1,Dy,Dx,Dz]
    Dy, Dx, Dz = dens_shape[2], dens_shape[3], dens_shape[4]

    # Cleanup the template to avoid confusion
    for plane_name, feat_shape in meta["plane_size"].items():
        ckpt_template["model_state_dict"][f"k0.{plane_name}"] = torch.zeros(feat_shape)
        ckpt_template["model_state_dict"]["density.grid"] = torch.zeros(1,1,Dy,Dx,Dz)


    # ---------------------------------------------------------------------
    # iterate over frames
    # ---------------------------------------------------------------------
    for frame_idx in tqdm(range(args.numframe)):
        fid = args.startframe + frame_idx
        # ckpt_cur = {**ckpt_template} 
        # Use deep copy: 
        ckpt_cur = copy.deepcopy(ckpt_template)
        
        # Q: What is this {**ckpt_template} syntax?
        # A: This syntax creates a shallow copy of the ckpt_template dictionary.


        # ------------------- restore feature planes -------------------
        for plane_name, feat_shape in meta["plane_size"].items():
            C, H, W = feat_shape[1], feat_shape[2], feat_shape[3]
            feat = torch.zeros(1, C, H, W)

            if args.packing_mode == "flatten":
                folder = os.path.join(rec_img_dir, plane_name)
                path = os.path.join(folder, f"im{fid + 1:05d}.png")
                arr = cv2.imread(path, -1)                                      # H×W×3 uint16
                if arr is None:
                    raise FileNotFoundError(f"Image not found: {path}")

                # Always RGB, scaled to [0,1], [1,3,Hp,Wp]
                y_pad = _load_rgb01(path)

                # crop back to the exact pre-pad size we saved at packing
                if plane_name not in orig_sizes_map:
                    raise RuntimeError(f"Missing orig_sizes for plane {plane_name}")
                H2, W2 = orig_sizes_map[plane_name][frame_idx]                  # (H2,W2)
                y = _crop_from_align(y_pad, (H2, W2))                           # [1,3,H2,W2]

                # invert packer -> [1,C,H,W] in [0,1]
                feat = unpack_rgb_to_planes(y, C, (H2, W2), mode=packing_mode)

            elif args.packing_mode == "separate":
                # base = os.path.join(dec_root, plane_name)
                # for c in range(C):
                #     sub = os.path.join(base, ("dcvc_" if args.dcvc else "") + f"c{c}_qp{args.qp}")
                #     img_name = f"im{fid + 1:05d}.png" if args.dcvc else f"im{fid + 1:05d}_decoded.png"
                #     img = cv2.imread(os.path.join(sub, img_name), -1)
                #     feat[0, c] = torch.from_numpy(img.astype(np.float32) / nbits)
                raise NotImplementedError("Separate packing_mode not implemented")

            elif args.packing_mode in ("grouped", "correlation"):
                # base = os.path.join(dec_root, plane_name)
                # groups = (C + 2) // 3
                # if args.packing_mode == "grouped":
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
                raise NotImplementedError("Grouped/Correlation packing_mode not implemented")
            elif args.packing_mode == "flatfour":
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
                raise ValueError("Unknown packing_mode")

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
        dens_folder = os.path.join(rec_img_dir, "density")
        path = os.path.join(dens_folder, f"im{fid + 1:05d}.png")
        arr = cv2.imread(path, -1)                                       # H×W×3 uint16
        if arr is None:
            raise FileNotFoundError(f"Image not found: {path}")

        # back to float01 tensor on CPU, [1,3,Hp,Wp]
        y_pad = _load_rgb01(path) 

        # crop to original mono canvas size saved during packing
        Hc, Wc = density_orig_list[frame_idx]
        y = _crop_from_align(y_pad, (Hc, Wc))                            # [1,3,Hc,Wc]

        # take mono canvas (any channel) and un-tile to [1,C=Dy,H=Dx,W=Dz]
        mono = y[:, 0]                                                   # [1,Hc,Wc]
        d01_chw = _untile_to_1xCHW(mono.squeeze(0), Dy, Dx, Dz)          # [1,Dy,Dx,Dz]

        # map back to raw density and view as [1,1,Dy,Dx,Dz]
        d5 = _dens_from01(d01_chw).view(1,1,Dy,Dx,Dz)
 

        # save restored checkpoint
        ckpt_cur["model_state_dict"]["density.grid"] = d5.clone()
        torch.save(ckpt_cur, os.path.join(rec_img_dir, f"fine_last_{fid}.tar"))

    print("[DONE] Restored checkpoints written to", rec_img_dir)


if __name__ == "__main__":
    main()
