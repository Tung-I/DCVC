#!/usr/bin/env python
"""
Convert optimized Tri‑Plane feature volumes into 2‑D image streams that can be fed to a
standard video codec.  Four packing strategies are supported and **two different
quantisation modes** (global vs. per‑channel), giving 8 baselines in total.

Packing strategies
------------------
1. **flatten**      – TeTriRF's original flatten‑and‑tile layout
2. **separate**    – one greyscale image per feature channel
3. **grouped**     – fixed RGB triplets (0‑2, 3‑5, 6‑8, 9‑11)
4. **correlation** – *new*: greedy correlation‑based clustering of the most
                     correlated 3‑channel groups

Quantisation modesp`
------------------
* **global   (default)**  – fixed numeric range [low_bound, high_bound]
* **per_channel**         – individual min/max range for each channel (frame‑wise)

Usage example
-------------
```
python posthoc_plane_packer.py \
      --logdir logs/out_triplane/flame_steak_image_jpeg_qp10 \
      --numframe 1 \
      --packing_mode flatten \
      --qmode global
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
from einops import rearrange

from TeTriRF.lib.dcvc_wrapper import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    _dens_to01, _dens_from01,
    _tile_1xCHW, _untile_to_1xCHW,
    _pad_to_align, _crop_from_align,
)
DCVC_ALIGN = 32

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

to8b  = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
_to16 = lambda x: ((2 ** 16 - 1) * np.clip(x, 0, 1)).astype(np.uint16)


def make_density_rgb_image(density_grid: torch.Tensor) -> torch.Tensor:
    """
    Training-consistent density packing:
      [1,1,Dy,Dx,Dz] --to01--> view as [1,C=Dy,H=Dx,W=Dz]
      --_tile_1xCHW--> mono canvas [Hc,Wc] --repeat 3ch--> [1,3,Hc,Wc]
      --pad align--> [1,3,Hp,Wp]
    Returns:
      y_pad : [1,3,Hp,Wp] float in [0,1]
      orig  : (Hc, Wc)    (for cropping back before un-tiling)
    """
    assert density_grid.dim() == 5 and density_grid.shape[:2] == (1,1)
    _, _, Dy, Dx, Dz = density_grid.shape
    d01 = _dens_to01(density_grid)               # [1,1,Dy,Dx,Dz]
    d01_chw = d01.view(1, Dy, Dx, Dz)            # [1,C,H,W] with C=Dy,H=Dx,W=Dz
    mono, (Hc, Wc) = _tile_1xCHW(d01_chw)        # [Hc,Wc]
    y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1,3,Hc,Wc]
    y_pad, orig = _pad_to_align(y, align=DCVC_ALIGN)    # [1,3,Hp,Wp], (Hc,Wc)
    return y_pad, orig



def density_quantize(density: torch.Tensor, nbits: int) -> torch.Tensor:
    """Simple hard‑clipped linear quantiser for density volumes."""
    levels = 2 ** nbits - 1
    data = density.clone()
    data.clamp_(-5, 30)
    data = (data + 5) / 35.0
    data = torch.round(data * levels) / levels
    return data


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
    assert C % 3 == 0, "correlation packing_mode expects C to be a multiple of 3"

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
    p.add_argument("--qmode", type=str, default="global", choices=["global", "per_channel"],
                   help="quantisation mode")
    p.add_argument("--codec", type=str, default="h265", help="placeholder – not used here")
    p.add_argument("--orient", choices=["yx", "yz", "xz"], default="xz", help="orientation of the feature planes")
    p.add_argument("--packing_mode", choices=["flatten", "separate", "grouped", "correlation", "mosaic"], default="flatten",
               help="Feature packing mode (match training cfg.dcvc.packing_mode)")
    return p.parse_args()


# Default global min/max used in TeTriRF
GLOBAL_LOW, GLOBAL_HIGH = -20.0, 20.0
NBITS = 2 ** 16 - 1  # 16‑bit PNG
CLIP_PCT = 0.1       # clip 0.5 % low / high tails (per channel)
PLANE_BOUNDS: Dict[str, List[Tuple[float, float]]] = {}

def _clip_percentiles(ch: torch.Tensor, pct: float) -> Tuple[float, float]:
    """Return (low, high) percentiles after clipping *pct*%% on each side."""
    if pct <= 0:
        return float(ch.min()), float(ch.max())
    low_q = torch.quantile(ch, pct / 100.0).item()
    high_q = torch.quantile(ch, 1.0 - pct / 100.0).item()
    if high_q - low_q < 1e-6:
        # fallback to full range when nearly constant
        return float(ch.min()), float(ch.max())
    return low_q, high_q

def quantise(feat: torch.Tensor, qmode: str, bounds_out: List[Tuple[float, float]], plane_name=None) -> torch.Tensor:
    """Quantise *feat* (shape [1, C, H, W]).

    * global   – fixed [-20, 20]
    * per_channel – min/max after clipping CLIP_PCT tails to mitigate outliers
    """
    if qmode == "global":
        low, high = GLOBAL_LOW, GLOBAL_HIGH
        bounds_out.extend([(low, high)] * feat.shape[1])
        norm = (feat - low) / (high - low)
        return torch.round(norm.clamp(0, 1) * NBITS) / NBITS

    # per‑channel
    C = feat.shape[1]
    q_ch = []
    for c in range(C):
        low, high = PLANE_BOUNDS[plane_name][c]         # <<< plane-specific lookup
        ch        = feat[0, c]
        bounds_out.append((low, high))
        norm = (ch.clamp(low, high) - low) / (high - low + 1e-8)
        q_ch.append(torch.round(norm * NBITS) / NBITS)
    return torch.stack(q_ch, dim=0).unsqueeze(0).clamp_(0, 1)




def main():
    args = parse_args()

    logdir = args.logdir.rstrip("/")

    # ---------------------------------------------------------------------
    # Pre-scan: collect global min / max for every feature channel
    #           across the whole segment (all planes, all frames)
    # ---------------------------------------------------------------------
    if args.qmode == "per_channel":
        print("[INFO] Pre-scanning frames to compute per-channel bounds…")
        ch_min, ch_max = None, None   # will become length-C lists

        for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
            ckpt_path = os.path.join(logdir, f"fine_last_{frameid}.tar")
            ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # ---------- optional foreground masking (same logic as pack loop) ----------
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

            for plane, feat_scan in planes_scan.items():          # [1,C,H,W]
                if masks is not None:
                    m = masks[plane].unsqueeze(1).repeat(1, feat_scan.shape[1], 1, 1)
                    feat_scan = feat_scan.masked_fill(m == 0, 0)

                C_scan = feat_scan.shape[1]
                if ch_min is None:                                # first time
                    ch_min = [float("inf")]  * C_scan
                    ch_max = [float("-inf")] * C_scan

                # update running min / max
                for c in range(C_scan):
                    ch         = feat_scan[0, c]
                    ch_min[c]  = min(ch_min[c],  ch.min().item())
                    ch_max[c]  = max(ch_max[c],  ch.max().item())

        # After scanning all frames, store bounds **per plane**
                PLANE_BOUNDS[plane] = list(zip(ch_min, ch_max))

        # Pretty-print once after loop
            print("[INFO] Per-channel bounds (segment-wide, per plane):")
            for p, lst in PLANE_BOUNDS.items():
                for idx, (lo, hi) in enumerate(lst):
                    print(f"  {p:>2s}-chan{idx:02d}: [{lo:.4f}, {hi:.4f}]")


    name = os.path.basename(logdir)
    out_root = os.path.join(logdir, f"planeimg_{args.startframe:02d}_{args.startframe + args.numframe - 1:02d}_{args.packing_mode}_{args.qmode}")
    os.makedirs(out_root, exist_ok=True)
    print(f"[INFO] Saving plane images to {out_root}")

    # Meta‑info to be stored per run
    meta_plane_bounds: Dict[str, List[List[Tuple[float, float]]]] = {}
    meta_plane_groups: Dict[str, List[List[List[int]]]] = {}  # only for correlation

    frame_cnt = 0
    for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
        frame_cnt += 1
        ckpt_path = os.path.join(logdir, f"fine_last_{frameid}.tar")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # ---------------- density grid → mask ----------------
        density = ckpt["model_state_dict"]["density.grid"].clone() # torch.Size([1, 1, 181, 181, 280])
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

            # ----------------------------------------------------------
            # quantise: use *segment* bounds when qmode = per_channel
            # ----------------------------------------------------------
            feat_q = quantise(feat, args.qmode, bounds_this_plane, plane_name=plane_name)
            meta_plane_bounds[plane_name].append(bounds_this_plane)  # list of frames

            C, H, W = feat_q.shape[1:]
            base_dir = os.path.join(out_root, plane_name)

            if args.packing_mode == "flatten":
                os.makedirs(base_dir, exist_ok=True)

                # Pack to 3-channel with align padding (exactly as training)
                x01 = feat_q                                 # already in [0,1] from your quantiser
                y_pad, orig_hw = pack_planes_to_rgb(x01, align=DCVC_ALIGN, mode=args.packing_mode)   # [1,3,Hp,Wp]
                y_u16 = _to16(y_pad[0].permute(1,2,0).cpu().numpy())                                 # H×W×3 uint16 (BGR-ish)

                # Save colour PNG
                cv2.imwrite(os.path.join(base_dir, f"im{frameid + 1:05d}.png"), y_u16)

                # Stash per-frame orig size so we can crop back before unpacking
                if "orig_sizes" not in PLANE_BOUNDS:
                    PLANE_BOUNDS["orig_sizes"] = {}
                PLANE_BOUNDS["orig_sizes"].setdefault(plane_name, []).append(tuple(orig_hw))

            elif args.packing_mode == "separate":
                for c in range(C):
                    ch_dir = os.path.join(base_dir, f"c{c}")
                    os.makedirs(ch_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(ch_dir, f"im{frameid + 1:05d}.png"),
                                _to16(feat_q[0, c].cpu().numpy()))

            elif args.packing_mode == "grouped":
                for gi in range(0, C, 3):
                    stream_dir = os.path.join(base_dir, f"stream{gi // 3}")
                    os.makedirs(stream_dir, exist_ok=True)
                    triplet = feat_q[0, gi : gi + 3]
                    bgr = np.stack([triplet[2], triplet[1], triplet[0]], axis=-1)
                    cv2.imwrite(os.path.join(stream_dir, f"im{frameid + 1:05d}.png"), _to16(bgr))

            elif args.packing_mode == "correlation":
                if plane_name not in meta_plane_groups:
                    meta_plane_groups[plane_name] = []
                current_groups = correlation_groups(feat)
                meta_plane_groups[plane_name].append(current_groups)
                for gi, idxs in enumerate(current_groups):
                    stream_dir = os.path.join(base_dir, f"stream{gi}")
                    os.makedirs(stream_dir, exist_ok=True)
                    triplet = feat_q[0, idxs]
                    bgr = np.stack([triplet[2], triplet[1], triplet[0]], axis=-1)
                    cv2.imwrite(os.path.join(stream_dir, f"im{frameid + 1:05d}.png"), _to16(bgr))

            elif args.packing_mode == "mosaic":
                # ───────────────  “flat-4-to-1”  ───────────────
                #   • 12-channel feature volume → single RGB image
                #   • Every 4 channels are tiled into one 2×2 mono image,
                #     giving three such mono images → mapped to B,G,R.
                # ------------------------------------------------
                assert C == 12 and C % 4 == 0, "mosaic expects exactly 12 channels"

                os.makedirs(base_dir, exist_ok=True)
                H2, W2 = H * 2, W * 2      # canvas for 4-way tiling
                mono_planes = []

                for fi in range(0, C, 4):                       # (0-3, 4-7, 8-11)
                    block = feat_q[:, fi : fi + 4]               # [1,4,H,W]
                    mono  = tile_maker(block, h=H2, w=W2)        # [H2,W2]
                    mono_planes.append(mono.cpu())

                # OpenCV expects BGR, so reverse RGB order
                bgr = np.stack([mono_planes[2], mono_planes[1], mono_planes[0]], axis=-1)
                cv2.imwrite(os.path.join(base_dir, f"im{frameid + 1:05d}.png"),
                            _to16(bgr))

        # ---------------- density plane ----------------
        dens_y_pad, dens_orig = make_density_rgb_image(density)               # [1,3,Hp,Wp], (Hc,Wc)
        dens_u16 = _to16(dens_y_pad[0].permute(1,2,0).cpu().numpy())           # H×W×3 uint16
        dens_dir = os.path.join(out_root, "density")
        os.makedirs(dens_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dens_dir, f"im{frameid + 1:05d}.png"), dens_u16)

        # Save orig mono-canvas size per frame for exact crop on unpack
        if "density_orig" not in PLANE_BOUNDS:
            PLANE_BOUNDS["density_orig"] = []
        PLANE_BOUNDS["density_orig"].append(tuple(dens_orig))                  # (Hc,Wc)


    # ---------------------------------------------------------------------
    # persist meta information once after the loop
    # ---------------------------------------------------------------------
    torch.save({
        "qmode": args.qmode,
        "bounds": meta_plane_bounds,
        "nbits": NBITS,
        "plane_size": {p: feat.shape for p, feat in planes.items()},
        "groups": meta_plane_groups if args.packing_mode == "correlation" else None,
        "orig_sizes": PLANE_BOUNDS.get("orig_sizes", {}),     # dict[plane][frame] -> (H2,W2)
        "density_orig": PLANE_BOUNDS.get("density_orig", []), # list[(Hc,Wc)] per frame
        "packing_mode": args.packing_mode,                    # remember how we packed planes
    }, os.path.join(out_root, "planes_frame_meta.nf"))


    print("[DONE] Conversion finished.")


if __name__ == "__main__":
    main()
