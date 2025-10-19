#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_plane_video_compress.py

Reconstruct (encode+decode) packed TriPlane canvases with HEVC (x265) or AV1 (libaom),
entirely in-memory, mirroring the training-time video roundtrip.

Input tree (from posthoc_plane_packer.py):
  <segment_root>/
    xy_plane/im00001.png ... im00010.png
    xz_plane/...
    yz_plane/...
    density/   (optional)

Output tree (default):
  <segment_root>/<codec>_crf{CRF}_g{GOP}_{pixfmt}_a{ALIGN}/
    xy_plane/im00001.png ...
    xz_plane/...
    yz_plane/...
    density/...
  + encoded_bits.txt
  + metrics.txt

BPP is computed as total_bits / (T * Hp * Wp) where (Hp,Wp) are post-align sizes, matching training.

python eval_plane_video_compress.py \
  --segment-root logs/dynerf_flame_steak/flame_steak_video_ds3 \
  --startframe 0 --numframe 10 --packing_mode flatten --qmode global \
  --codec hevc --crf 28 --gop 10 --pix-fmt yuv444p

python eval_plane_video_compress.py \
    --startframe 0 --numframe 20 --packing_mode flatten \
    --qmode global --gop 20 --pix-fmt yuv444p \
    --segment-root logs/dynerf_flame_steak/av1_qp20 --codec av1 --qp 20
"""

import argparse
import io
import os
import sys
from fractions import Fraction
from typing import List, Tuple, Dict

import av  # PyAV
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from src.models.model_utils import (
    hevc_video_roundtrip,
    av1_video_roundtrip,
    vp9_video_roundtrip,
)

# ---------------------------------------------------------------------
# Utilities (file IO, sorting, tensor helpers)
# ---------------------------------------------------------------------

def _natkey(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_pngs_sorted(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    fs = [f for f in os.listdir(d) if f.lower().endswith('.png')]
    fs.sort(key=_natkey)
    return [os.path.join(d, f) for f in fs]

def load_png_stack(paths: List[str]) -> torch.Tensor:
    """Return [T,3,H,W] float in [0,1] on CPU."""
    frames = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        arr = np.array(im, dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
        frames.append(t)
    if not frames:
        return None
    return torch.stack(frames, dim=0)

def save_png_stack(out_dir: str, rec: torch.Tensor, names: List[str]):
    """rec: [T,3,H,W] float in [0,1] CPU."""
    os.makedirs(out_dir, exist_ok=True)
    T = rec.shape[0]
    for i in range(T):
        arr = (rec[i].clamp(0,1)*255.0 + 0.5).byte().permute(1,2,0).cpu().numpy()
        Image.fromarray(arr, mode='RGB').save(os.path.join(out_dir, os.path.basename(names[i])))

def pad_to_align(canv: torch.Tensor, align: int) -> Tuple[torch.Tensor, int, int]:
    """canv: [T,3,H,W] -> [T,3,Hp,Wp], pad bottom/right with replicate."""
    _, _, H, W = canv.shape
    pad_h = (align - (H % align)) % align
    pad_w = (align - (W % align)) % align
    if pad_h == 0 and pad_w == 0:
        return canv, H, W
    canv_pad = F.pad(canv, (0, pad_w, 0, pad_h), mode='replicate')
    return canv_pad, H, W

# ---------------------------------------------------------------------
# Main eval: compress each stream directory as a video segment
# ---------------------------------------------------------------------

STREAM_DIRS = ["xy_plane", "xz_plane", "yz_plane", "density"]

def run_one_stream(
    in_dir: str,
    out_dir: str,
    *,
    codec: str,
    fps: int,
    gop: int,
    qp: int,
    preset: str,
    cpu_used: str,
    pix_fmt: str,
    align: int,
    packing_mode: str,
) -> Dict[str, float]:
    """
    Compress a single stream directory (sequence of packed PNG canvases), write recon PNGs,
    and return metrics computed on the packed canvases:
      { 'T','H','W','Hp','Wp','total_bits','bpp','psnr' }
    """
    names = list_pngs_sorted(in_dir)
    if not names:
        return {}

    # Load packed canvases (always 3-channel on disk)
    canv = load_png_stack(names)  # [T,3,H,W], CPU float in [0,1]
    T, _, H, W = canv.shape

    # Match training: pad to align, then encode/decode, then crop back
    canv_pad, H2, W2 = pad_to_align(canv, align=align)   # [T,3,Hp,Wp]
    Hp, Wp = canv_pad.shape[-2:]

    # Decide grayscale path:
    #  - Planes: grayscale if packing_mode == "flatten" (we duplicated channels when saving)
    #  - Density: always grayscale (packed as mono tiled, saved as RGB duplicates)
    stream_name = os.path.basename(in_dir.rstrip("/"))
    is_density = (stream_name == "density")
    use_gray = is_density or (packing_mode == "flatten")

    # Build the input to the codec helpers to match training behavior
    if use_gray:
        mono = canv_pad[:, :1, ...].contiguous()  # [T,1,Hp,Wp] — take the first (duplicated) channel
        if codec == "hevc":
            rec_mono, total_bits = hevc_video_roundtrip(
                mono, fps=fps, gop=gop, qp=qp, preset=preset, pix_fmt=pix_fmt, grayscale=True
            )
        elif codec == "av1":
            rec_mono, total_bits = av1_video_roundtrip(
                mono, fps=fps, gop=gop, qp=qp, cpu_used=cpu_used, pix_fmt=pix_fmt, grayscale=True
            )
        elif codec == "vp9":
            rec_mono, total_bits = vp9_video_roundtrip(
                mono, fps=fps, gop=gop, qp=qp, cpu_used=cpu_used, pix_fmt=pix_fmt, grayscale=True
            )
        else:
            raise ValueError(f"Unsupported --codec {codec}. Use 'hevc', 'av1', or 'vp9'.")
        # Expand back to 3 channels so the unpacking script (and PNG writer) expect the same shape
        rec_pad = rec_mono.repeat(1, 3, 1, 1)  # [T,3,Hp,Wp]
    else:
        if codec == "hevc":
            rec_pad, total_bits = hevc_video_roundtrip(
                canv_pad, fps=fps, gop=gop, qp=qp, preset=preset, pix_fmt=pix_fmt
            )
        elif codec == "av1":
            rec_pad, total_bits = av1_video_roundtrip(
                canv_pad, fps=fps, gop=gop, qp=qp, cpu_used=cpu_used, pix_fmt=pix_fmt
            )
        elif codec == "vp9":
            rec_pad, total_bits = vp9_video_roundtrip(
                canv_pad, fps=fps, gop=gop, qp=qp, cpu_used=cpu_used, pix_fmt=pix_fmt
            )
        else:
            raise ValueError(f"Unsupported --codec {codec}. Use 'hevc', 'av1', or 'vp9'.")

    # Crop back to original packed canvas size (matches training before unpack)
    rec = rec_pad[..., :H, :W].contiguous()  # [T,3,H,W]

    # Save reconstructed canvases
    save_png_stack(out_dir, rec, names)

    # Metrics on canvases (diagnostic)
    mse = F.mse_loss(rec, canv)
    psnr = float(10.0 * torch.log10((1.0 ** 2) / (mse + 1e-12)))  # peak=1.0
    bpp  = float(total_bits) / float(T * Hp * Wp)

    return dict(T=T, H=H, W=W, Hp=Hp, Wp=Wp, total_bits=float(total_bits), bpp=bpp, psnr=psnr)


def main():
    p = argparse.ArgumentParser(description="Evaluate packed-plane video compression (HEVC/AV1/VP9) with QP, matching training.")
    p.add_argument('--segment-root', required=True,
                   help="Path to the segment folder containing xy_plane/xz_plane/yz_plane[/density].")
    p.add_argument('--codec', choices=['hevc', 'av1', 'vp9'], required=True)
    p.add_argument('--qp', type=int, required=True, help="Constant quantizer (QP).")
    p.add_argument('--gop', type=int, default=20)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument("--startframe", type=int, default=0, help="Start frame id (inclusive)")
    p.add_argument("--numframe", type=int, default=1,  help="Number of frames")
    p.add_argument("--packing_mode", choices=["flatten", "mosaic", "flat4", "separate", "grouped", "correlation", "flatfour"],
                   required=True, help="Packing mode used when generating the canvases. 'flatten' triggers grayscale path.")
    p.add_argument("--qmode", choices=["global", "per_channel"], required=True)
    p.add_argument('--preset', type=str, default='medium', help="HEVC x265 preset (ultrafast..placebo).")
    p.add_argument('--cpu-used', type=str, default='6', help="AV1/VP9 cpu-used (0..8; lower=slower/better).")
    p.add_argument('--pix-fmt', type=str, default='yuv444p', help="Prefer yuv444p for packed canvases.")
    p.add_argument('--align', type=int, default=32, help="Alignment used when packing (DCVC_ALIGN).")
    p.add_argument('--outdir', default=None, help="Where to write recon images & logs. Default: auto under segment root.")
    args = p.parse_args()

    seg = os.path.abspath(args.segment_root)
    if not os.path.isdir(seg):
        print(f"[ERR] segment-root not found: {seg}", file=sys.stderr)
        sys.exit(1)

    # Default output folder naming to mirror training knobs
    if args.outdir is None:
        out_root = os.path.join(seg, f"compressed_{args.codec}_qp{args.qp}_g{args.gop}_{args.pix_fmt}")
    else:
        out_root = os.path.abspath(args.outdir)
    os.makedirs(out_root, exist_ok=True)

    all_metrics = {}
    total_bits_sum = 0.0
    total_pixels_padded = 0
    S = args.startframe
    N = args.numframe

    STREAM_DIRS = ["xy_plane", "xz_plane", "yz_plane", "density"]
    for sub in STREAM_DIRS:
        in_dir  = os.path.join(seg, f"planeimg_{S:02d}_{N-1:02d}_{args.packing_mode}_flatten_{args.qmode}", sub)
        out_dir = os.path.join(out_root, sub)
        if not os.path.isdir(in_dir):
            raise FileNotFoundError(f"Missing input directory: {in_dir}")

        print(f"[{args.codec.upper()}] Compressing stream: {sub}")
        metrics = run_one_stream(
            in_dir=in_dir,
            out_dir=out_dir,
            codec=args.codec,
            fps=args.fps, gop=args.gop, qp=args.qp,
            preset=args.preset, cpu_used=args.cpu_used,
            pix_fmt=args.pix_fmt,
            align=args.align,
            packing_mode=args.packing_mode,
        )
        if metrics:
            all_metrics[sub] = metrics
            total_bits_sum += metrics['total_bits']
            total_pixels_padded += int(metrics['T'] * metrics['Hp'] * metrics['Wp'])
            print(f"  -> {sub}: bits={metrics['total_bits']:.0f}, bpp={metrics['bpp']:.6f}, PSNR={metrics['psnr']:.3f} dB")
        else:
            print(f"  -> {sub}: (no frames)")

    # encoded_bits.txt (compat with prior convention)
    bits_txt = os.path.join(out_root, "encoded_bits.txt")
    with open(bits_txt, "w") as f:
        for sub, m in all_metrics.items():
            f.write(f"{sub}: total_bits={int(m['total_bits'])}  bpp={m['bpp']:.8f}  T={m['T']}  Hp={m['Hp']}  Wp={m['Wp']}\n")
        if total_pixels_padded > 0:
            overall_bpp = total_bits_sum / float(total_pixels_padded)
            f.write(f"TOTAL: total_bits={int(total_bits_sum)}  bpp={overall_bpp:.8f}\n")
    print(f"[OK] wrote {bits_txt}")

    # metrics.txt with PSNRs (extra)
    met_txt = os.path.join(out_root, "metrics.txt")
    with open(met_txt, "w") as f:
        for sub, m in all_metrics.items():
            f.write(f"{sub}: PSNR={m['psnr']:.6f} dB  (T={m['T']}, HxW={m['H']}x{m['W']}, Hp x Wp={m['Hp']}x{m['Wp']})\n")
    print(f"[OK] wrote {met_txt}")

    print(f"[DONE] Output at: {out_root}")

if __name__ == "__main__":
    main()