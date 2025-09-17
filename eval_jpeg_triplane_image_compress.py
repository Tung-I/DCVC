#!/usr/bin/env python
"""
Apply JPEG compression to packed TriPlane images and write back 16-bit PNGs
exactly where `image2plane.py` expects them (non-DCVC path).

Directory conventions (matching your packer + unpacker):
- Base packed root: {root}/planeimg_{S:02d}_{N:02d}_{packing_mode}_{qmode}
- We read original 16b PNGs from the base structure and write decoded PNGs to
  sibling folders suffixed with `_qp{qp}` and filenames `..._decoded.png`.

Supported strategies:
  flatten      : {root}/{plane}/imXXXXX.png              -> {plane}_qp{qp}/imXXXXX_decoded.png
  separate    : {root}/{plane}/c{c}/imXXXXX.png         -> {plane}/c{c}_qp{qp}/imXXXXX_decoded.png
  grouped     : {root}/{plane}/stream{g}/imXXXXX.png    -> {plane}/stream{g}_qp{qp}/imXXXXX_decoded.png
  correlation : same as grouped (uses stored streams)
  flatfour    : {root}/{plane}/imXXXXX.png              -> {plane}_qp{qp}/imXXXXX_decoded.png
  density     : {root}/density/imXXXXX.png              -> density_qp{qp}/imXXXXX_decoded.png

Notes:
- JPEG is 8-bit; we down-convert 16-bit PNG -> 8-bit for JPEG, then up-convert
  decoded JPEG back to 16-bit PNG by multiplying by 257, so that downstream
  de-quantization in `image2plane.py` remains consistent.
- `qp` here maps to OpenCV's IMWRITE_JPEG_QUALITY in [1..100] (higher = better).

Example:
    python jpeg_triplane_image_compress.py --logdir logs/out_triplane/flame_steak_image --startframe 0 --numframe 1 --packing_mode flatten --qmode global --qp 4
"""

import os
import re
import argparse
from typing import List, Tuple

import numpy as np
import cv2

# ------------------------------- I/O helpers ---------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_subdirs(p: str, pat: str = None) -> List[str]:
    if not os.path.isdir(p):
        return []
    subs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    if pat:
        rx = re.compile(pat)
        subs = [d for d in subs if rx.fullmatch(d)]
    return sorted(subs, key=lambda x: (len(x), x))

def imread_any(path: str):
    """Read image preserving bit-depth; returns np.ndarray."""
    return cv2.imread(path, -1)

def to_uint8_from_u16(img16: np.ndarray) -> np.ndarray:
    """
    16-bit (0..65535) -> 8-bit (0..255) with rounding.
    For grayscale HxW and color HxWx3.
    """
    if img16.dtype != np.uint16:
        # Accept 8-bit sources gracefully (shouldn't happen in your pipeline)
        return img16.astype(np.uint8)
    # Round: val8 = round(val16 / 257)
    return ((img16.astype(np.uint32) + 128) // 257).astype(np.uint8)

def to_uint16_from_u8(img8: np.ndarray) -> np.ndarray:
    """
    8-bit (0..255) -> 16-bit (0..65535) by scale 257.
    """
    if img8.dtype != np.uint8:
        img8 = img8.astype(np.uint8)
    # val16 = val8 * 257
    return (img8.astype(np.uint16) * 257).astype(np.uint16)

def jpeg_roundtrip(img8: np.ndarray, quality: int, is_color: bool) -> np.ndarray:
    """
    Encode to JPEG in-memory then decode back (returns 8-bit array).
    """
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    # Ensure proper shape for OpenCV
    if not is_color and img8.ndim == 2:
        enc_ok, buf = cv2.imencode(".jpg", img8, params)
    else:
        # Guarantee a 3-channel array for color path
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        enc_ok, buf = cv2.imencode(".jpg", img8, params)
    if not enc_ok:
        raise RuntimeError("JPEG encoding failed.")
    
    encoded_bits = int(buf.size) * 8

    dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if dec is None:
        raise RuntimeError("JPEG decoding failed.")
    # If grayscale desired but decoder returned 3ch, convert back
    if not is_color and dec.ndim == 3 and dec.shape[2] == 3:
        dec = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY)
    return dec, encoded_bits

# ------------------------------- Core logic ----------------------------------

def compress_one_image(src_png: str, dst_png: str, is_color: bool, qp: int, keep_3ch=True):
    """16b PNG -> JPEG(qp) -> 16b PNG (decoded)."""
    if not os.path.isfile(src_png):
        raise FileNotFoundError(src_png)
    img16 = imread_any(src_png)
    if img16 is None:
        raise RuntimeError(f"Failed to read image: {src_png}")

    # Convert 16b -> 8b
    img8 = to_uint8_from_u16(img16)

    # JPEG round-trip
    dec8, encoded_bits = jpeg_roundtrip(img8, quality=qp, is_color=is_color)

    # Convert back to 16-bit
    out16 = to_uint16_from_u8(dec8)

    # if keep_3ch, check if out16 is one-channel; if so, replicate the channel
    if keep_3ch and out16.ndim == 2:
        out16 = np.stack([out16] * 3, axis=-1)

    if keep_3ch:
        assert out16.ndim == 3 and out16.shape[2] == 3, f"Expected 3-channel output, got shape {out16.shape}"    

    ensure_dir(os.path.dirname(dst_png))
    if not cv2.imwrite(dst_png, out16):
        raise RuntimeError(f"Failed to write: {dst_png}")
    
    return encoded_bits

def process_flatten(dec_root: str, out_root: str, plane_names: List[str], fid: int, qp: int, color: bool):
    """
    flatten  : per-plane mono image
    flatfour: per-plane color image
    """
    total_bits = 0
    for plane in plane_names:
        src_dir = os.path.join(dec_root, plane)
        src_img = os.path.join(src_dir, f"im{fid + 1:05d}.png")
        dst_dir = os.path.join(out_root, f"{plane}")
        dst_img = os.path.join(dst_dir, f"im{fid + 1:05d}.png")
        encoded_bits = compress_one_image(src_img, dst_img, is_color=color, qp=qp)

        total_bits += encoded_bits
    return total_bits

def process_separate(dec_root: str, plane_names: List[str], fid: int, qp: int):
    """
    separate: per-plane, per-channel mono images in c{c} subdirs
    """
    for plane in plane_names:
        base = os.path.join(dec_root, plane)
        ch_dirs = list_subdirs(base, r"c\d+")
        for chdir in ch_dirs:
            src_dir = os.path.join(base, chdir)
            src_img = os.path.join(src_dir, f"im{fid + 1:05d}.png")
            dst_dir = os.path.join(base, f"{chdir}_qp{qp}")
            dst_img = os.path.join(dst_dir, f"im{fid + 1:05d}_decoded.png")
            compress_one_image(src_img, dst_img, is_color=False, qp=qp)

def process_grouped_like(dec_root: str, plane_names: List[str], fid: int, qp: int):
    """
    grouped/correlation: per-plane, per-stream color images in stream{g} subdirs
    """
    for plane in plane_names:
        base = os.path.join(dec_root, plane)
        streams = list_subdirs(base, r"stream\d+")
        for sdir in streams:
            src_dir = os.path.join(base, sdir)
            src_img = os.path.join(src_dir, f"im{fid + 1:05d}.png")
            dst_dir = os.path.join(base, f"{sdir}_qp{qp}")
            dst_img = os.path.join(dst_dir, f"im{fid + 1:05d}_decoded.png")
            compress_one_image(src_img, dst_img, is_color=True, qp=qp)

def process_density(dec_root: str, out_root: str, fid: int, qp: int, is_color=False):
    dens_src_dir = os.path.join(dec_root, "density")
    if not os.path.isdir(dens_src_dir):
        return  # allow absence
    src_img = os.path.join(dens_src_dir, f"im{fid + 1:05d}.png")
    dst_dir = os.path.join(out_root, f"density")
    dst_img = os.path.join(dst_dir, f"im{fid + 1:05d}.png")
    encoded_bits = compress_one_image(src_img, dst_img, is_color=is_color, qp=qp)

    return encoded_bits

# ---------------------------------- Main -------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--logdir", required=True, help="Root of checkpoints & planeimg_*")
    p.add_argument("--startframe", type=int, default=0, help="Start frame id (inclusive)")
    p.add_argument("--numframe", type=int, default=1, help="Number of frames")
    p.add_argument("--packing_mode", choices=["flatten", "separate", "grouped", "correlation", "flatfour"], required=True)
    p.add_argument("--qmode", choices=["global", "per_channel"], required=True)
    p.add_argument("--qp", type=int, default=40, help="JPEG quality [1..100]; higher = better")
    return p.parse_args()

def main():
    args = parse_args()

    S = args.startframe
    N = args.startframe + args.numframe - 1
    dec_root = os.path.join(args.logdir, f"planeimg_{S:02d}_{N:02d}_{args.packing_mode}_{args.qmode}")
    out_root = os.path.join(args.logdir, f"planeimg_{S:02d}_{N:02d}_{args.packing_mode}_{args.qmode}_jpeg_qp{args.qp}")
    os.makedirs(out_root, exist_ok=True)
    if not os.path.isdir(dec_root):
        raise FileNotFoundError(f"Packed planes not found: {dec_root}")

    # Planes are subdirs in dec_root (typically 'xy','xz','yz'), exclude 'density' and already-compressed *_qp* dirs
    all_subs = list_subdirs(dec_root)
    plane_names = [d for d in all_subs if d != "density" and not re.fullmatch(r".*_qp\d+", d)]

    grand_total_bits = 0
    for fid in range(args.startframe, args.startframe + args.numframe):
        if args.packing_mode == "flatten":
            grand_total_bits += process_flatten(dec_root, out_root, plane_names, fid, args.qp, color=False) # Keep 3 channels for unpacking
        elif args.packing_mode == "flatfour":
            process_flatten(dec_root, plane_names, fid, args.qp, color=True)
        elif args.packing_mode == "separate":
            process_separate(dec_root, plane_names, fid, args.qp)
        elif args.packing_mode in ("grouped", "correlation"):
            process_grouped_like(dec_root, plane_names, fid, args.qp)
        else:
            raise ValueError("Unknown packing_mode")

        # density is written by your packer regardless of packing_mode; compress it too
        grand_total_bits += process_density(dec_root, out_root, fid, args.qp, is_color=False)  # keep 1 channel for unpacking

    # Write a single integer to encoded_bits.txt (total bits across all JPEGs)
    bits_path = os.path.join(out_root, "encoded_bits.txt")
    with open(bits_path, "w") as f:
        f.write(str(int(grand_total_bits)) + "\n")

    print("[DONE] JPEG-compressed & decoded PNGs ready under:", dec_root)
    print(f"[INFO] Total encoded size = {grand_total_bits} bits")

if __name__ == "__main__":
    main()
