#!/usr/bin/env python3
import os
import re
import argparse
from typing import List

import numpy as np
import cv2

"""
Usage:
python eval_jpeg_triplane_image_compress.py \
    --logdir logs/nerf_synthetic/lego_image \
        --startframe 0 --numframe 1 --qmode global \
        --plane_packing_mode flatten --grid_packing_mode flatten --qp 20
"""

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
    """Read image preserving bit-depth; returns np.ndarray (cv2 returns BGR for color)."""
    return cv2.imread(path, -1)

def to_uint8_from_u16(img16: np.ndarray) -> np.ndarray:
    """16-bit (0..65535) -> 8-bit (0..255) with rounding."""
    if img16 is None:
        raise RuntimeError("to_uint8_from_u16: input is None")
    if img16.dtype == np.uint8:
        return img16
    if img16.dtype != np.uint16:
        img = np.clip(img16, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return ((img.astype(np.uint32) + 128) // 257).astype(np.uint8)
    return ((img16.astype(np.uint32) + 128) // 257).astype(np.uint8)

def to_uint16_from_u8(img8: np.ndarray) -> np.ndarray:
    """8-bit (0..255) -> 16-bit (0..65535) by scale 257."""
    if img8.dtype != np.uint8:
        img8 = img8.astype(np.uint8)
    return (img8.astype(np.uint16) * 257).astype(np.uint16)

def jpeg_roundtrip(img8_bgr: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    enc_ok, buf = cv2.imencode(".jpg", img8_bgr, params)
    if not enc_ok:
        raise RuntimeError("JPEG encoding failed.")
    encoded_bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)  # BGR8 or 1ch
    if dec is None:
        raise RuntimeError("JPEG decoding failed.")
    if dec.ndim == 2:
        dec = cv2.cvtColor(dec, cv2.COLOR_GRAY2BGR)
    return dec, encoded_bits

# ------------------------------- Core logic ----------------------------------

def compress_one_image(src_png: str, dst_png: str, qp: int) -> int:
    """
    16b PNG on disk (standard RGB) -> cv2.imread gives BGR16 in memory
      -> 16->8 (BGR) -> JPEG round-trip (BGR8) -> 8->16 (BGR)
      -> cv2.imwrite (BGR16) -> RGB PNG on disk.
    """
    if not os.path.isfile(src_png):
        raise FileNotFoundError(src_png)
    img16_bgr = imread_any(src_png)
    if img16_bgr is None:
        raise RuntimeError(f"Failed to read image: {src_png}")

    img8_bgr = to_uint8_from_u16(img16_bgr)
    dec8_bgr, encoded_bits = jpeg_roundtrip(img8_bgr, quality=qp)
    out16_bgr = to_uint16_from_u8(dec8_bgr)

    ensure_dir(os.path.dirname(dst_png))
    if not cv2.imwrite(dst_png, out16_bgr):
        raise RuntimeError(f"Failed to write: {dst_png}")
    return encoded_bits

def process_planes(dec_root: str, out_root: str, plane_names: List[str], fid: int, qp: int) -> int:
    total_bits = 0
    for plane in plane_names:
        src_img = os.path.join(dec_root, plane, f"im{fid + 1:05d}.png")
        dst_img = os.path.join(out_root, plane, f"im{fid + 1:05d}.png")
        total_bits += compress_one_image(src_img, dst_img, qp=qp)
    return total_bits

def process_density(dec_root: str, out_root: str, fid: int, qp: int) -> int:
    dens_src = os.path.join(dec_root, "density", f"im{fid + 1:05d}.png")
    dens_dst = os.path.join(out_root, "density", f"im{fid + 1:05d}.png")
    return compress_one_image(dens_src, dens_dst, qp=qp)

# ---------------------------------- Main -------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--logdir", required=True, help="Root of checkpoints & planeimg_*")
    p.add_argument("--startframe", type=int, default=0, help="Start frame id (inclusive)")
    p.add_argument("--numframe", type=int, default=1, help="Number of frames")
    p.add_argument("--plane_packing_mode", choices=["flatten", "mosaic", "flat4"], required=True,
                   help="Must match the packer/unpacker plane mode")
    p.add_argument("--grid_packing_mode", choices=["flatten", "mosaic", "flat4"], required=True,
                   help="Must match the packer/unpacker grid mode")
    p.add_argument("--qmode", choices=["global", "per_channel"], required=True)
    p.add_argument("--qp", type=int, default=40, help="JPEG quality [1..100]; higher = better")
    return p.parse_args()

def main():
    args = parse_args()
    S = args.startframe
    N = args.startframe + args.numframe - 1

    # Input folder written by packer:
    dec_root = os.path.join(
        args.logdir,
        f"planeimg_{S:02d}_{N:02d}_{args.plane_packing_mode}_{args.grid_packing_mode}_{args.qmode}"
    )
    if not os.path.isdir(dec_root):
        raise FileNotFoundError(f"Packed planes not found: {dec_root}")

    # Output folder for JPEG-decoded PNGs:
    out_root = os.path.join(
        args.logdir,
        f"planeimg_{S:02d}_{N:02d}_{args.plane_packing_mode}_{args.grid_packing_mode}_{args.qmode}_jpeg_qp{args.qp}"
    )
    os.makedirs(out_root, exist_ok=True)

    # Plane subdirs (e.g., 'xy','xz','yz'); exclude 'density' and pre-existing *_qp* dirs
    all_subs = list_subdirs(dec_root)
    plane_names = [d for d in all_subs if d != "density" and not re.fullmatch(r".*_qp\d+", d)]

    grand_total_bits = 0
    for fid in range(args.startframe, args.startframe + args.numframe):
        grand_total_bits += process_planes(dec_root, out_root, plane_names, fid, args.qp)
        grand_total_bits += process_density(dec_root, out_root, fid, args.qp)

    with open(os.path.join(out_root, "encoded_bits.txt"), "w") as f:
        f.write(str(int(grand_total_bits)) + "\n")

    print("[DONE] JPEG-compressed & decoded PNGs written to:", out_root)
    print(f"[INFO] Total encoded size = {grand_total_bits} bits")

if __name__ == "__main__":
    main()