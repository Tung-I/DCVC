#!/usr/bin/env python3
"""
Plot an RD curve (PSNR vs. bitrate in Mbit) for JPEG-compressed packed TriPlanes.

Expected layout under --root (example):
logs/out_triplane/flame_steak_image/
  triplanes_00_00_qp10_tiling_global/
    encoded_bits.txt
    render_test/0_psnr.txt
  triplanes_00_00_qp20_tiling_global/
    ...

Usage:
  python plot_rd_triplane.py \
    --root logs/out_triplane/flame_steak_image \
    --out rd_curve.png --csv rd_points.csv --annotate

Notes:
- Bitrate unit: Mbit (bits / 1e6).
- Points are sorted on x-axis by bitrate.
"""

import os
import re
import argparse
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import csv

FOLDER_RX = re.compile(r"^triplanes_\d{2}_\d{2}_qp(\d+)_([a-z]+)_([a-z_]+)$")

def read_float(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        s = f.read().strip()
        try:
            return float(s)
        except ValueError:
            return None

def collect_results(root: str) -> List[Tuple[float, float, Optional[int], str]]:
    """
    Returns list of (bitrate_mbit, psnr_db, qp_int_or_None, folder_name)
    Only includes folders where both files exist and are parseable.
    """
    points = []
    list_dir = os.listdir(root)
    # reverse the list_dir
    list_dir.reverse()
    print(list_dir)
    for name in sorted(list_dir):
        m = FOLDER_RX.match(name)
        if not m:
            continue
        qp = int(m.group(1)) if m else None
        folder = os.path.join(root, name)

        bits_path = os.path.join(folder, "encoded_bits.txt")
        psnr_path = os.path.join(folder, "render_test", "0_psnr.txt")

        bits = read_float(bits_path)
        psnr = read_float(psnr_path)
        if bits is None or psnr is None:
            continue

        bitrate_mbit = bits / 1e6  # bits -> Mbit
        points.append((bitrate_mbit, psnr, qp, name))
    # Sort by bitrate (decreasing)
    points.sort(key=lambda x: x[0], reverse=True)
    # points.sort(key=lambda x: x[0])
    return points

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", required=True, help="Directory containing triplanes_* result folders")
    ap.add_argument("--out", default="rd_curve.png", help="Output plot image filename (saved under --root if relative)")
    ap.add_argument("--csv", default="rd_points.csv", help="Output CSV of collected points (saved under --root if relative). Use '' to skip.")
    ap.add_argument("--annotate", action="store_true", help="Annotate points with QP")
    return ap.parse_args()

def save_csv(path: str, rows: List[Tuple[float, float, Optional[int], str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitrate_Mbit", "psnr_dB", "qp", "folder"])
        for r in rows:
            w.writerow(r)

def main():
    args = parse_args()
    root = args.root

    points = collect_results(root)
    if not points:
        raise SystemExit("No valid points found. Check folder names and presence of encoded_bits.txt / render_test/0_psnr.txt.")

    x = np.array([p[0] for p in points], dtype=float)  # bitrate in Mbit
    y = np.array([p[1] for p in points], dtype=float)  # PSNR in dB
    qps = [p[2] for p in points]
    names = [p[3] for p in points]

    # Resolve output paths
    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    csv_path = (args.csv if os.path.isabs(args.csv) else os.path.join(root, args.csv)) if args.csv else None

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("Bitrate (Mbit)")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate–Distortion (JPEG on packed TriPlane)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if args.annotate:
        for xi, yi, qp, name in zip(x, y, qps, names):
            label = f"QP{qp}" if qp is not None else name
            # Offset slightly so text doesn't overlap the marker
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    if csv_path:
        save_csv(csv_path, points)

    print(f"[DONE] Saved RD plot: {out_path}")
    if csv_path:
        print(f"[DONE] Saved CSV: {csv_path}")
    print("Points:")
    for (br, psnr, qp, name) in points:
        print(f"  {name:>40s}  bitrate={br:.3f} Mbit  PSNR={psnr:.3f} dB  (QP={qp})")

if __name__ == "__main__":
    main()
