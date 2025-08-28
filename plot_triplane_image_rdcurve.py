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

"""
Usage:
    python plot_triplane_image_rdcurve.py --root logs/out_triplane/flame_steak_image --annotate
"""

import os
import re
import glob
import csv
import json
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Configure your DCVC sets here
#   name -> { label -> folder }
# Each 'folder' must contain:
#   - compress_stats_f*_q*.json  (with total_bits or planes+density bits)
#   - render_test/0_psnr.txt
# ==============================
set1: Dict[str, str] = {
    'qp48': 'logs/out_triplane/flame_steak_image_dcvc_qp48_dens/dcvc_qp48',
    'qp36': 'logs/out_triplane/flame_steak_image_dcvc_qp36_dens/dcvc_qp36',
    'qp24': 'logs/out_triplane/flame_steak_image_dcvc_qp24_dens/dcvc_qp24',
    'qp12': 'logs/out_triplane/flame_steak_image_dcvc_qp12_dens/dcvc_qp12',
    'qp0' : 'logs/out_triplane/flame_steak_image_dcvc_qp0_dens/dcvc_qp0'
}
# Example extra set (leave empty or add later)
set2: Dict[str, str] = {
    'qp48': 'logs/out_triplane/flame_steak_image/dcvc_qp48',
    'qp36': 'logs/out_triplane/flame_steak_image/dcvc_qp36',
    'qp24': 'logs/out_triplane/flame_steak_image/dcvc_qp24',
    'qp12': 'logs/out_triplane/flame_steak_image/dcvc_qp12',
    'qp0' : 'logs/out_triplane/flame_steak_image/dcvc_qp0'
}

SETS: Dict[str, Dict[str, str]] = {
    "Neural Image Codec + Our Codec-Aware Training": set1,
    "Neural Image Codec": set2,
}

# ==============================
# JPEG baseline folder matcher
# ==============================
FOLDER_RX = re.compile(r"^planeimg_\d{2}_\d{2}_[a-z]+_[a-z_]+_qp(\d+)$")
QP_LABEL_RX = re.compile(r".*?qp(\d+)", re.IGNORECASE)

def read_float(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return None

def save_csv(path: str, rows: List[Tuple]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # infer header by row length
        if rows and len(rows[0]) == 4:
            w.writerow(["bitrate_Mbit", "psnr_dB", "qp", "folder_or_label"])
        elif rows and len(rows[0]) == 3:
            w.writerow(["bitrate_Mbit", "psnr_dB", "qp"])
        else:
            w.writerow(["bitrate_Mbit", "psnr_dB"])
        for r in rows:
            w.writerow(r)

# -------------------------------
# JPEG baseline (existing layout)
# -------------------------------
def collect_jpeg_results(root: str) -> List[Tuple[float, float, Optional[int], str]]:
    """
    Scan subfolders under root that match FOLDER_RX.
    For each folder, read:
      - encoded_bits.txt  (total bits over all planes)
      - render_test/0_psnr.txt
    Returns list of (bitrate_Mbit, psnr_dB, qp, folder_name)
    """
    points = []
    for name in sorted(os.listdir(root)):
        m = FOLDER_RX.match(name)
        if not m:
            continue
        qp = int(m.group(1))
        folder = os.path.join(root, name)
        bits = read_float(os.path.join(folder, "encoded_bits.txt"))
        psnr = read_float(os.path.join(folder, "render_test", "0_psnr.txt"))
        if bits is None or psnr is None:
            continue
        bitrate_mbit = bits / 1e6
        points.append((bitrate_mbit, psnr, qp, name))

    # Order from lower QP to higher QP; if two QPs equal, order by bitrate ascending
    points.sort(key=lambda p: (p[2] if p[2] is not None else 10**9, p[0]))
    return points

# -------------------------------
# DCVC sets (stats JSON layout)
# -------------------------------
def _read_dcvc_bits_from_json(folder: str) -> Optional[float]:
    """
    Find 'compress_stats_f*_q*.json' in folder and return total bits.
    If multiple files exist, pick the first after sorting by name.
    """
    cand = sorted(glob.glob(os.path.join(folder, "compress_stats_f*_q*.json")))
    if not cand:
        return None
    path = cand[0]
    try:
        with open(path, "r") as f:
            j = json.load(f)
        if "total_bits" in j:
            return float(j["total_bits"])
        if "planes" in j and "density" in j:
            pb = sum(float(j["planes"][ax]["bits"]) for ax in ("xy", "xz", "yz") if ax in j["planes"])
            db = float(j["density"]["bits"])
            return pb + db
    except Exception:
        return None
    return None

def _parse_qp_from_label(label: str) -> Optional[int]:
    m = QP_LABEL_RX.match(label)
    return int(m.group(1)) if m else None

def collect_dcvc_set(paths: Dict[str, str]) -> List[Tuple[float, float, Optional[int], str]]:
    """
    For each (label -> folder):
      - total bits from compress_stats_f*_q*.json
      - PSNR from render_test/0_psnr.txt
      - QP parsed from label if possible (qpNN)
    Returns list of (bitrate_Mbit, psnr_dB, qp, label)
    """
    points = []
    for label, folder in paths.items():
        bits = _read_dcvc_bits_from_json(folder)
        psnr = read_float(os.path.join(folder, "render_test", "0_psnr.txt"))
        if bits is None or psnr is None:
            print(f"[warn] skip {label}: bits={bits} psnr={psnr} (folder={folder})")
            continue
        bitrate_mbit = bits / 1e6
        qp = _parse_qp_from_label(label)
        points.append((bitrate_mbit, psnr, qp, label))

    # Prefer ordering by QP (ascending), fallback to bitrate
    points.sort(key=lambda p: (p[2] if p[2] is not None else 10**9, p[0]))
    return points

# -------------------------------
# CLI / Plot
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", required=True, help="Dir containing JPEG-style baseline folders (planeimg_*_qpXX)")
    ap.add_argument("--out", default="/home/tungichen_umass_edu/DCVC/plots/rd_curve.png", help="Output plot filename (absolute or relative to --root)")
    ap.add_argument("--csv", default="/home/tungichen_umass_edu/DCVC/plots/rd_points_jpeg.csv", help="CSV for JPEG points ('' to skip)")
    ap.add_argument("--annotate", action="store_true", help="Annotate points with QP/labels")
    return ap.parse_args()

def main():
    args = parse_args()
    root = args.root

    # ---- Collect baseline (JPEG) ----
    jpeg_points = collect_jpeg_results(root)
    if not jpeg_points:
        raise SystemExit("No valid JPEG points under --root. "
                         "Check folder names and presence of encoded_bits.txt & render_test/0_psnr.txt.")

    # ---- Collect DCVC sets ----
    curves = []  # list of (name, points[(bitrate, psnr, qp, label)])
    for set_name, mapping in SETS.items():
        pts = collect_dcvc_set(mapping)
        if pts:
            curves.append((set_name, pts))

    # ---- Prepare plot data ----
    x_j = np.array([p[0] for p in jpeg_points], dtype=float)
    y_j = np.array([p[1] for p in jpeg_points], dtype=float)
    qps_j = [p[2] for p in jpeg_points]
    names_j = [p[3] for p in jpeg_points]

    # Resolve output paths
    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    csv_path = (args.csv if (args.csv and os.path.isabs(args.csv)) else
                (os.path.join(root, args.csv) if args.csv else None))

    # ---- Plot ----
    plt.figure(figsize=(7.5, 5.0))

    # JPEG baseline
    plt.plot(x_j, y_j, marker="o", linestyle="-", label="JPEG Compression")

    # DCVC curves
    markers = ["s", "^", "v", "D", "P", "X", "*", "<", ">"]
    for idx, (set_name, pts) in enumerate(curves):
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        plt.plot(x, y, marker=markers[idx % len(markers)], linestyle="-", label=set_name)

    plt.xlabel("Bitrate (Mbit)")
    plt.ylabel("PSNR (dB)")
    plt.title("Rendering Quality (I-frame, one TriPlane)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.65)
    plt.legend()

    if args.annotate:
        # Annotate JPEG points: QP if present
        for xi, yi, qp, name in zip(x_j, y_j, qps_j, names_j):
            label = f"QP{qp}" if qp is not None else name
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

        # Annotate DCVC points with label (often 'qpXX')
        for _, pts in curves:
            for (xi, yi, qp, label) in pts:
                tag = f"QP{qp}" if qp is not None else label
                plt.annotate(tag, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Natural axis directions: left->right increasing bitrate; bottom->top increasing PSNR
    # (No invert_xaxis call here.)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

    # ---- CSV outputs ----
    if csv_path:
        save_csv(csv_path, jpeg_points)
    # One CSV per DCVC set
    for set_name, pts in curves:
        rows = [(p[0], p[1], p[2] if p[2] is not None else "") for p in pts]
        csv_name = f"plots/rd_points_{re.sub(r'[^a-zA-Z0-9]+','_',set_name.lower())}.csv"
        csv_full = csv_name if os.path.isabs(csv_name) else os.path.join(root, csv_name)
        save_csv(csv_full, rows)

    print(f"[DONE] Saved RD plot: {out_path}")
    if csv_path:
        print(f"[DONE] Saved JPEG CSV: {csv_path}")
    for set_name, _ in curves:
        csv_name = f"plots/rd_points_{re.sub(r'[^a-zA-Z0-9]+','_',set_name.lower())}.csv"
        csv_full = csv_name if os.path.isabs(csv_name) else os.path.join(root, csv_name)
        print(f"[DONE] Saved DCVC CSV ({set_name}): {csv_full}")

    print("\nJPEG Points (ordered by QP asc):")
    for (br, psnr, qp, name) in jpeg_points:
        print(f"  {name:>40s}  bitrate={br:.3f} Mbit  PSNR={psnr:.3f} dB  QP={qp}")

    for set_name, pts in curves:
        print(f"\n{set_name} Points (ordered by QP asc):")
        for (br, psnr, qp, label) in pts:
            tag = f"QP{qp}" if qp is not None else label
            print(f"  {tag:>8s}  bitrate={br:.3f} Mbit  PSNR={psnr:.3f} dB")

if __name__ == "__main__":
    main()

