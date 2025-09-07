import os
import re
import glob
import csv
import json
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Your DCVC sets (explicit mapping)
# ==============================
set1: Dict[str, str] = {
    'qp48': 'logs/out_triplane/flame_steak_image/dcvc_qp48',
    'qp36': 'logs/out_triplane/flame_steak_image/dcvc_qp36',
    'qp24': 'logs/out_triplane/flame_steak_image/dcvc_qp24',
    'qp12': 'logs/out_triplane/flame_steak_image/dcvc_qp12',
    'qp0' : 'logs/out_triplane/flame_steak_image/dcvc_qp0'
}
set2: Dict[str, str] = {
    'qp48': 'logs/out_triplane/flame_steak_image_dcvc_qp48_dens/dcvc_qp48',
    'qp36': 'logs/out_triplane/flame_steak_image_dcvc_qp36_dens/dcvc_qp36',
    'qp24': 'logs/out_triplane/flame_steak_image_dcvc_qp24_dens/dcvc_qp24',
    'qp12': 'logs/out_triplane/flame_steak_image_dcvc_qp12_dens/dcvc_qp12',
    'qp0' : 'logs/out_triplane/flame_steak_image_dcvc_qp0_dens/dcvc_qp0'
}
set3: Dict[str, str] = {
    'qp80': 'logs/out_triplane/flame_steak_image/planeimg_00_00_flatten_global_qp80',
    'qp60': 'logs/out_triplane/flame_steak_image/planeimg_00_00_flatten_global_qp60',
    'qp40': 'logs/out_triplane/flame_steak_image/planeimg_00_00_flatten_global_qp40',
    'qp20': 'logs/out_triplane/flame_steak_image/planeimg_00_00_flatten_global_qp20',
    'qp10': 'logs/out_triplane/flame_steak_image/planeimg_00_00_flatten_global_qp10'
}
set4: Dict[str, str] = {
    'qp80': 'logs/out_triplane/flame_steak_image_jpeg_qp80/planeimg_00_00_flatten_global_qp80',
    'qp60': 'logs/out_triplane/flame_steak_image_jpeg_qp60/planeimg_00_00_flatten_global_qp60',
    'qp40': 'logs/out_triplane/flame_steak_image_jpeg_qp40/planeimg_00_00_flatten_global_qp40',
    'qp20': 'logs/out_triplane/flame_steak_image_jpeg_qp20/planeimg_00_00_flatten_global_qp20',
    'qp10': 'logs/out_triplane/flame_steak_image_jpeg_qp10/planeimg_00_00_flatten_global_qp10'
}

# ==============================
# Generic readers
# ==============================
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
        if rows and len(rows[0]) == 4:
            w.writerow(["bitrate_Mbit", "psnr_dB", "qp", "label_or_folder"])
        elif rows and len(rows[0]) == 3:
            w.writerow(["bitrate_Mbit", "psnr_dB", "qp"])
        else:
            w.writerow(["bitrate_Mbit", "psnr_dB"])
        for r in rows:
            w.writerow(r)

def _read_bits_encoded_txt(folder: str, rel_path: str = "encoded_bits.txt") -> Optional[float]:
    return read_float(os.path.join(folder, rel_path))

def _read_bits_dcvc_json(folder: str, pattern: str = "compress_stats_f*_q*.json") -> Optional[float]:
    cand = sorted(glob.glob(os.path.join(folder, pattern)))
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

def _read_psnr_file(folder: str, rel_path: str = "render_test/0_psnr.txt") -> Optional[float]:
    return read_float(os.path.join(folder, rel_path))

def _parse_qp(text: str, rx: Optional[str]) -> Optional[int]:
    if not rx:
        return None
    m = re.match(rx, text) if rx.startswith("^") else re.search(rx, text)
    return int(m.group(1)) if m else None

# ==============================
# Curve specs (declarative)
# ==============================
# You can add more curves by appending a new dict to CURVES.
# - kind = "folder_scan" (scan subdirs) or "mapping" (label->folder)
# - bits.type ∈ {"encoded_bits_txt", "dcvc_json"}
# - psnr.type == "file" (path relative to folder)
# - qp_regex is used to parse QP from folder name (folder_scan) or from label (mapping)
#
# For JPEG baseline, we'll pass --root_jpeg at CLI and use it here.
CURVES: List[Dict[str, Any]] = [
    # {
    #     "name": "JPEG Compression",
    #     "enabled": True,
    #     "kind": "folder_scan",
    #     # root is provided by CLI: --root_jpeg
    #     "root_from_cli": "root_jpeg",
    #     "folder_regex": r"^planeimg_\d{2}_\d{2}_[a-z]+_[a-z_]+_qp(\d+)$",
    #     "qp_regex": r".*?qp(\d+)",
    #     "bits": {"type": "encoded_bits_txt", "path": "encoded_bits.txt"},
    #     "psnr": {"type": "file", "path": "render_test/0_psnr.txt"},
    # },
    {
        "name": "JPEG",
        "enabled": True,
        "kind": "mapping",
        "paths": set3,  
        "qp_regex": r".*?qp(\d+)",
        "bits": {"type": "encoded_bits_txt", "path": "encoded_bits.txt"},
        "psnr": {"type": "file", "path": "render_test/0_psnr.txt"},
    },
    {
        "name": "JPEG + Our STE Training",
        "enabled": True,
        "kind": "mapping",
        "paths": set4, 
        "qp_regex": r".*?qp(\d+)",
        "bits": {"type": "encoded_bits_txt", "path": "encoded_bits.txt"},
        "psnr": {"type": "file", "path": "render_test/0_psnr.txt"},
    },
    {
        "name": "DCVC",
        "enabled": True,
        "kind": "mapping",
        "paths": set1,
        "qp_regex": r".*?qp(\d+)",
        "bits": {"type": "dcvc_json", "pattern": "compress_stats_f*_q*.json"},
        "psnr": {"type": "file", "path": "render_test/0_psnr.txt"},
    },
    {
        "name": "DCVC + Our STE Training",
        "enabled": True,
        "kind": "mapping",
        "paths": set2,  # label -> folder
        "qp_regex": r".*?qp(\d+)",  # parse QP from label like "qp36"
        "bits": {"type": "dcvc_json", "pattern": "compress_stats_f*_q*.json"},
        "psnr": {"type": "file", "path": "render_test/0_psnr.txt"},
    },
]

# ==============================
# Collectors
# ==============================
def _read_bits(bits_spec: Dict[str, Any], folder: str) -> Optional[float]:
    btype = bits_spec.get("type")
    if btype == "encoded_bits_txt":
        return _read_bits_encoded_txt(folder, bits_spec.get("path", "encoded_bits.txt"))
    if btype == "dcvc_json":
        return _read_bits_dcvc_json(folder, bits_spec.get("pattern", "compress_stats_f*_q*.json"))
    raise ValueError(f"Unknown bits.type: {btype}")

def _read_psnr(psnr_spec: Dict[str, Any], folder: str) -> Optional[float]:
    ptype = psnr_spec.get("type")
    if ptype == "file":
        return _read_psnr_file(folder, psnr_spec.get("path", "render_test/0_psnr.txt"))
    raise ValueError(f"Unknown psnr.type: {ptype}")

def collect_folder_scan(curve: Dict[str, Any], root_dir: str) -> List[Tuple[float, float, Optional[int], str]]:
    folder_rx = re.compile(curve["folder_regex"])
    qp_rx = curve.get("qp_regex")  # optional
    points = []
    if not os.path.isdir(root_dir):
        print(f"[warn] folder_scan root does not exist: {root_dir}")
        return points

    for name in sorted(os.listdir(root_dir)):
        m = folder_rx.match(name)
        if not m:
            continue
        qp = int(m.group(1)) if m.groups() else _parse_qp(name, qp_rx)
        folder = os.path.join(root_dir, name)

        bits = _read_bits(curve["bits"], folder)
        psnr = _read_psnr(curve["psnr"], folder)
        if bits is None or psnr is None:
            print(f"[warn] skip {name}: bits={bits} psnr={psnr} (folder={folder})")
            continue

        bitrate_mbit = bits / 1e6
        points.append((bitrate_mbit, psnr, qp, name))

    # Order by QP, then bitrate
    points.sort(key=lambda p: (p[2] if p[2] is not None else 10**9, p[0]))
    return points

def collect_mapping(curve: Dict[str, Any]) -> List[Tuple[float, float, Optional[int], str]]:
    paths: Dict[str, str] = curve["paths"]
    qp_rx = curve.get("qp_regex")
    points = []
    for label, folder in paths.items():
        bits = _read_bits(curve["bits"], folder)
        psnr = _read_psnr(curve["psnr"], folder)
        if bits is None or psnr is None:
            print(f"[warn] skip {label}: bits={bits} psnr={psnr} (folder={folder})")
            continue
        qp = _parse_qp(label, qp_rx)
        bitrate_mbit = bits / 1e6
        points.append((bitrate_mbit, psnr, qp, label))

    points.sort(key=lambda p: (p[2] if p[2] is not None else 10**9, p[0]))
    return points

# ==============================
# CLI / Plot
# ==============================
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root_jpeg", help="Root with JPEG-style baseline folders (planeimg_*_qpXX). Used by the 'JPEG Compression' curve spec.")
    ap.add_argument("--annotate", action="store_true", help="Annotate points with QP/labels")
    ap.add_argument("--root_dir", default="plots")
    return ap.parse_args()

def _resolve_path(base_root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(base_root, p)

def main():
    args = parse_args()
    # Collect points per curve
    curve_points: List[Tuple[str, List[Tuple[float, float, Optional[int], str]]]] = []

    for spec in CURVES:
        if not spec.get("enabled", True):
            continue
        name = spec["name"]
        kind = spec["kind"]

        if kind == "folder_scan":
            root_key = spec.get("root_from_cli", "root_jpeg")
            root_dir = getattr(args, root_key, None)
            if not root_dir:
                print(f"[warn] skip curve '{name}': missing CLI root '{root_key}'")
                continue
            pts = collect_folder_scan(spec, root_dir)
        elif kind == "mapping":
            pts = collect_mapping(spec)
        else:
            print(f"[warn] unknown curve kind '{kind}' for '{name}', skipping")
            continue

        if pts:
            curve_points.append((name, pts))
        else:
            print(f"[warn] curve '{name}' produced 0 points; skipping in plot")

    if not curve_points:
        raise SystemExit("No valid points collected for any curve. Check paths/regex/specs.")

    # Resolve output
    out_path = os.path.join(args.root_dir, "rd_curve.png")
    csv_dir = args.root_dir
    os.makedirs(args.root_dir, exist_ok=True)

    # ---- Plot ----
    plt.figure(figsize=(7.5, 5.0))
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

    for idx, (curve_name, pts) in enumerate(curve_points):
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        marker = markers[idx % len(markers)]
        plt.plot(x, y, marker=marker, linestyle="-", label=curve_name)

    plt.xlabel("Bitrate (Mbit)", fontsize=14)
    plt.ylabel("PSNR (dB)", fontsize=14)
    plt.title("I-frame (one TriPlane)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.65)
    plt.legend()

    if args.annotate:
        for curve_name, pts in curve_points:
            for (xi, yi, qp, label) in pts:
                tag = f"QP{qp}" if qp is not None else label
                plt.annotate(tag, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    # ---- CSV outputs (one per curve) ----
    for curve_name, pts in curve_points:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", curve_name.strip().lower())
        csv_path = os.path.join(csv_dir, f"rd_points_{safe}.csv")
        rows = [(p[0], p[1], p[2] if p[2] is not None else "", p[3]) for p in pts]
        save_csv(csv_path, rows)

    print(f"[DONE] Saved RD plot: {out_path}")
    for curve_name, pts in curve_points:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", curve_name.strip().lower())
        csv_path = os.path.join(csv_dir, f"rd_points_{safe}.csv")
        print(f"[DONE] Saved CSV ({curve_name}): {csv_path}")

    # Pretty print
    for curve_name, pts in curve_points:
        print(f"\n{curve_name} (ordered by QP asc):")
        for (br, psnr, qp, label) in pts:
            tag = f"QP{qp}" if qp is not None else label
            print(f"  {tag:>10s}  bitrate={br:.3f} Mbit  PSNR={psnr:.3f} dB")

if __name__ == "__main__":
    main()
