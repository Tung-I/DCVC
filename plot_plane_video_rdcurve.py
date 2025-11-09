#!/usr/bin/env python3
import os, re, glob, json, csv, argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Your sets (explicit mapping)
# ==============================
set_vp9: Dict[str, str] = {
    'qp28': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_vp9_qp28_g20_yuv444p',
    'qp32': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_vp9_qp32_g20_yuv444p',
    'qp36': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_vp9_qp36_g20_yuv444p',
    'qp40': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_vp9_qp40_g20_yuv444p',
    'qp44': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_vp9_qp44_g20_yuv444p',
}
set_hevc: Dict[str, str] = {
    'qp28': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_qp28_g20_yuv444p',
    'qp32': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_qp32_g20_yuv444p',
    'qp36': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_qp36_g20_yuv444p',
    'qp40': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_qp40_g20_yuv444p',
    'qp44': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_qp44_g20_yuv444p',
}
set_av1: Dict[str, str] = {
    'qp62': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp62_g20_yuv444p',
    'qp56': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp56_g20_yuv444p',
    'qp50': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp50_g20_yuv444p',
    'qp44': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp44_g20_yuv444p',
    'qp38': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp38_g20_yuv444p',
    # 'qp32': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp32_g20_yuv444p',
    # 'qp26': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp26_g20_yuv444p',
    # 'qp20': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_qp20_g20_yuv444p',
}
set_vp9_ours: Dict[str, str] = {
    'qp28': 'logs/dynerf_flame_steak/vp9_qp28/compressed_vp9_qp28_g20_yuv444p',
    'qp32': 'logs/dynerf_flame_steak/vp9_qp32/compressed_vp9_qp32_g20_yuv444p',
    'qp36': 'logs/dynerf_flame_steak/vp9_qp36/compressed_vp9_qp36_g20_yuv444p',
    'qp40': 'logs/dynerf_flame_steak/vp9_qp40/compressed_vp9_qp40_g20_yuv444p',
    'qp44': 'logs/dynerf_flame_steak/vp9_qp44/compressed_vp9_qp44_g20_yuv444p',
}
set_hevc_ours: Dict[str, str] = {
    'qp28': 'logs/dynerf_flame_steak/hevc_qp28/compressed_hevc_qp28_g20_yuv444p',
    'qp32': 'logs/dynerf_flame_steak/hevc_qp32/compressed_hevc_qp32_g20_yuv444p',
    'qp36': 'logs/dynerf_flame_steak/hevc_qp36/compressed_hevc_qp36_g20_yuv444p',
    'qp40': 'logs/dynerf_flame_steak/hevc_qp40/compressed_hevc_qp40_g20_yuv444p',
    'qp44': 'logs/dynerf_flame_steak/hevc_qp44/compressed_hevc_qp44_g20_yuv444p',
}
set_av1_ours: Dict[str, str] = {
    'qp62': 'logs/dynerf_flame_steak/av1_qp62/compressed_av1_qp62_g20_yuv444p',
    'qp56': 'logs/dynerf_flame_steak/av1_qp56/compressed_av1_qp56_g20_yuv444p',
    'qp50': 'logs/dynerf_flame_steak/av1_qp50/compressed_av1_qp50_g20_yuv444p',
    'qp44': 'logs/dynerf_flame_steak/av1_qp44/compressed_av1_qp44_g20_yuv444p',
    'qp38': 'logs/dynerf_flame_steak/av1_qp38/compressed_av1_qp38_g20_yuv444p',
    # 'qp32': 'logs/dynerf_flame_steak/av1_qp32/compressed_av1_qp32_g20_yuv444p',
    # 'qp26': 'logs/dynerf_flame_steak/av1_qp26/compressed_av1_qp26_g20_yuv444p',
    # 'qp20': 'logs/dynerf_flame_steak/av1_qp20/compressed_av1_qp20_g20_yuv444p',
}

# ==============================
# Generic readers
# ==============================
def read_float(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return float(str(f.read()).strip())
    except Exception:
        return None

def save_csv(path: str, rows: List[Tuple]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitrate_Mbit", "psnr_dB", "ssim", "lpips", "qp", "label_or_folder"])
        for r in rows:
            w.writerow(r)

# ---- Bits: read TOTAL from encoded_bits.txt ----
def read_total_bits_from_encoded_bits_txt(folder: str, rel_path: str = "encoded_bits.txt") -> Optional[float]:
    """
    Expect file like:
        xy_plane: total_bits=... ...
        xz_plane: total_bits=... ...
        yz_plane: total_bits=... ...
        density : total_bits=... ...
        TOTAL   : total_bits=13659104  bpp=...
    Returns total_bits as float (bits).
    """
    path = os.path.join(folder, rel_path)
    if not os.path.isfile(path):
        return None
    total_bits = None
    sum_fallback = 0.0
    found_any = False
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m_total = re.search(r"^TOTAL\s*:.*?total_bits\s*=\s*([0-9]+)", line)
                if m_total:
                    total_bits = float(m_total.group(1))
                m_each = re.search(r"total_bits\s*=\s*([0-9]+)", line)
                if m_each:
                    sum_fallback += float(m_each.group(1))
                    found_any = True
        if total_bits is not None:
            return total_bits
        if found_any:
            return sum_fallback
    except Exception:
        return None
    return None

# ---- Metrics: average over render_test/{*}_{metric}.txt ----
def read_avg_metric_from_render_dir(folder: str, metric: str, rel_dir: str = "render_test") -> Optional[float]:
    """
    metric in {"psnr","ssim","lpips"}
    Looks for all files matching *_{metric}.txt and averages them.
    Falls back to {metric}.txt named as '0_metric.txt' if pattern yields nothing.
    """
    assert metric in ("psnr", "ssim", "lpips")
    rdir = os.path.join(folder, rel_dir)
    if not os.path.isdir(rdir):
        return None

    paths = sorted(glob.glob(os.path.join(rdir, f"*_{metric}.txt")))
    vals: List[float] = []
    if paths:
        for p in paths:
            v = read_float(p)
            if v is not None:
                vals.append(v)
        if vals:
            return float(np.mean(vals))
        return None

    # fallback single-file convention
    single = os.path.join(rdir, f"0_{metric}.txt")
    return read_float(single)

def _parse_qp(text: str, rx: Optional[str]) -> Optional[int]:
    """Parse a QP from label (e.g., 'qp28' -> 28)."""
    if not rx:
        return None
    m = re.match(rx, text) if rx.startswith("^") else re.search(rx, text)
    return int(m.group(1)) if m else None

# ==============================
# Curve specs
# ==============================
CURVES: List[Dict[str, Any]] = [
    {
        "name": "TeTriRF-VP9",
        "enabled": True,
        "kind": "mapping",
        "paths": set_vp9,  # label -> folder
        "level_regex": r".*?qp(\d+)",          # QP, not CRF
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "metrics": {"dir": "render_test"},     # read psnr/ssim/lpips from here
    },
    {
        "name": "TeTriRF-HEVC",
        "enabled": True,
        "kind": "mapping",
        "paths": set_hevc,
        "level_regex": r".*?qp(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "metrics": {"dir": "render_test"},
    },
    {
        "name": "TeTriRF-AV1",
        "enabled": True,
        "kind": "mapping",
        "paths": set_av1,
        "level_regex": r".*?qp(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "metrics": {"dir": "render_test"},
    },
    {
        "name": "CatRF-VP9",
        "enabled": True,
        "kind": "mapping",
        "paths": set_vp9_ours,
        "level_regex": r".*?qp(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "metrics": {"dir": "render_test"},
    },
    {
        "name": "CatRF-HEVC",
        "enabled": True,
        "kind": "mapping",
        "paths": set_hevc_ours,
        "level_regex": r".*?qp(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "metrics": {"dir": "render_test"},
    },
    {
        "name": "CatRF-AV1",
        "enabled": True,
        "kind": "mapping",
        "paths": set_av1_ours,
        "level_regex": r".*?qp(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "metrics": {"dir": "render_test"},
    }
]

# ==============================
# Collectors
# ==============================
def _read_bits(bits_spec: Dict[str, Any], folder: str) -> Optional[float]:
    btype = bits_spec.get("type")
    if btype == "encoded_bits_total":
        return read_total_bits_from_encoded_bits_txt(folder, bits_spec.get("path", "encoded_bits.txt"))
    elif btype == "dcvc_json":
        pat = bits_spec.get("pattern", "compress_stats_f*_q*.json")
        cand = sorted(glob.glob(os.path.join(folder, pat)))
        if not cand:
            return None
        try:
            with open(cand[0], "r") as f:
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
    else:
        raise ValueError(f"Unknown bits.type: {btype}")

def collect_mapping(curve: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths: Dict[str, str] = curve["paths"]
    lvl_rx = curve.get("level_regex")
    out: List[Dict[str, Any]] = []
    for label, folder in paths.items():
        bits = _read_bits(curve["bits"], folder)
        if bits is None:
            print(f"[warn] skip {label}: bits=None (folder={folder})")
            continue
        psnr = read_avg_metric_from_render_dir(folder, "psnr", curve["metrics"].get("dir", "render_test"))
        ssim = read_avg_metric_from_render_dir(folder, "ssim", curve["metrics"].get("dir", "render_test"))
        lpips = read_avg_metric_from_render_dir(folder, "lpips", curve["metrics"].get("dir", "render_test"))
        if psnr is None and ssim is None and lpips is None:
            print(f"[warn] skip {label}: no metrics found (folder={folder})")
            continue
        qp = _parse_qp(label, lvl_rx)
        bitrate_mbit = bits / 1e6
        out.append(dict(
            bitrate=bitrate_mbit, psnr=psnr, ssim=ssim, lpips=lpips, qp=qp, label=label, folder=folder
        ))
    # sort by (QP asc, bitrate asc)
    out.sort(key=lambda d: ((d["qp"] if d["qp"] is not None else 10**9), d["bitrate"]))
    return out

# ==============================
# Styles (color by codec, marker by method)
# ==============================
def parse_range(arg: Optional[str]) -> Optional[Tuple[float, float]]:
    if not arg:
        return None
    try:
        a, b = [float(x.strip()) for x in arg.split(",")]
        return (a, b)
    except Exception:
        return None

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--annotate", action="store_true", help="Annotate points with QP/labels")
    ap.add_argument("--root_dir", default="plots", help="Where to save outputs")
    ap.add_argument("--title", default="TriPlane video RD curves", help="Super title for the figure")

    # Colors (per codec)
    ap.add_argument("--color_vp9",  default="#1f77b4", help="Color for VP9 curves")
    ap.add_argument("--color_hevc", default="#2ca02c", help="Color for HEVC curves")
    ap.add_argument("--color_av1",  default="#d62728", help="Color for AV1 curves")

    # Markers (per method)
    ap.add_argument("--marker_ours",     default="*", help="Marker for 'Ours-*' curves")
    ap.add_argument("--marker_tetrirf",  default="o", help="Marker for 'TeTriRF-*' curves")

    # Axis ranges (per subplot)
    ap.add_argument("--xlim_psnr",  default="0,7", help="xmin,xmax for PSNR subplot")
    ap.add_argument("--ylim_psnr",  default="23,32", help="ymin,ymax for PSNR subplot")
    ap.add_argument("--xlim_ssim",  default="0,7", help="xmin,xmax for SSIM subplot")
    ap.add_argument("--ylim_ssim",  default="0.7,0.93", help="ymin,ymax for SSIM subplot")
    ap.add_argument("--xlim_lpips", default="0,7", help="xmin,xmax for LPIPS subplot")
    ap.add_argument("--ylim_lpips", default="0.24,0.53", help="ymin,ymax for LPIPS subplot")
    return ap.parse_args()

def method_and_codec_from_name(curve_name: str) -> Tuple[str, str]:
    # e.g., "Ours-AV1" -> ("Ours", "AV1"), "TeTriRF-VP9" -> ("TeTriRF", "VP9")
    if "-" in curve_name:
        m, c = curve_name.split("-", 1)
        return m.strip(), c.strip().upper()
    return curve_name.strip(), curve_name.strip().upper()

def color_for_codec(codec: str, args) -> str:
    codec = codec.upper()
    if codec == "VP9":
        return args.color_vp9
    if codec == "HEVC":
        return args.color_hevc
    if codec == "AV1":
        return args.color_av1
    return "#000000"

def marker_for_method(method: str, args) -> str:
    m = method.lower()
    if m.startswith("catrf"):
        return args.marker_ours
    if m.startswith("tetrirf"):
        return args.marker_tetrirf
    # fallback
    return "o"

# ==============================
# Main
# ==============================
def main():
    args = parse_args()

    # Collect points per curve
    curve_points: List[Tuple[str, List[Dict[str, Any]]]] = []
    for spec in CURVES:
        if not spec.get("enabled", True):
            continue
        name = spec["name"]
        kind = spec["kind"]
        if kind != "mapping":
            print(f"[warn] unknown/unsupported curve kind '{kind}' for '{name}', skipping")
            continue
        pts = collect_mapping(spec)
        if pts:
            curve_points.append((name, pts))
        else:
            print(f"[warn] curve '{name}' produced 0 points; skipping in plot")

    if not curve_points:
        raise SystemExit("No valid points collected for any curve. Check paths/specs.")

    # Resolve output
    out_path = os.path.join(args.root_dir, "rd_curve.png")
    pdf_out_path = os.path.join(args.root_dir, "rd_curve.pdf")
    os.makedirs(args.root_dir, exist_ok=True)

    # ---- Plot: 3 subplots (PSNR / SSIM / LPIPS) ----
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), squeeze=True)
    ax_psnr, ax_ssim, ax_lpips = axes

    # Keep one legend for all axes: collect handles on the PSNR axis
    legend_handles = []
    legend_labels  = []

    for curve_name, pts in curve_points:
        method, codec = method_and_codec_from_name(curve_name)
        color  = color_for_codec(codec, args)
        marker = marker_for_method(method, args)

        # Sort by QP (asc) for drawing lines in a sensible direction
        pts_sorted = sorted(pts, key=lambda d: (d["qp"] if d["qp"] is not None else 10**9, d["bitrate"]))

        # Prepare arrays for each metric (filter out None)
        # Bitrate in Mbps
        br_psnr  = [d["bitrate"] for d in pts_sorted if d["psnr"]  is not None]
        y_psnr   = [d["psnr"]    for d in pts_sorted if d["psnr"]  is not None]
        br_ssim  = [d["bitrate"] for d in pts_sorted if d["ssim"]  is not None]
        y_ssim   = [d["ssim"]    for d in pts_sorted if d["ssim"]  is not None]
        br_lpips = [d["bitrate"] for d in pts_sorted if d["lpips"] is not None]
        y_lpips  = [d["lpips"]   for d in pts_sorted if d["lpips"] is not None]

        # Labels: just the curve name (Method-Codec)
        lbl = curve_name

        # Draw lines/markers
        h1, = ax_psnr.plot(br_psnr,  y_psnr,  marker=marker, linestyle="-", color=color, label=lbl)
        ax_ssim.plot(br_ssim,  y_ssim,  marker=marker, linestyle="-", color=color, label=lbl)
        ax_lpips.plot(br_lpips, y_lpips, marker=marker, linestyle="-", color=color, label=lbl)

        # Remember one handle per curve for a single legend (avoid duplicates)
        legend_handles.append(h1)
        legend_labels.append(lbl)

        # Optional per-point annotation (QP) on PSNR subplot (to avoid clutter)
        if args.annotate:
            for d in pts_sorted:
                if d["psnr"] is None:
                    continue
                tag = f"QP{d['qp']}" if d['qp'] is not None else d['label']
                ax_psnr.annotate(tag, (d["bitrate"], d["psnr"]),
                                 textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Axis labels & titles
    for ax in axes:
        ax.set_xlabel("Bitrate (Mbit/s)", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.65)
        ax.tick_params(axis='both', labelsize=14)

    ax_psnr.set_ylabel("PSNR (dB)", fontsize=14)
    ax_ssim.set_ylabel("SSIM", fontsize=14)
    ax_lpips.set_ylabel("LPIPS", fontsize=14)

    # ax_psnr.set_title("PSNR", fontsize=12)
    # ax_ssim.set_title("SSIM", fontsize=12)
    # ax_lpips.set_title("LPIPS", fontsize=12)

    # Axis ranges (optional, per subplot)
    lims = {
        "x_psnr": parse_range(args.xlim_psnr),
        "y_psnr": parse_range(args.ylim_psnr),
        "x_ssim": parse_range(args.xlim_ssim),
        "y_ssim": parse_range(args.ylim_ssim),
        "x_lpips": parse_range(args.xlim_lpips),
        "y_lpips": parse_range(args.ylim_lpips),
    }
    if lims["x_psnr"]:  ax_psnr.set_xlim(*lims["x_psnr"])
    if lims["y_psnr"]:  ax_psnr.set_ylim(*lims["y_psnr"])
    if lims["x_ssim"]:  ax_ssim.set_xlim(*lims["x_ssim"])
    if lims["y_ssim"]:  ax_ssim.set_ylim(*lims["y_ssim"])
    if lims["x_lpips"]: ax_lpips.set_xlim(*lims["x_lpips"])
    if lims["y_lpips"]: ax_lpips.set_ylim(*lims["y_lpips"])

    # Single legend across subplots (top)
    ncol = min(len(legend_labels), 6)
    fig.legend(legend_handles, legend_labels, loc="upper center", fontsize=14, ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.98))

    # Super title + layout
    # fig.suptitle(args.title, y=0.92, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # leave space for suptitle + legend

    # Save
    plt.savefig(out_path, dpi=220)
    plt.savefig(pdf_out_path)
    plt.close()
    print(f"[DONE] Saved RD plot: {out_path}")

    # ---- CSV outputs (one per curve) ----
    for curve_name, pts in curve_points:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", curve_name.strip().lower())
        csv_path = os.path.join(args.root_dir, f"rd_points_{safe}.csv")
        rows = [(d["bitrate"], d["psnr"] if d["psnr"] is not None else "",
                 d["ssim"] if d["ssim"] is not None else "",
                 d["lpips"] if d["lpips"] is not None else "",
                 d["qp"] if d["qp"] is not None else "",
                 d["label"]) for d in pts]
        save_csv(csv_path, rows)
        print(f"[DONE] Saved CSV ({curve_name}): {csv_path}")

    # Pretty print (ordered by QP asc)
    for curve_name, pts in curve_points:
        print(f"\n{curve_name} (ordered by QP asc):")
        for d in pts:
            tag = f"QP{d['qp']}" if d['qp'] is not None else d['label']
            print(f"  {tag:>8s}  bitrate={d['bitrate']:.3f} Mbit/s"
                  f"{'  PSNR=' + f'{d['psnr']:.3f} dB' if d['psnr'] is not None else ''}"
                  f"{'  SSIM=' + f'{d['ssim']:.4f}'  if d['ssim'] is not None else ''}"
                  f"{'  LPIPS=' + f'{d['lpips']:.4f}' if d['lpips'] is not None else ''}")

if __name__ == "__main__":
    main()
