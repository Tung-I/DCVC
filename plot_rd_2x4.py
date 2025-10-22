#!/usr/bin/env python3
import os, re, glob, json, argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ======================================================================================
# Your path sets (same idea as your current script). Edit these to your runs.
# ======================================================================================
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
}

# ======================================================================================
# Generic readers (unchanged from your script, SSIM removed)
# ======================================================================================
def read_float(path: str) -> Optional[float]:
    if not os.path.isfile(path): return None
    try:
        with open(path, "r") as f:
            return float(str(f.read()).strip())
    except Exception:
        return None

def read_total_bits_from_encoded_bits_txt(folder: str, rel_path: str = "encoded_bits.txt") -> Optional[float]:
    path = os.path.join(folder, rel_path)
    if not os.path.isfile(path): return None
    total_bits = None
    sum_fallback = 0.0
    found_any = False
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
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

def read_avg_metric_from_render_dir(folder: str, metric: str, rel_dir: str = "render_test") -> Optional[float]:
    assert metric in ("psnr", "lpips")
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
    if not rx: return None
    m = re.match(rx, text) if rx.startswith("^") else re.search(rx, text)
    return int(m.group(1)) if m else None

# ======================================================================================
# Curve specs → same source folders; we’ll choose subsets per subplot.
# ======================================================================================
CURVES: List[Dict[str, Any]] = [
    {"name": "TeTriRF-HEVC", "kind": "mapping", "paths": set_hevc,      "level_regex": r".*?qp(\d+)",
     "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"}, "metrics": {"dir": "render_test"}},
    {"name": "Ours-HEVC",    "kind": "mapping", "paths": set_hevc_ours,  "level_regex": r".*?qp(\d+)",
     "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"}, "metrics": {"dir": "render_test"}},
    {"name": "TeTriRF-VP9",  "kind": "mapping", "paths": set_vp9,       "level_regex": r".*?qp(\d+)",
     "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"}, "metrics": {"dir": "render_test"}},
    {"name": "Ours-VP9",     "kind": "mapping", "paths": set_vp9_ours,  "level_regex": r".*?qp(\d+)",
     "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"}, "metrics": {"dir": "render_test"}},
    {"name": "TeTriRF-AV1",  "kind": "mapping", "paths": set_av1,       "level_regex": r".*?qp(\d+)",
     "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"}, "metrics": {"dir": "render_test"}},
    {"name": "Ours-AV1",     "kind": "mapping", "paths": set_av1_ours,  "level_regex": r".*?qp(\d+)",
     "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"}, "metrics": {"dir": "render_test"}},
]

# ======================================================================================
# Baselines JSON
# ======================================================================================
def load_baselines_json(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not os.path.isfile(path):
        print(f"[warn] baselines_json not found: {path}")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    methods = data.get("methods", [])
    normd: List[Dict[str, Any]] = []
    for m in methods:
        name   = m.get("name", "Baseline")
        marker = m.get("marker", "x")
        color  = m.get("color", "#7f7f7f")
        size   = int(m.get("size", 50))
        pts    = []
        for p in m.get("points", []):
            br   = p.get("bitrate_Mbit", None)
            psnr = p.get("psnr", p.get("psnr_dB", None))
            lp   = p.get("lpips", None)
            lab  = p.get("label", None)
            if br is None or (psnr is None and lp is None):
                continue
            pts.append({"bitrate": float(br), "psnr": (None if psnr is None else float(psnr)),
                        "lpips": (None if lp is None else float(lp)), "label": lab})
        if pts:
            normd.append({"name": name, "marker": marker, "color": color, "size": size, "points": pts})
    return normd

# ======================================================================================
# CLI & style helpers
# ======================================================================================
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
    ap.add_argument("--root_dir", default="plots", help="Where to save the figure")
    ap.add_argument("--out", default="rd_2x4.png", help="Output filename inside root_dir")
    ap.add_argument("--annotate", action="store_true", help="Annotate QP on our curves")
    ap.add_argument("--baselines_json", default=None, help="JSON file with baseline points (for col-4 only)")

    # Global axis ranges (applied to all subplots unless overridden)
    ap.add_argument("--xlim", default=None, help="xmin,xmax for bitrate (applies to all subplots)")
    ap.add_argument("--ylim_psnr", default=None, help="ymin,ymax for PSNR subplots")
    ap.add_argument("--ylim_lpips", default=None, help="ymin,ymax for LPIPS subplots")

    # Optional: separate xlim for the 4th column (summary AV1+baselines)
    ap.add_argument("--xlim_summary", default=None, help="xmin,xmax for col-4 only")

    # Colors for our two families (kept simple)
    ap.add_argument("--color_tetrirf", default="#1f77b4", help="Line color for TeTriRF")
    ap.add_argument("--color_ours",    default="#d62728", help="Line color for Ours")
    ap.add_argument("--marker_tetrirf", default="o", help="Marker for TeTriRF")
    ap.add_argument("--marker_ours",    default="*", help="Marker for Ours")

    return ap.parse_args()

# ======================================================================================
# Collectors
# ======================================================================================
def _read_bits(bits_spec: Dict[str, Any], folder: str) -> Optional[float]:
    btype = bits_spec.get("type")
    if btype == "encoded_bits_total":
        return read_total_bits_from_encoded_bits_txt(folder, bits_spec.get("path", "encoded_bits.txt"))
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
        lpips = read_avg_metric_from_render_dir(folder, "lpips", curve["metrics"].get("dir", "render_test"))
        if psnr is None and lpips is None:
            print(f"[warn] skip {label}: no metrics found (folder={folder})")
            continue
        qp = _parse_qp(label, lvl_rx)
        bitrate_mbit = bits / 1e6
        out.append(dict(
            bitrate=bitrate_mbit, psnr=psnr, lpips=lpips, qp=qp, label=label, folder=folder
        ))
    out.sort(key=lambda d: ((d["qp"] if d["qp"] is not None else 10**9), d["bitrate"]))
    return out

# ======================================================================================
# Plotting
# ======================================================================================
def draw_curve(ax, x, y, label, color, marker, annotate=False, anns=None):
    if len(x) == 0 or len(y) == 0:
        return None
    h, = ax.plot(x, y, marker=marker, linestyle="-", color=color, label=label)
    if annotate and anns:
        for (xi, yi, tag) in anns:
            ax.annotate(tag, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9)
    return h

def main():
    args = parse_args()
    os.makedirs(args.root_dir, exist_ok=True)
    out_path = os.path.join(args.root_dir, args.out)

    # Collect points from folders
    collected: Dict[str, List[Dict[str, Any]]] = {}
    for spec in CURVES:
        if spec.get("kind") != "mapping":
            continue
        pts = collect_mapping(spec)
        if pts:
            collected[spec["name"]] = pts
        else:
            print(f"[warn] curve '{spec['name']}' produced 0 points")

    # Baselines (for col-4)
    baselines = load_baselines_json(args.baselines_json)

    # Figure: 2 rows (PSNR / LPIPS) × 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.5), squeeze=True)
    (ax_psnr_h, ax_psnr_v, ax_psnr_a, ax_psnr_sum) = axes[0]
    (ax_lpips_h, ax_lpips_v, ax_lpips_a, ax_lpips_sum) = axes[1]

    # No titles (removed)

    # Helper to extract sorted arrays and annotations
    def prep_arrays(name: str):
        pts = collected.get(name, [])
        pts_sorted = sorted(pts, key=lambda d: (d["qp"] if d["qp"] is not None else 10**9, d["bitrate"]))
        br  = [d["bitrate"] for d in pts_sorted]
        yP  = [d["psnr"]    for d in pts_sorted if d["psnr"]  is not None]
        xP  = [d["bitrate"] for d in pts_sorted if d["psnr"]  is not None]
        yL  = [d["lpips"]   for d in pts_sorted if d["lpips"] is not None]
        xL  = [d["bitrate"] for d in pts_sorted if d["lpips"] is not None]
        ann = [(d["bitrate"], d["psnr"], f"QP{d['qp']}") for d in pts_sorted if d["psnr"] is not None and d["qp"] is not None]
        return (xP, yP, xL, yL, ann)

    # --- Column 1: HEVC (TeTriRF vs Ours)
    hevc_tt = prep_arrays("TeTriRF-HEVC")
    hevc_ou = prep_arrays("Ours-HEVC")
    draw_curve(ax_psnr_h, hevc_tt[0], hevc_tt[1], "TeTriRF-HEVC", args.color_tetrirf, args.marker_tetrirf, annotate=args.annotate, anns=hevc_tt[4])
    draw_curve(ax_psnr_h, hevc_ou[0], hevc_ou[1], "Ours-HEVC",    args.color_ours,    args.marker_ours,    annotate=args.annotate, anns=hevc_ou[4])
    draw_curve(ax_lpips_h, hevc_tt[2], hevc_tt[3], "TeTriRF-HEVC", args.color_tetrirf, args.marker_tetrirf)
    draw_curve(ax_lpips_h, hevc_ou[2], hevc_ou[3], "Ours-HEVC",    args.color_ours,    args.marker_ours)

    # --- Column 2: VP9 (TeTriRF vs Ours)
    vp9_tt = prep_arrays("TeTriRF-VP9")
    vp9_ou = prep_arrays("Ours-VP9")
    draw_curve(ax_psnr_v, vp9_tt[0], vp9_tt[1], "TeTriRF-VP9", args.color_tetrirf, args.marker_tetrirf, annotate=args.annotate, anns=vp9_tt[4])
    draw_curve(ax_psnr_v, vp9_ou[0], vp9_ou[1], "Ours-VP9",    args.color_ours,    args.marker_ours,    annotate=args.annotate, anns=vp9_ou[4])
    draw_curve(ax_lpips_v, vp9_tt[2], vp9_tt[3], "TeTriRF-VP9", args.color_tetrirf, args.marker_tetrirf)
    draw_curve(ax_lpips_v, vp9_ou[2], vp9_ou[3], "Ours-VP9",    args.color_ours,    args.marker_ours)

    # --- Column 3: AV1 (TeTriRF vs Ours)
    av1_tt = prep_arrays("TeTriRF-AV1")
    av1_ou = prep_arrays("Ours-AV1")
    draw_curve(ax_psnr_a, av1_tt[0], av1_tt[1], "TeTriRF-AV1", args.color_tetrirf, args.marker_tetrirf, annotate=args.annotate, anns=av1_tt[4])
    draw_curve(ax_psnr_a, av1_ou[0], av1_ou[1], "Ours-AV1",    args.color_ours,    args.marker_ours,    annotate=args.annotate, anns=av1_ou[4])
    draw_curve(ax_lpips_a, av1_tt[2], av1_tt[3], "TeTriRF-AV1", args.color_tetrirf, args.marker_tetrirf)
    draw_curve(ax_lpips_a, av1_ou[2], av1_ou[3], "Ours-AV1",    args.color_ours,    args.marker_ours)

    # --- Column 4: AV1 + Baselines (log-scale x-axis for both rows)
    draw_curve(ax_psnr_sum, av1_tt[0], av1_tt[1], "TeTriRF-AV1", args.color_tetrirf, args.marker_tetrirf, annotate=args.annotate, anns=av1_tt[4])
    draw_curve(ax_psnr_sum, av1_ou[0], av1_ou[1], "Ours-AV1",    args.color_ours,    args.marker_ours,    annotate=args.annotate, anns=av1_ou[4])
    draw_curve(ax_lpips_sum, av1_tt[2], av1_tt[3], "TeTriRF-AV1", args.color_tetrirf, args.marker_tetrirf)
    draw_curve(ax_lpips_sum, av1_ou[2], av1_ou[3], "Ours-AV1",    args.color_ours,    args.marker_ours)

    # baselines: scatter only (added to column 4 axes)
    for b in baselines:
        name   = b["name"]; marker = b["marker"]; color = b["color"]; size = b["size"]
        # PSNR points
        xP = [p["bitrate"] for p in b["points"] if p["psnr"]  is not None]
        yP = [p["psnr"]    for p in b["points"] if p["psnr"]  is not None]
        # LPIPS points
        xL = [p["bitrate"] for p in b["points"] if p["lpips"] is not None]
        yL = [p["lpips"]   for p in b["points"] if p["lpips"] is not None]
        if xP:
            ax_psnr_sum.scatter(xP, yP, marker=marker, c=color, s=size, label=name, zorder=3)
        if xL:
            ax_lpips_sum.scatter(xL, yL, marker=marker, c=color, s=size, label=name, zorder=3)

    # Axes labels / grids
    for ax in [ax_psnr_h, ax_psnr_v, ax_psnr_a, ax_psnr_sum]:
        ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.65)
    for ax in [ax_lpips_h, ax_lpips_v, ax_lpips_a, ax_lpips_sum]:
        ax.set_ylabel("LPIPS", fontsize=11)
        ax.set_xlabel("Bitrate (Mbit/s)", fontsize=11)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.65)

    # ---- Make column 4 (both rows) x-axis log10 with nice 10^k ticks
    for ax in (ax_psnr_sum, ax_lpips_sum):
        ax.set_xscale('log', base=10)
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))

    # Legends on all subfigures:
    #   PSNR row  -> lower right
    #   LPIPS row -> upper right
    psnr_axes  = [ax_psnr_h, ax_psnr_v, ax_psnr_a, ax_psnr_sum]
    lpips_axes = [ax_lpips_h, ax_lpips_v, ax_lpips_a, ax_lpips_sum]
    for ax in psnr_axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", frameon=False)
    for ax in lpips_axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", frameon=False)

    # Axis limits
    xlim_all = parse_range(args.xlim)
    xlim_sum = parse_range(args.xlim_summary)
    ylim_ps  = parse_range(args.ylim_psnr)
    ylim_lp  = parse_range(args.ylim_lpips)

    for i, ax in enumerate(psnr_axes):
        if (i == 3) and xlim_sum:
            ax.set_xlim(*xlim_sum)
        elif xlim_all:
            ax.set_xlim(*xlim_all)
        if ylim_ps:
            ax.set_ylim(*ylim_ps)
    for i, ax in enumerate(lpips_axes):
        if (i == 3) and xlim_sum:
            ax.set_xlim(*xlim_sum)
        elif xlim_all:
            ax.set_xlim(*xlim_all)
        if ylim_lp:
            ax.set_ylim(*ylim_lp)

    plt.tight_layout()
    plt.savefig(out_path, dpi=230)
    plt.close()
    print(f"[DONE] saved: {out_path}")

if __name__ == "__main__":
    main()
