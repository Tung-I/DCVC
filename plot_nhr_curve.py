#!/usr/bin/env python3
import os
import argparse
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt

# ============================================================
# 1) USER-EDITABLE DATA: put your RD points here
#    Each curve has:
#       - codec: "VP9" / "HEVC" / "AV1" (used for color)
#       - method: "TeTriRF" / "CatRF" / whatever (used for marker)
#       - points: list of dicts with:
#           { "qp": int, "bitrate": float, "psnr": float }
# ============================================================

CURVE_DATA: Dict[str, Dict[str, Any]] = {
    "TeTriRF-VP9": {
        "codec": "VP9",
        "method": "TeTriRF",
        "points": [
            {"qp": 28, "bitrate": 3.204, "psnr": 31.516},
            {"qp": 32, "bitrate": 2.114, "psnr": 31.256},
            {"qp": 36, "bitrate": 1.417, "psnr": 30.816},
            {"qp": 40, "bitrate": 0.948, "psnr": 30.132}
        ],
    },
    "TeTriRF-HEVC": {
        "codec": "HEVC",
        "method": "TeTriRF",
        "points": [
            {"qp": 28, "bitrate": 1.759, "psnr": 31.087},
            {"qp": 32, "bitrate": 0.913, "psnr": 30.485},
            {"qp": 40, "bitrate": 0.500, "psnr": 29.384},
            {"qp": 44, "bitrate": 0.292, "psnr": 27.654}
        ],
    },
    "TeTriRF-AV1": {
        "codec": "AV1",
        "method": "TeTriRF",
        "points": [
            {"qp": 44, "bitrate": 0.722, "psnr": 30.319},
            {"qp": 50, "bitrate": 0.482, "psnr": 29.371},
            {"qp": 56, "bitrate": 0.309, "psnr": 28.207},
            {"qp": 62, "bitrate": 0.194, "psnr": 26.509}
        ],
    },
    "TeTriRF-DCVC": {
        "codec": "DCVC",
        "method": "TeTriRF",
        "points": [
            {"qp": 60, "bitrate": 0.390, "psnr": 31.53},
            {"qp": 48, "bitrate": 0.222, "psnr": 30.92},
            {"qp": 36, "bitrate": 0.131, "psnr": 29.93},
            {"qp": 24, "bitrate": 0.079, "psnr": 28.40}
        ],
    },
    "CatRF-VP9": {
        "codec": "VP9",
        "method": "CatRF",
        "points": [
            {"qp": 28, "bitrate": 1.861, "psnr": 32.354},
            {"qp": 32, "bitrate": 1.391, "psnr": 32.194},
            {"qp": 36, "bitrate": 1.071, "psnr": 31.913},
            {"qp": 40, "bitrate": 0.802, "psnr": 31.720}
        ],
    },
    "CatRF-HEVC": {
        "codec": "HEVC",
        "method": "CatRF",
        "points": [
            {"qp": 32, "bitrate": 1.177, "psnr": 32.233},
            {"qp": 36, "bitrate": 0.736, "psnr": 31.679},
            {"qp": 40, "bitrate": 0.469, "psnr": 31.114},
            {"qp": 44, "bitrate": 0.311, "psnr": 30.365}
        ],
    },
    "CatRF-AV1": {
        "codec": "AV1",
        "method": "CatRF",
        "points": [
            {"qp": 44, "bitrate": 0.623, "psnr": 32.603},
            {"qp": 50, "bitrate": 0.443, "psnr": 31.785},
            {"qp": 56, "bitrate": 0.310, "psnr": 31.133},
            {"qp": 62, "bitrate": 0.214, "psnr": 30.557}
        ],
    },
    "CatRF-DCVC": {
        "codec": "DCVC",
        "method": "CatRF",
        "points": [
            {"qp": 60, "bitrate": 0.297, "psnr": 32.153},
            {"qp": 48, "bitrate": 0.195, "psnr": 31.858},
            {"qp": 36, "bitrate": 0.129, "psnr": 31.444},
            {"qp": 24, "bitrate": 0.083, "psnr": 30.854}
        ],
    },
    "VRVVC": {
        "codec": " ",
        "method": "VRVVC",
        "points": [
            {"qp": 60, "bitrate": 1.257, "psnr": 32.464},
            {"qp": 48, "bitrate": 0.574, "psnr": 31.927},
            {"qp": 36, "bitrate": 0.296, "psnr": 30.759},
            {"qp": 30, "bitrate": 0.221, "psnr": 29.743}
            
        ],
    },
}

# ============================================================
# 2) GLOBAL AXIS LIMITS (edit here instead of CLI arguments)
#    Set to None if you want matplotlib to auto-scale that axis.
# ============================================================

XLIM_PSNR: Tuple[float, float] | None = (0.0, 7.0)
YLIM_PSNR: Tuple[float, float] | None = (26.0, 33.0)

# ============================================================
# 3) Style helpers: colors by codec, markers by method
# ============================================================

COLORS = {
    "VP9":  "#1f77b4",
    "HEVC": "#2ca02c",
    "AV1":  "#d62728",
    "DCVC":  "#F5A627",
}

MARKERS = {
    "TETRIRF": "o",
    "CATRF":   "*",
}

def color_for_codec(codec: str) -> str:
    return COLORS.get(codec.upper(), "#000000")

def marker_for_method(method: str) -> str:
    key = method.upper()
    for prefix, marker in MARKERS.items():
        if key.startswith(prefix):
            return marker
    return "o"

# ============================================================
# 4) CLI (minimal)
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--annotate", action="store_true",
                    help="Annotate points with QP labels on the plot")
    ap.add_argument("--root_dir", default="plots",
                    help="Directory to save outputs")
    ap.add_argument("--title", default="TriPlane video RD curves",
                    help="Title for the figure")
    return ap.parse_args()

# ============================================================
# 5) Main plotting
# ============================================================

def main():
    args = parse_args()

    # Collect curves that actually have points
    curve_points: List[Tuple[str, Dict[str, Any]]] = []
    for name, cfg in CURVE_DATA.items():
        pts = cfg.get("points", [])
        if pts:
            curve_points.append((name, cfg))
        else:
            print(f"[warn] curve '{name}' has no points; skipping")

    if not curve_points:
        raise SystemExit("No points to plot. Fill CURVE_DATA first.")

    os.makedirs(args.root_dir, exist_ok=True)
    out_path_png = os.path.join(args.root_dir, "rd_curve_nhr.png")
    out_path_pdf = os.path.join(args.root_dir, "rd_curve_nhr.pdf")

    # Single subplot: PSNR vs bitrate
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))

    legend_handles = []
    legend_labels  = []

    for curve_name, cfg in curve_points:
        codec  = cfg.get("codec", "unknown").upper()
        method = cfg.get("method", curve_name.split("-", 1)[0])
        pts    = cfg.get("points", [])

        # Sort points by QP (asc) and then bitrate
        pts_sorted = sorted(
            pts,
            key=lambda d: (d.get("qp", 10**9), d.get("bitrate", 0.0))
        )

        # Extract arrays, require psnr
        br_psnr = [p["bitrate"]*2 for p in pts_sorted if p.get("psnr") is not None]  # Convert to Mbit/s at 20 fps
        y_psnr  = [p["psnr"]    for p in pts_sorted if p.get("psnr") is not None]

        if not br_psnr:
            print(f"[warn] curve '{curve_name}' has no valid PSNR points; skipping")
            continue

        color  = color_for_codec(codec)
        marker = marker_for_method(method)
        label  = curve_name

        h_psnr, = ax.plot(
            br_psnr, y_psnr,
            marker=marker, linestyle="-",
            color=color, label=label
        )

        legend_handles.append(h_psnr)
        legend_labels.append(label)

        # Optional QP annotations
        if args.annotate:
            for p in pts_sorted:
                if p.get("psnr") is None:
                    continue
                qp = p.get("qp")
                tag = f"QP{qp}" if qp is not None else ""
                ax.annotate(
                    tag,
                    (p["bitrate"], p["psnr"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9
                )

    # Labels, grid, ticks
    ax.set_xlabel("Bitrate (Mbit/s)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.65)
    ax.tick_params(axis="both", labelsize=14)

    # Apply global axis limits if not None
    if XLIM_PSNR is not None:
        ax.set_xlim(*XLIM_PSNR)
    if YLIM_PSNR is not None:
        ax.set_ylim(*YLIM_PSNR)

    # Legend
    if legend_handles:
        ncol = min(len(legend_labels), 3)
        ax.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            fontsize=10,
            ncol=ncol,
            frameon=False,
        )

    # Title + layout
    ax.set_title(args.title, fontsize=14)
    plt.tight_layout()

    # Save
    plt.savefig(out_path_png, dpi=220)
    plt.savefig(out_path_pdf)
    plt.close()
    print(f"[DONE] Saved RD plot PNG: {out_path_png}")
    print(f"[DONE] Saved RD plot PDF: {out_path_pdf}")

if __name__ == "__main__":
    main()
