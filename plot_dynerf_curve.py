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
            {"qp": 28, "bitrate": 2.217, "psnr": 28.853},
            {"qp": 32, "bitrate": 1.596, "psnr": 27.701},
            # {"qp": 36, "bitrate": 1.213, "psnr": 26.801},
            {"qp": 40, "bitrate": 0.915, "psnr": 25.773},
            {"qp": 44, "bitrate": 0.685, "psnr": 24.624}
        ],
    },
    "TeTriRF-HEVC": {
        "codec": "HEVC",
        "method": "TeTriRF",
        "points": [
            {"qp": 28, "bitrate": 3.406, "psnr": 30.378},
            {"qp": 32, "bitrate": 1.958, "psnr": 28.927},
            # {"qp": 36, "bitrate": 1.152, "psnr": 27.014},
            {"qp": 40, "bitrate": 0.704, "psnr": 25.235},
            {"qp": 44, "bitrate": 0.437, "psnr": 23.679}
        ],
    },
    "TeTriRF-AV1": {
        "codec": "AV1",
        "method": "TeTriRF",
        "points": [
            {"qp": 38, "bitrate": 4.350, "psnr": 31.279},
            {"qp": 44, "bitrate": 2.859, "psnr": 30.608},
            # {"qp": 50, "bitrate": 1.884, "psnr": 29.559},
            {"qp": 56, "bitrate": 1.120, "psnr": 27.899},
            {"qp": 62, "bitrate": 0.471, "psnr": 24.713}
        ],
    },
    "TeTriRF-DCVC": {
        "codec": "DCVC",
        "method": "TeTriRF",
        # "points": [
        #     {"qp": 60, "bitrate": 1.033, "psnr": 25.56},
        #     {"qp": 48, "bitrate": 0.602, "psnr": 24.09},
        #     {"qp": 36, "bitrate": 0.365, "psnr": 22.45},
        #     {"qp": 24, "bitrate": 0.218, "psnr": 21.13}
        # ],
        "points": [
            {"qp": 60, "bitrate": 2.066, "psnr": 25.56},
            {"qp": 48, "bitrate": 1.204, "psnr": 24.09},
            {"qp": 36, "bitrate": 0.730, "psnr": 22.45},
            {"qp": 24, "bitrate": 0.436, "psnr": 21.13}
        ],
    },
    "CatRF-VP9": {
        "codec": "VP9",
        "method": "CatRF",
        "points": [
            # {"qp": 28, "bitrate": 5.322, "psnr": 30.883},
            {"qp": 32, "bitrate": 3.421, "psnr": 30.507},
            {"qp": 36, "bitrate": 2.292, "psnr": 29.843},
            {"qp": 40, "bitrate": 1.560, "psnr": 29.127},
            {"qp": 44, "bitrate": 1.080, "psnr": 28.237},
        ],
    },
    "CatRF-HEVC": {
        "codec": "HEVC",
        "method": "CatRF",
        "points": [
            # {"qp": 28, "bitrate": 6.315, "psnr": 31.644},
            {"qp": 32, "bitrate": 3.165, "psnr": 31.247},
            {"qp": 36, "bitrate": 1.716, "psnr": 30.693},
            {"qp": 40, "bitrate": 0.984, "psnr": 29.764},
            {"qp": 44, "bitrate": 0.576, "psnr": 28.588}
        ],
    },
    # "CatRF-AV1": {
    #     "codec": "AV1",
    #     "method": "CatRF",
    #     "points": [
    #         # {"qp": 38, "bitrate": 5.306, "psnr": 31.493},
    #         {"qp": 44, "bitrate": 3.643, "psnr": 31.420},
    #         {"qp": 50, "bitrate": 2.458, "psnr": 31.032},
    #         {"qp": 56, "bitrate": 1.583, "psnr": 30.595},
    #         {"qp": 62, "bitrate": 0.673, "psnr": 29.327}
    #     ],
    # },
    "CatRF-AV1-TV": {
        "codec": "AV1",
        "method": "CatRF",
        # "points": [
        #     {"qp": 44, "bitrate": 1.260, "psnr": 31.37},
        #     {"qp": 50, "bitrate": 0.931, "psnr": 31.00},
        #     {"qp": 56, "bitrate": 0.671, "psnr": 30.51},
        #     {"qp": 62, "bitrate": 0.328, "psnr": 29.16}
        # ],
        "points": [
            {"qp": 44, "bitrate": 2.520, "psnr": 31.37},
            {"qp": 50, "bitrate": 1.862, "psnr": 31.00},
            {"qp": 56, "bitrate": 1.342, "psnr": 30.51},
            {"qp": 62, "bitrate": 0.656, "psnr": 29.16}
        ],
    },
    "CatRF-DCVC-TV": {
        "codec": "DCVC",
        "method": "CatRF",
        # "points": [
        #     {"qp": 24, "bitrate": 0.268, "psnr": 28.81},
        #     {"qp": 36, "bitrate": 0.419, "psnr": 29.81},
        #     {"qp": 48, "bitrate": 0.649, "psnr": 30.63},
        #     {"qp": 60, "bitrate": 1.002, "psnr": 31.18}
        # ],
        "points": [
            {"qp": 60, "bitrate": 2.004, "psnr": 31.18},
            {"qp": 48, "bitrate": 1.298, "psnr": 30.63},
            {"qp": 36, "bitrate": 0.838, "psnr": 29.81},
            {"qp": 24, "bitrate": 0.536, "psnr": 28.81}
        ],
    },
    "VRVVC": {
        "codec": " ",
        "method": "VRVVC",
        # "points": [
        #     {"qp": 24, "bitrate": 0.315, "psnr": 28.81},
        #     {"qp": 36, "bitrate": 0.550, "psnr": 29.81},
        #     {"qp": 48, "bitrate": 0.949, "psnr": 30.63},
        #     {"qp": 60, "bitrate": 1.650, "psnr": 31.18}
        # ],
        # "points": [
        #     {"qp": 54, "bitrate": 1.074, "psnr": 29.65},
        #     {"qp": 56, "bitrate": 1.233, "psnr": 30.316},
        #     {"qp": 58, "bitrate": 1.407, "psnr": 30.96},
        # ],
        # "points": [
        #     {"qp": 60, "bitrate": 3.300, "psnr": 31.73},
        #     {"qp": 48, "bitrate": 1.898, "psnr": 31.17},
        #     {"qp": 36, "bitrate": 1.100, "psnr": 30.39},
        #     {"qp": 24, "bitrate": 0.630, "psnr": 29.14}
        # ],
        "points": [
            {"qp": 60, "bitrate": 3.300, "psnr": 31.73},
            {"qp": 48, "bitrate": 2.814, "psnr": 30.96},
            {"qp": 36, "bitrate": 2.466, "psnr": 30.316},
            {"qp": 24, "bitrate": 2.148, "psnr": 29.65}
        ],
    },
}

# ============================================================
# 2) GLOBAL AXIS LIMITS (edit here instead of CLI arguments)
#    Set to None if you want matplotlib to auto-scale that axis.
# ============================================================

XLIM_PSNR: Tuple[float, float] | None = (0.0, 5.0)
YLIM_PSNR: Tuple[float, float] | None = (21.0, 32.0)

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
    out_path_png = os.path.join(args.root_dir, "rd_curve.png")
    out_path_pdf = os.path.join(args.root_dir, "rd_curve.pdf")

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
        br_psnr = [p["bitrate"] for p in pts_sorted if p.get("psnr") is not None]
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
            fontsize=8,
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
