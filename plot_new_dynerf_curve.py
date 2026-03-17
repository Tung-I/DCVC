#!/usr/bin/env python3
import os
import argparse
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt

# ============================================================
# 1) USER-EDITABLE DATA (your RD points)
# ============================================================

CURVE_DATA: Dict[str, Dict[str, Any]] = {
    "TeTriRF-VP9": {
        "codec": "VP9",
        "method": "TeTriRF",
        "points": [
            {"qp": 28, "bitrate": 2.217, "psnr": 28.853},
            {"qp": 32, "bitrate": 1.596, "psnr": 27.701},
            {"qp": 40, "bitrate": 0.915, "psnr": 25.773},
            {"qp": 44, "bitrate": 0.685, "psnr": 24.624},
        ],
    },
    "TeTriRF-HEVC": {
        "codec": "HEVC",
        "method": "TeTriRF",
        "points": [
            {"qp": 28, "bitrate": 3.406, "psnr": 30.378},
            {"qp": 32, "bitrate": 1.958, "psnr": 28.927},
            {"qp": 40, "bitrate": 0.704, "psnr": 25.235},
            {"qp": 44, "bitrate": 0.437, "psnr": 23.679},
        ],
    },
    "TeTriRF-AV1": {
        "codec": "AV1",
        "method": "TeTriRF",
        "points": [
            {"qp": 38, "bitrate": 4.350, "psnr": 31.279},
            {"qp": 44, "bitrate": 2.859, "psnr": 30.608},
            {"qp": 56, "bitrate": 1.120, "psnr": 27.899},
            {"qp": 62, "bitrate": 0.471, "psnr": 24.713},
        ],
    },
    "GIFStream": {
        "codec": " ",
        "method": "GIFStream",
        "points": [
            # {"qp": 60, "bitrate": 8.100, "psnr": 32.31},
            {"qp": 48, "bitrate": 4.520, "psnr": 32.23},
            {"qp": 36, "bitrate": 3.539, "psnr": 31.52},
            {"qp": 24, "bitrate": 2.148, "psnr": 30.22},
        ],
    },
    "CATRF-VP9": {
        "codec": "VP9",
        "method": "CATRF",
        "points": [
            {"qp": 32, "bitrate": 3.421, "psnr": 30.507},
            {"qp": 36, "bitrate": 2.292, "psnr": 29.843},
            {"qp": 40, "bitrate": 1.560, "psnr": 29.127},
            {"qp": 44, "bitrate": 1.080, "psnr": 28.237},
        ],
    },
    "CATRF-HEVC": {
        "codec": "HEVC",
        "method": "CATRF",
        "points": [
            {"qp": 32, "bitrate": 3.165, "psnr": 31.247},
            {"qp": 36, "bitrate": 1.716, "psnr": 30.693},
            {"qp": 40, "bitrate": 0.984, "psnr": 29.764},
            {"qp": 44, "bitrate": 0.576, "psnr": 28.588},
        ],
    },
    "CATRF-AV1": {
        "codec": "AV1",
        "method": "CATRF",
        "points": [
            {"qp": 44, "bitrate": 2.520, "psnr": 31.37},
            {"qp": 50, "bitrate": 1.862, "psnr": 31.00},
            {"qp": 56, "bitrate": 1.342, "psnr": 30.51},
            {"qp": 62, "bitrate": 0.656, "psnr": 29.16},
        ],
    },
    "CATRF-DCVC": {
        "codec": "DCVC",
        "method": "CATRF",
        "points": [
            {"qp": 60, "bitrate": 2.004, "psnr": 31.18},
            {"qp": 48, "bitrate": 1.298, "psnr": 30.63},
            {"qp": 36, "bitrate": 0.838, "psnr": 29.81},
            {"qp": 24, "bitrate": 0.536, "psnr": 28.81},
        ],
    },
    # "VRVVC": {
    #     "codec": " ",
    #     "method": "VRVVC",
    #     "points": [
    #         {"qp": 60, "bitrate": 3.300, "psnr": 31.73},
    #         {"qp": 48, "bitrate": 2.814, "psnr": 30.96},
    #         {"qp": 36, "bitrate": 2.466, "psnr": 30.316},
    #         {"qp": 24, "bitrate": 2.148, "psnr": 29.65},
    #     ],
    # },

}

# ============================================================
# 2) PER-CURVE STYLE: color / marker / linestyle
# ============================================================

CURVE_STYLES: Dict[str, Dict[str, Any]] = {
    # TeTriRF curves
    "TeTriRF-VP9":  {"color": 'blue',      "marker": "o", "linestyle": "--"},
    "TeTriRF-HEVC": {"color": 'green',     "marker": "o", "linestyle": "--"},
    "TeTriRF-AV1":  {"color": '#D41717',   "marker": "o", "linestyle": "--"},
    "GIFStream":    {"color": "violet","marker": "d", "linestyle": "-"},

    # CATRF curves
    "CATRF-VP9":    {"color": 'dodgerblue',"marker": "s", "linestyle": "-"},
    "CATRF-HEVC":   {"color": 'limegreen', "marker": "s", "linestyle": "-"},
    "CATRF-AV1":    {"color": 'red',       "marker": "s", "linestyle": "-"},
    "CATRF-DCVC":   {"color": "slategray", "marker": "s", "linestyle": "-"},

    # VRVVC
    # "VRVVC":        {"color": "darkorange","marker": "o", "linestyle": "-"},

    
}

def style_for_curve(curve_name: str) -> Tuple[str, str, str]:
    """Return (color, marker, linestyle) for a given curve name."""
    style = CURVE_STYLES.get(curve_name, {})
    color = style.get("color", "black")
    marker = style.get("marker", "o")
    linestyle = style.get("linestyle", "-")
    return color, marker, linestyle

# ============================================================
# 3) Axes / grid style (applied to all figures)
#    Adjust these to change grid + boundary appearance
# ============================================================

AXES_STYLE = {
    "xlabel": "Bitrate (Mbps)",
    "ylabel": "PSNR (dB)",

    # ---- Line and border style (grid + boundaries) ----
    "border_color": "0.8",  # boundary (axes spines) color
    "border_width": 0.8,
    "grid_color":   "0.8",  # grid line color
    "grid_width":   0.8,
    "grid_linestyle": "-",  # solid gray grid, same as boundaries
}

# ============================================================
# 4) FIGURE CONFIGS (main + three zoom subfigures)
# ============================================================

FIGURE_SPECS: List[Dict[str, Any]] = [
    {
        # MAIN figure: roughly square
        "name": "main",
        "outfile": "rd_main_all",
        "methods": "ALL",     # plot all curves in CURVE_DATA
        "fig_width": 5.8,     # inches
        "fig_height": 5.0,    # inches
        "xlim": (0.0, 5.0),
        # "xlim": (0.0, 8.5),
        "ylim": (23.0, 33.0),
        "xticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        # "xticks": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        "yticks": [24.0, 26.0, 28.0, 30.0, 32.0],
        "show_xlabel": True,
        "show_ylabel": True,
        "show_legend": True,
        "legend_loc": "lower right",
        "legend_ncol": 2,
    },
    {
        # Subfigure 1: VP9 zoom
        "name": "zoom_vp9",
        "outfile": "rd_zoom_vp9",
        "methods": ["TeTriRF-VP9", "CATRF-VP9"],
        "fig_width": 4.0,
        "fig_height": 1.6,
        "xlim": (0.5, 3.5),
        "ylim": (24.0, 32.0),
        "xticks": [1.0, 2.0, 3.0],
        "yticks": [24.0, 28.0, 32.0],
        "show_xlabel": False,
        "show_ylabel": False,
        "show_legend": False,
        "legend_loc": "lower right",
        "legend_ncol": 2,
    },
    {
        # Subfigure 2: HEVC zoom
        "name": "zoom_hevc",
        "outfile": "rd_zoom_hevc",
        "methods": ["TeTriRF-HEVC", "CATRF-HEVC"],
        "fig_width": 4.0,
        "fig_height": 1.6,
        "xlim": (0.0, 4.0),
        "ylim": (23.0, 32.0),
        "xticks": [0.0, 1.0, 2.0, 3.0, 4.0],
        "yticks": [24.0, 28.0, 32.0],
        "show_xlabel": False,
        "show_ylabel": False,
        "show_legend": False,
        "legend_loc": "lower right",
        "legend_ncol": 2,
    },
    {
        # Subfigure 3: AV1 zoom
        "name": "zoom_av1",
        "outfile": "rd_zoom_av1",
        "methods": ["TeTriRF-AV1", "CATRF-AV1"],
        "fig_width": 4.0,
        "fig_height": 1.6,
        "xlim": (0.0, 5.0),
        "ylim": (24.0, 32.0),
        "xticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "yticks": [24.0, 28.0, 32.0],
        "show_xlabel": False,
        "show_ylabel": False,
        "show_legend": False,
        "legend_loc": "lower right",
        "legend_ncol": 2,
    },
]

DEFAULT_ROOT_DIR = "plots_new_video"
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 12

# ============================================================
# 5) CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--annotate", action="store_true",
                    help="Annotate points with QP labels on the plot.")
    ap.add_argument("--root_dir", default=DEFAULT_ROOT_DIR,
                    help="Directory to save outputs.")
    ap.add_argument("--no-main", action="store_true",
                    help="Skip the main figure, only generate subfigures.")
    ap.add_argument("--no-sub", action="store_true",
                    help="Skip subfigures, only generate the main figure.")
    return ap.parse_args()

# ============================================================
# 6) Plotting helpers
# ============================================================

def apply_axes_style(ax, xlim, ylim, xticks, yticks, show_xlabel, show_ylabel):
    """Apply labels, ticks, grid, borders, and axis limits."""
    # Axis labels
    if show_xlabel:
        ax.set_xlabel(AXES_STYLE.get("xlabel", "Bitrate"), fontsize=TITLE_FONT_SIZE)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(AXES_STYLE.get("ylabel", "Metric"), fontsize=TITLE_FONT_SIZE)
    else:
        ax.set_ylabel("")

    # Line style settings
    grid_color = AXES_STYLE.get("grid_color", "0.8")
    grid_width = AXES_STYLE.get("grid_width", 0.8)
    grid_ls    = AXES_STYLE.get("grid_linestyle", "--")

    # Grid
    ax.grid(
        True,
        linestyle=grid_ls,
        linewidth=grid_width,
        color=grid_color,
        alpha=1.0,
    )
    ax.tick_params(axis="both", labelsize=12)

    # Axis limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Manual ticks if provided
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Spines: same style as grid (per your request)
    border_color = AXES_STYLE.get("border_color", grid_color)
    border_width = AXES_STYLE.get("border_width", grid_width)
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(border_width)


def plot_single_figure(
    fig_spec: Dict[str, Any],
    all_curves: Dict[str, Dict[str, Any]],
    root_dir: str,
    annotate: bool
):
    """Plot one figure according to fig_spec, saving PNG/PDF."""
    name = fig_spec["name"]
    outfile = fig_spec["outfile"]
    methods_spec = fig_spec["methods"]
    fig_w = fig_spec["fig_width"]
    fig_h = fig_spec["fig_height"]
    xlim = fig_spec.get("xlim", None)
    ylim = fig_spec.get("ylim", None)
    xticks = fig_spec.get("xticks", None)
    yticks = fig_spec.get("yticks", None)
    show_xlabel = fig_spec.get("show_xlabel", True)
    show_ylabel = fig_spec.get("show_ylabel", True)
    show_legend = fig_spec.get("show_legend", True)
    legend_loc = fig_spec.get("legend_loc", "best")
    legend_ncol = fig_spec.get("legend_ncol", 1)

    # Determine which curves to plot
    if isinstance(methods_spec, str) and methods_spec.upper() == "ALL":
        curve_names = list(all_curves.keys())
    else:
        curve_names = methods_spec

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    legend_handles = []
    legend_labels = []

    for curve_name in curve_names:
        cfg = all_curves.get(curve_name)
        if cfg is None:
            print(f"[warn] figure '{name}': curve '{curve_name}' not found; skipping")
            continue

        pts = cfg.get("points", [])
        if not pts:
            print(f"[warn] figure '{name}': curve '{curve_name}' has no points; skipping")
            continue

        # Sort by QP then bitrate
        pts_sorted = sorted(
            pts,
            key=lambda d: (d.get("qp", 10**9), d.get("bitrate", 0.0))
        )

        br = [p["bitrate"] for p in pts_sorted if p.get("psnr") is not None]
        ps = [p["psnr"]    for p in pts_sorted if p.get("psnr") is not None]

        if not br:
            print(f"[warn] figure '{name}': curve '{curve_name}' has no valid PSNR; skipping")
            continue

        color, marker, linestyle = style_for_curve(curve_name)

        h, = ax.plot(
            br, ps,
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=curve_name,
            linewidth=1.2,
            markersize=6,
        )

        legend_handles.append(h)
        legend_labels.append(curve_name)

        # Optional QP annotation
        if annotate:
            for p in pts_sorted:
                if p.get("psnr") is None:
                    continue
                qp = p.get("qp")
                tag = f"QP{qp}" if qp is not None else ""
                ax.annotate(
                    tag,
                    (p["bitrate"], p["psnr"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

    apply_axes_style(
        ax,
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
    )

    if show_legend and legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc=legend_loc,
            fontsize=LEGEND_FONT_SIZE,
            ncol=legend_ncol,
            frameon=False,
        )

    fig.tight_layout()
    os.makedirs(root_dir, exist_ok=True)
    png_path = os.path.join(root_dir, f"{outfile}.png")
    pdf_path = os.path.join(root_dir, f"{outfile}.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[DONE] Saved figure '{name}' as {png_path} and {pdf_path}")


# ============================================================
# 7) Main
# ============================================================

def main():
    args = parse_args()
    root_dir = args.root_dir

    for spec in FIGURE_SPECS:
        is_main = (spec["name"] == "main")
        if args.no_main and is_main:
            continue
        if args.no_sub and not is_main:
            continue
        plot_single_figure(
            fig_spec=spec,
            all_curves=CURVE_DATA,
            root_dir=root_dir,
            annotate=args.annotate,
        )

if __name__ == "__main__":
    main()
