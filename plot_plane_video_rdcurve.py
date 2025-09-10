#!/usr/bin/env python3
import os, re, glob, json, csv, argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Your sets (explicit mapping)
# ==============================
set_hevc: Dict[str, str] = {
    'crf44': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_crf44_g10_yuv444p',
    'crf28': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_crf28_g10_yuv444p',
    'crf12': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_crf12_g10_yuv444p',
}
set_av1: Dict[str, str] = {
    'crf44': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_crf44_g10_yuv444p',
    'crf28': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_crf28_g10_yuv444p',
    'crf12': 'logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_av1_crf12_g10_yuv444p',
}
set_hevc_adap: Dict[str, str] = {
    'crf44': 'logs/dynerf_flame_steak/hevc_crf44/compressed_hevc_crf44_g10_yuv444p',
    'crf28': 'logs/dynerf_flame_steak/hevc_crf28/compressed_hevc_crf28_g10_yuv444p',
    'crf12': 'logs/dynerf_flame_steak/hevc_crf12/compressed_hevc_crf12_g10_yuv444p',
}
set_av1_adap: Dict[str, str] = {
    'crf44': 'logs/dynerf_flame_steak/av1_crf44/compressed_av1_crf44_g10_yuv444p',
    'crf28': 'logs/dynerf_flame_steak/av1_crf28/compressed_av1_crf28_g10_yuv444p',
    'crf12': 'logs/dynerf_flame_steak/av1_crf12/compressed_av1_crf12_g10_yuv444p',
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
        if rows and len(rows[0]) == 4:
            w.writerow(["bitrate_Mbit", "psnr_dB", "level", "label_or_folder"])
        elif rows and len(rows[0]) == 3:
            w.writerow(["bitrate_Mbit", "psnr_dB", "level"])
        else:
            w.writerow(["bitrate_Mbit", "psnr_dB"])
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
            # If file lacks explicit TOTAL line, fall back to sum of entries
            return sum_fallback
    except Exception:
        return None
    return None

# ---- PSNR: average over render_test/{i}_psnr.txt ----
def read_avg_psnr_from_render_dir(folder: str, rel_dir: str = "render_test") -> Optional[float]:
    """
    Finds all files matching render_test/*_psnr.txt and averages their float values.
    Falls back to render_test/0_psnr.txt if pattern yields nothing.
    """
    rdir = os.path.join(folder, rel_dir)
    if not os.path.isdir(rdir):
        return None

    paths = sorted(glob.glob(os.path.join(rdir, "*_psnr.txt")))
    vals: List[float] = []
    if paths:
        for p in paths:
            v = read_float(p)
            if v is not None:
                vals.append(v)
        if vals:
            return float(np.mean(vals))
        return None

    # Fallback to a single-file convention
    single = os.path.join(rdir, "0_psnr.txt")
    v = read_float(single)
    return v

def _parse_level(text: str, rx: Optional[str]) -> Optional[int]:
    """
    Parse a numeric level from label (e.g., 'crf28' -> 28).
    Use rx like r'.*?crf(\d+)' or r'.*?qp(\d+)'.
    """
    if not rx:
        return None
    m = re.match(rx, text) if rx.startswith("^") else re.search(rx, text)
    return int(m.group(1)) if m else None

# ==============================
# Curve specs
# ==============================
CURVES: List[Dict[str, Any]] = [
    {
        "name": "HEVC",
        "enabled": True,
        "kind": "mapping",
        "paths": set_hevc,  # label -> folder
        "level_regex": r".*?crf(\d+)",  # pull CRF from label (e.g., 'crf28')
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "psnr": {"type": "avg_render_dir", "dir": "render_test"},
    },
    {
        "name": "AV1",
        "enabled": True,
        "kind": "mapping",
        "paths": set_av1,  # label -> folder
        "level_regex": r".*?crf(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "psnr": {"type": "avg_render_dir", "dir": "render_test"},
    },
    {
        "name": "HEVC (codec-adapted)",
        "enabled": True,
        "kind": "mapping",
        "paths": set_hevc_adap,  # label -> folder
        "level_regex": r".*?crf(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "psnr": {"type": "avg_render_dir", "dir": "render_test"},
    },
    {
        "name": "AV1 (codec-adapted)",
        "enabled": True,
        "kind": "mapping",
        "paths": set_av1_adap,  # label -> folder
        "level_regex": r".*?crf(\d+)",
        "bits": {"type": "encoded_bits_total", "path": "encoded_bits.txt"},
        "psnr": {"type": "avg_render_dir", "dir": "render_test"},
    }
    # If you later want to enable a folder scan, add a spec with kind="folder_scan"
    # and a "folder_regex" to match folder names, similar to the old script.
]

# ==============================
# Collectors
# ==============================
def _read_bits(bits_spec: Dict[str, Any], folder: str) -> Optional[float]:
    btype = bits_spec.get("type")
    if btype == "encoded_bits_total":
        return read_total_bits_from_encoded_bits_txt(folder, bits_spec.get("path", "encoded_bits.txt"))
    elif btype == "dcvc_json":
        # Kept for completeness; unused in your current HEVC/AV1 flow
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

def _read_psnr(psnr_spec: Dict[str, Any], folder: str) -> Optional[float]:
    ptype = psnr_spec.get("type")
    if ptype == "avg_render_dir":
        return read_avg_psnr_from_render_dir(folder, psnr_spec.get("dir", "render_test"))
    elif ptype == "file":
        # legacy single-file path
        return read_float(os.path.join(folder, psnr_spec.get("path", "render_test/0_psnr.txt")))
    else:
        raise ValueError(f"Unknown psnr.type: {ptype}")

def collect_folder_scan(curve: Dict[str, Any], root_dir: str) -> List[Tuple[float, float, Optional[int], str]]:
    folder_rx = re.compile(curve["folder_regex"])
    lvl_rx = curve.get("level_regex")  # optional
    points = []
    if not os.path.isdir(root_dir):
        print(f"[warn] folder_scan root does not exist: {root_dir}")
        return points

    for name in sorted(os.listdir(root_dir)):
        m = folder_rx.match(name)
        if not m:
            continue
        level = int(m.group(1)) if m.groups() else _parse_level(name, lvl_rx)
        folder = os.path.join(root_dir, name)

        bits = _read_bits(curve["bits"], folder)
        psnr = _read_psnr(curve["psnr"], folder)
        if bits is None or psnr is None:
            print(f"[warn] skip {name}: bits={bits} psnr={psnr} (folder={folder})")
            continue

        # Per your assumption T=FPS, segment duration=1s ⇒ Mbps = total_bits / 1e6
        bitrate_mbit = bits / 1e6
        points.append((bitrate_mbit, psnr, level, name))

    # Order by level (CRF/QP) then bitrate
    points.sort(key=lambda p: (p[2] if p[2] is not None else 10**9, p[0]))
    return points

def collect_mapping(curve: Dict[str, Any]) -> List[Tuple[float, float, Optional[int], str]]:
    paths: Dict[str, str] = curve["paths"]
    lvl_rx = curve.get("level_regex")
    points = []
    for label, folder in paths.items():
        bits = _read_bits(curve["bits"], folder)
        psnr = _read_psnr(curve["psnr"], folder)
        if bits is None or psnr is None:
            print(f"[warn] skip {label}: bits={bits} psnr={psnr} (folder={folder})")
            continue
        level = _parse_level(label, lvl_rx)
        bitrate_mbit = bits / 1e6
        points.append((bitrate_mbit, psnr, level, label))

    points.sort(key=lambda p: (p[2] if p[2] is not None else 10**9, p[0]))
    return points

# ==============================
# CLI / Plot
# ==============================
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # If you add a folder_scan curve, wire its root via CLI like this:
    ap.add_argument("--root_jpeg", help="(unused unless a folder_scan spec references it)")
    ap.add_argument("--annotate", action="store_true", help="Annotate points with CRF/labels")
    ap.add_argument("--root_dir", default="plots")
    ap.add_argument("--title", default="TriPlane video (10 FPS, 10 frames per segment)", help="Plot title")
    return ap.parse_args()

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
        raise SystemExit("No valid points collected for any curve. Check paths/specs.")

    # Resolve output
    out_path = os.path.join(args.root_dir, "rd_curve.png")
    csv_dir = args.root_dir
    os.makedirs(args.root_dir, exist_ok=True)

    # ---- Plot ----
    plt.figure(figsize=(7.5, 5.0))
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

    for idx, (curve_name, pts) in enumerate(curve_points):
        x = np.array([p[0] for p in pts], dtype=float)  # Mbps
        y = np.array([p[1] for p in pts], dtype=float)  # dB
        marker = markers[idx % len(markers)]
        plt.plot(x, y, marker=marker, linestyle="-", label=curve_name)

    plt.xlabel("Bitrate (Mbit/s)", fontsize=14)
    plt.ylabel("PSNR (dB)", fontsize=14)
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.65)
    plt.legend()

    if args.annotate:
        for curve_name, pts in curve_points:
            for (xi, yi, level, label) in pts:
                tag = (f"CRF{level}" if level is not None else label)
                plt.annotate(tag, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    # ---- CSV outputs (one per curve) ----
    for curve_name, pts in curve_points:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", curve_name.strip().lower())
        csv_path = os.path.join(csv_dir, f"rd_points_{safe}.csv")
        rows = [(p[0], p[1], (p[2] if p[2] is not None else ""), p[3]) for p in pts]
        save_csv(csv_path, rows)

    print(f"[DONE] Saved RD plot: {out_path}")
    for curve_name, pts in curve_points:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", curve_name.strip().lower())
        csv_path = os.path.join(csv_dir, f"rd_points_{safe}.csv")
        print(f"[DONE] Saved CSV ({curve_name}): {csv_path}")

    # Pretty print
    for curve_name, pts in curve_points:
        print(f"\n{curve_name} (ordered by level asc):")
        for (br, psnr, level, label) in pts:
            tag = f"CRF{level}" if level is not None else label
            print(f"  {tag:>10s}  bitrate={br:.3f} Mbit/s  PSNR={psnr:.3f} dB")

if __name__ == "__main__":
    main()
