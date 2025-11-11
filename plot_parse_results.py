#!/usr/bin/env python3
import os
import re
import glob
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

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

def read_total_bits_from_encoded_bits_txt(folder: str,
                                          rel_path: str = "encoded_bits.txt") -> Optional[float]:
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

def read_avg_psnr_from_render_dir(folder: str,
                                  rel_dir: str = "render_test") -> Optional[float]:
    """
    Looks for all files matching *_{psnr}.txt and averages them.
    Falls back to '0_psnr.txt' if pattern yields nothing.
    """
    metric = "psnr"
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
        "metrics": {"dir": "render_test"},     # read psnr from here
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
    },
]

# ==============================
# Collectors
# ==============================

def _read_bits(bits_spec: Dict[str, Any], folder: str) -> Optional[float]:
    btype = bits_spec.get("type")
    if btype == "encoded_bits_total":
        return read_total_bits_from_encoded_bits_txt(
            folder,
            bits_spec.get("path", "encoded_bits.txt"),
        )
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
        psnr = read_avg_psnr_from_render_dir(
            folder,
            curve["metrics"].get("dir", "render_test"),
        )
        if psnr is None:
            print(f"[warn] skip {label}: no PSNR found (folder={folder})")
            continue
        qp = _parse_qp(label, lvl_rx)
        bitrate_mbit = bits / 1e6
        out.append(
            dict(
                bitrate=bitrate_mbit,
                psnr=psnr,
                qp=qp,
                label=label,
                folder=folder,
            )
        )
    # sort by (QP asc, bitrate asc)
    out.sort(
        key=lambda d: (
            d["qp"] if d["qp"] is not None else 10**9,
            d["bitrate"],
        )
    )
    return out

# ==============================
# Main
# ==============================

def main():
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
            print(f"[warn] curve '{name}' produced 0 valid points")

    if not curve_points:
        raise SystemExit("No valid points collected for any curve. Check paths/specs.")

    # Pretty print (ordered by QP asc)
    for curve_name, pts in curve_points:
        print(f"\n{curve_name} (ordered by QP asc):")
        for d in pts:
            tag = f"QP{d['qp']}" if d["qp"] is not None else d["label"]
            print(
                f"  {tag:>8s}  "
                f"bitrate={d['bitrate']:.6f} Mbit/s  "
                f"PSNR={d['psnr']:.3f} dB"
            )

if __name__ == "__main__":
    main()
