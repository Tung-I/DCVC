#!/usr/bin/env python
"""
Tri-Plane vs. RGB coding statistics (per-plane/stream granularity)
================================================================
*   **Pre-quant histograms** — unchanged (density + three planes)
*   **Spatial metrics** for **12 Tri-Plane streams** (3 planes × 4 RGB triplets)
    under **global** *and* **per-channel** quantisation, plus the RGB baseline.
*   **Temporal metrics** (MAD + optical-flow magnitude) for the same streams.
*   All numbers written to `<outdir>/metrics.txt` in tab-separated form.

Key spatial metrics
-------------------
| Symbol | Meaning (per frame, averaged) |
|--------|--------------------------------|
| **K95** | #first zig-zag DCT coeffs capturing ≥95 % energy. Large ⇒ poor compaction |
| **H_DCT** | Entropy (bits) of quantised coeff magnitudes (q=1) |
| **PNGbpp** | Bits-per-pixel of lossless PNG (quick compressibility proxy) |

Temporal metrics
----------------
| Symbol | Meaning |
|--------|---------|
| **MAD** | motion-compensated mean-abs difference (lower ⇒ more redundancy) |
| **FlowMag** | mean Farneback optical-flow magnitude (pixels) |

Usage
-----
```bash
python triplane_stats.py \
  --ckpt          logs/.../fine_last_0.tar \
  --plane-dir     logs/.../planeimg_00_19_grouped_global \
  --plane-dir-pc  logs/.../planeimg_00_19_grouped_per_channel \
  --rgb-dir       data/n3d/flame_steak/llff \
  --frame-count   20 \
  --outdir        analysis_flame_steak
```
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.fftpack import dct
from tqdm import tqdm

NBITS      = 16
PNG_SCALE  = (2 ** NBITS) - 1
ZIGZAG_8   = [(0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),
              (4,0),(3,1),(2,2),(1,3),(0,4),(0,5),(1,4),(2,3),(3,2),(4,1),
              (5,0),(6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),(0,7),(1,6),
              (2,5),(3,4),(4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
              (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),(7,2),(7,3),
              (6,4),(5,5),(4,6),(3,7),(4,7),(5,6),(6,5),(7,4),(7,5),(6,6),
              (5,7),(6,7),(7,6),(7,7)]

# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def _load_grouped_triplets(img_dir: Path, frame_count: int) -> List[np.ndarray]:
    """Returns list of RGB float images (0‑1). Handles new/old directory naming."""
    images = []
    for fid in range(frame_count):
        fn = img_dir / f"im{fid+1:05d}.png"
        if not fn.exists():
            raise FileNotFoundError(fn)
        arr = cv2.imread(str(fn), cv2.IMREAD_UNCHANGED).astype(np.float32) / PNG_SCALE

        if arr.shape[0] > 2048 or arr.shape[1] > 2048:
            arr = cv2.resize(arr, (arr.shape[1]//4, arr.shape[0]//4), interpolation=cv2.INTER_AREA)
        elif arr.shape[0] > 1024 or arr.shape[1] > 1024:
            # downsample 4x if larger than 1024 pixels in either dimension
            arr = cv2.resize(arr, (arr.shape[1]//2, arr.shape[0]//2), interpolation=cv2.INTER_AREA)
        
    
        if arr.ndim == 2:  # grayscale, duplicate channels
            arr = np.stack([arr]*3, -1)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        images.append(arr)
    return images


def _load_rgb_frames(rgb_dir: Path, frame_count: int) -> List[np.ndarray]:
    """Load first‐view frames with new ‘camera_0/image_0000.png’ convention."""
    frames = []
    for i in range(frame_count):
        fn = rgb_dir / f"{i}/images/image_0000.jpg"
        if not fn.exists():
            raise FileNotFoundError(fn)
        img = cv2.imread(str(fn), cv2.IMREAD_UNCHANGED)
        # downsample 4x
        if img.shape[0] > 2048 or img.shape[1] > 2048:
            img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), interpolation=cv2.INTER_AREA)
        elif img.shape[0] > 1024 or img.shape[1] > 1024:
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        # conver to float32 and normalise
        arr = img.astype(np.float32) / 255.0
        frames.append(arr)
    return frames

# -------------------------------------------------------------------------
# Metrics – spatial
# -------------------------------------------------------------------------

def _luma(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2]


def dct_k95(img: np.ndarray) -> float:
    """Return K (#coeffs) required for 95 % energy in one frame."""
    y = _luma(img).astype(np.float32)
    H, W = y.shape[:2]
    y = y[: H//8*8, : W//8*8].reshape(-1, 8, 8)
    needed = []
    for blk in y:
        coeff = dct(dct(blk.T, norm="ortho").T, norm="ortho")
        e = coeff**2
        total = e.sum()
        cum = 0.
        for k,(i,j) in enumerate(ZIGZAG_8,1):
            cum += e[i,j]
            if cum / total >= 0.95:
                needed.append(k)
                break
    return float(np.mean(needed))


def dct_entropy(img: np.ndarray, q: int = 1) -> float:
    y = _luma(img).astype(np.float32)
    H, W = y.shape[:2]
    y = y[: H//8*8, : W//8*8].reshape(-1, 8, 8)
    hist = np.zeros(256, dtype=np.int64)
    for blk in y:
        coeff = np.rint(dct(dct(blk.T, norm="ortho").T, norm="ortho") / q).astype(np.int32)
        vals, counts = np.unique(np.abs(coeff), return_counts=True)
        hist[vals[:256]] += counts[:256]
    p = hist / hist.sum()
    p = p[p>0]
    return -np.sum(p*np.log2(p))


def png_bpp(imgs: List[np.ndarray]) -> float:
    sizes = []
    for img in imgs:
        success, buf = cv2.imencode(".png", (img*255).astype(np.uint8))
        if success:
            sizes.append(len(buf)*8 / (img.shape[0]*img.shape[1]))
    return float(np.mean(sizes))

# -------------------------------------------------------------------------
# Metrics – temporal
# -------------------------------------------------------------------------

def temporal_mad(imgs: List[np.ndarray]) -> float:
    if len(imgs) < 2:
        return 0.0
    diffs = []
    for a,b in zip(imgs[:-1], imgs[1:]):
        a_gray = cv2.cvtColor((a*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        b_gray = cv2.cvtColor((b*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        shift,_ = cv2.phaseCorrelate(a_gray.astype(np.float32), b_gray.astype(np.float32))
        dy,dx = shift
        M = np.float32([[1,0,-dx],[0,1,-dy]])
        b_align = cv2.warpAffine(b_gray, M, (b_gray.shape[1], b_gray.shape[0]))
        diffs.append(np.mean(np.abs(a_gray.astype(np.float32)-b_align)))
    return float(np.mean(diffs))


def flow_mag(imgs: List[np.ndarray]) -> float:
    if len(imgs) < 2:
        return 0.0
    mags = []
    for a,b in zip(imgs[:-1], imgs[1:]):
        a_gray = cv2.cvtColor((a*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        b_gray = cv2.cvtColor((b*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(np.linalg.norm(flow, axis=2).mean())
    return float(np.mean(mags))

# -------------------------------------------------------------------------
# Pre‑quantisation histogram (for completeness)
# -------------------------------------------------------------------------

def plot_pre_quant(ckpt: Path, outdir: Path):
    ck=torch.load(ckpt,map_location="cpu",weights_only=False)
    dens=ck["model_state_dict"]["density.grid"].flatten().numpy()
    planes={k.split(".")[-1]:v.flatten().numpy() for k,v in ck["model_state_dict"].items() if "k0" in k and "plane" in k and "residual" not in k}
    fig,ax=plt.subplots(1,4,figsize=(18,4))
    for i,(title,data) in enumerate([("density",dens)]+list(planes.items())):
        counts,bins,_=ax[i].hist(data,bins=100,density=True,alpha=.7)
        ax[i].set_title(title)
        ax[i].set_xlabel("Value"); ax[i].set_ylabel("PDF")
        ax[i].set_ylim(0,counts.max()*1.1)
    fig.tight_layout(); fig.savefig(outdir/"pre_quant_hist.png",dpi=200); plt.close(fig)
# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------

def main():
    pa=argparse.ArgumentParser()
    # pa.add_argument("--ckpt",type=Path,required=True)
    pa.add_argument("--logdir", required=True, help="Path to Tri‑plane checkpoints (.tar files)")
    pa.add_argument("--rgb-dir",type=Path,required=True)
    pa.add_argument("--frame-count",type=int,default=20)
    pa.add_argument("--outdir",type=Path,default=Path("analysis"))
    args=pa.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)

    # print("► Pre‑quantisation histograms …")
    # plot_pre_quant(args.ckpt,args.outdir)

    rgb_seq=_load_rgb_frames(args.rgb_dir,args.frame_count)

    # ------------------------------------------------------------------
    # iterate 3 planes × 4 streams × {global,per‑ch}
    # ------------------------------------------------------------------
    results=[]
    results.append(("RGB",rgb_seq))

    for strategy in ["tiling", "separate", "grouped", "correlation", "flatfour"]:
        for q in ["global", "per_channel"]:
            for plane in ["xy_plane","xz_plane","yz_plane"]:
                print(f"► {strategy} {q} {plane} ...")
                if strategy == "correlation" or strategy == "grouped":
                    img_dir = Path(args.logdir) / f"planeimg_00_19_{strategy}_{q}" / plane / "stream0"
                elif strategy == "separate":
                    img_dir = Path(args.logdir) / f"planeimg_00_19_{strategy}_{q}" / plane / "c0"
                else:
                    img_dir = Path(args.logdir) / f"planeimg_00_19_{strategy}_{q}" / plane
                seq =_load_grouped_triplets(img_dir, args.frame_count)
                results.append((f"{strategy}_{q}_{plane}", seq))


    # ------------------------------------------------------------------
    # compute metrics
    # ------------------------------------------------------------------
    lines=["Name\tK95\tH_DCT\tPNGbpp\tMAD\tFlowMag"]
    for name,seq in results:
        k=np.mean([dct_k95(img) for img in seq])
        h=np.mean([dct_entropy(img) for img in seq])
        bpp=png_bpp(seq)
        mad=temporal_mad(seq)
        fl=flow_mag(seq)
        lines.append(f"{name}\t{k:.2f}\t{h:.2f}\t{bpp:.3f}\t{mad:.2f}\t{fl:.2f}")
        print(f"{name:<20}: K95 {k:6.2f} | H {h:5.2f} bits | PNG {bpp:5.2f} bpp | MAD {mad:6.2f} | Flow {fl:5.2f}")

    (args.outdir/"metrics.txt").write_text("\n".join(lines))
    print("\nMetrics saved to",args.outdir/"metrics.txt")

if __name__=="__main__":
    main()

