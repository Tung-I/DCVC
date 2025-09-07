# lib/plane_codec_dcvc.py  (new file)
import torch
from einops import rearrange
import numpy as np
import torch.nn.functional as F
import math
import cv2
from typing import Tuple

DCVC_ALIGN = 32

def to_uint8_from_float01(img_f01_hw_or_hw3: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255] with rounding; accepts HxW (mono) or HxWx3 (BGR)."""
    img = np.clip(img_f01_hw_or_hw3, 0.0, 1.0)
    return np.rint(img * 255.0).astype(np.uint8)

def to_float01_from_uint8(img_u8_hw_or_hw3: np.ndarray) -> np.ndarray:
    return img_u8_hw_or_hw3.astype(np.float32) / 255.0

def jpeg_roundtrip_color(img_f01_bgr: np.ndarray, quality: int) -> Tuple[np.ndarray, int]:
    """
    JPEG encode/decode round-trip on CPU.
    img_f01_bgr: HxWx3 float in [0,1] (BGR order for OpenCV).
    Returns: (decoded_f01_bgr, encoded_bits)
    """
    img_u8 = to_uint8_from_float01(img_f01_bgr)               # HxWx3 uint8 BGR
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg, ...) failed")
    bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)                   # HxWx3 uint8 BGR
    if dec is None:
        raise RuntimeError("cv2.imdecode failed")
    return to_float01_from_uint8(dec), bits

def pack_planes_to_rgb(x, align=DCVC_ALIGN, mode: str = "flatten"):
    """
    x : [T, C, H, W], C ∈ {4, 12}
    -> y_pad : [T, 3, H2_pad, W2_pad]   (H2_pad, W2_pad are multiples of `align`)
       orig  : (H2_orig, W2_orig)
    modes:
      - "mosaic"  (default): original behavior
      - "flatten": for C=12, tile channels to a mono 3x4 grid [T,1,3H,4W], then repeat to RGB
                   for C=4, this reduces to the same 2x2 mono as mosaic.
    """
    T, C, H, W = x.shape
    if mode not in ("mosaic", "flatten"):
        raise ValueError(f"pack: unknown mode '{mode}'")

    if mode == "mosaic":
        if C == 4:
            mono = F.pixel_shuffle(x, 2)              # [T,1,2H,2W]
            y = mono.repeat(1, 3, 1, 1)               # [T,3,2H,2W]
        elif C == 12:
            # 3 groups of 4 channels each → pixel-shuffle each group → concat as RGB
            xg = x.view(T, 3, 4, H, W)
            tiles = [F.pixel_shuffle(xg[:, g], 2) for g in range(3)]  # each [T,1,2H,2W]
            y = torch.cat(tiles[::-1], dim=1)         # [T,3,2H,2W]  (B,G,R)→RGB-ish
        else:
            raise ValueError(f"pack: C must be 4 or 12 (got {C})")

    else:  # mode == "flatten"
        if C == 12:
            # Arrange 12 channels as a 3×4 tile grid (mono), then duplicate to 3 channels
            # [T,12,H,W] -> [T,1,3H,4W]
            mono = rearrange(x, 'T (r c) H W -> T 1 (r H) (c W)', r=3, c=4)
            y = mono.repeat(1, 3, 1, 1)               # [T,3,3H,4W]
        elif C == 4:
            # With 4 channels we can only form a 2×2 grid; same as pixel_shuffle→mono
            mono = F.pixel_shuffle(x, 2)              # [T,1,2H,2W]
            y = mono.repeat(1, 3, 1, 1)               # [T,3,2H,2W]
        else:
            raise ValueError(f"pack(flatten): C must be 4 or 12 (got {C})")

    # pad to multiples of `align`
    _, _, h2, w2 = y.shape
    pad_h = (align - h2 % align) % align
    pad_w = (align - w2 % align) % align
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')

    return y_pad, (h2, w2)   # keep (H2_orig, W2_orig) so we can crop on unpack


def unpack_rgb_to_planes(y_pad, C, orig_size, mode: str = "flatten"):
    """
    y_pad : [T,3,H2_pad,W2_pad]  (from pack_planes_to_rgb)
    C     : 4 or 12
    orig_size : (H2_orig, W2_orig)  (returned by pack)
    -> x : [T,C,H,W]
    """
    if mode not in ("mosaic", "flatten"):
        raise ValueError(f"unpack: unknown mode '{mode}'")

    H2, W2 = orig_size
    y = y_pad[..., :H2, :W2]                     # remove padding

    if mode == "mosaic":
        if C == 4:
            mono = y[:, :1]                      # any channel
            x = F.pixel_unshuffle(mono, 2)       # [T,4,H,W]
            return x
        elif C == 12:
            # invert the mosaic (split RGB, unshuffle, concat)
            b, g, r = y.split(1, dim=1)
            blocks = [F.pixel_unshuffle(ch, 2) for ch in (r, g, b)]
            return torch.cat(blocks, dim=1)      # [T,12,H,W]
        else:
            raise ValueError(f"unpack(mosaic): C must be 4 or 12 (got {C})")

    else:  # mode == "flatten"
        if C == 12:
            # y was mono repeated 3×; take first channel and invert 3×4 tiling
            mono = y[:, :1]                      # [T,1,3H,4W]
            # infer H,W from H2,W2 and the 3×4 tiling
            if H2 % 3 != 0 or W2 % 4 != 0:
                raise ValueError(f"unpack(flatten): orig_size {(H2,W2)} not divisible by (3,4)")
            H = H2 // 3
            W = W2 // 4
            x = rearrange(mono, 'T 1 (r H) (c W) -> T (r c) H W', r=3, c=4, H=H, W=W)  # [T,12,H,W]
            return x
        elif C == 4:
            # flatten with C=4 was 2×2 mono; invert as usual
            mono = y[:, :1]                      # [T,1,2H,2W]
            x = F.pixel_unshuffle(mono, 2)       # [T,4,H,W]
            return x
        else:
            raise ValueError(f"unpack(flatten): C must be 4 or 12 (got {C})")

def normalize_planes(
        seq, 
        mode="global", 
        global_range=(-20.0, 20.0), 
        eps=1e-6):
    """
    Normalize tri-plane tensor to [0,1].
    seq : [T,C,H,W] (float32/float16)
    Returns:
      seq_n    : normalized to [0,1]
      c_min    : broadcastable min used
      scale    : broadcastable (max-min)
    """
    if mode == "per_channel":
        # per-channel min/max over T,H,W
        c_min = seq.amin(dim=(0, 2, 3), keepdim=True)             # [1,C,1,1]
        c_max = seq.amax(dim=(0, 2, 3), keepdim=True)
    elif mode == "global":
        lo, hi = global_range
        c_min = torch.as_tensor(lo, dtype=seq.dtype, device=seq.device).view(1, 1, 1, 1)
        c_max = torch.as_tensor(hi, dtype=seq.dtype, device=seq.device).view(1, 1, 1, 1)
    else:
        raise ValueError(f"Unknown quant_mode: {mode}")

    scale = (c_max - c_min).clamp_(eps)
    seq_n = ((seq - c_min) / scale).clamp_(0, 1)
    return seq_n, c_min, scale


# ------------------------------------------------------------------

def pad_to_align(x, align=DCVC_ALIGN, mode="replicate"):
    """
    x : [B,3,H,W]  -> pad bottom/right so H,W are multiples of `align`.
    Returns y_pad, (H_orig, W_orig)
    """
    B, C, H, W = x.shape
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    y = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return y, (H, W)


def crop_from_align(x, orig_size):
    """
    x : [B,3,H_pad,W_pad] ; orig_size=(H_orig, W_orig)
    """
    H, W = orig_size
    return x[..., :H, :W]


# ===== Density helpers (packing/unpacking) =====
def dens_to01(d: torch.Tensor) -> torch.Tensor:
    # fixed global mapping used in your packer
    return (d.clamp(-5.0, 30.0) + 5.0) / 35.0

def dens_from01(t01: torch.Tensor) -> torch.Tensor:
    return t01 * 35.0 - 5.0

def tile_1xCHW(feat: torch.Tensor):
    """[1,C,H,W] -> mono canvas [Hc,Wc], row-wise."""
    assert feat.dim() == 4 and feat.size(0) == 1
    _, C, H, W = feat.shape
    tiles_w = int(math.ceil(math.sqrt(C)))
    tiles_h = int(math.ceil(C / tiles_w))
    Hc, Wc = tiles_h * H, tiles_w * W
    canvas = feat.new_zeros(Hc, Wc)
    filled = 0
    for r in range(tiles_h):
        y = 0
        for c in range(tiles_w):
            if filled >= C:
                break
            canvas[r*H:(r+1)*H, y:y+W] = feat[0, filled]
            y += W
            filled += 1
    return canvas, (Hc, Wc)

def untile_to_1xCHW(canvas: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
    """inverse of _tile_1xCHW"""
    Hc, Wc = canvas.shape[-2:]
    tiles_h = Hc // H
    tiles_w = Wc // W
    out = canvas.new_zeros(1, C, H, W)
    filled = 0
    for r in range(tiles_h):
        y = 0
        for c in range(tiles_w):
            if filled >= C:
                break
            out[0, filled] = canvas[r*H:(r+1)*H, y:y+W]
            y += W
            filled += 1
    return out