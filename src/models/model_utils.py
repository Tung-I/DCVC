# lib/plane_codec_dcvc.py  (new file)
import torch
from einops import rearrange
import numpy as np
import torch.nn.functional as F
import math
import cv2
from typing import Tuple, Optional

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

def sandwich_planes_to_rgb(
    x01: torch.Tensor,                            # [T,C,H,W] in [0,1]
    pre_unet: torch.nn.Module,                   # SmallUNet(C->3)
    pre_mlp: torch.nn.Module,                    # MLP 1x1 (C->3)
    bound_pre: torch.nn.Module,                  # BoundedProjector(3)
    align: int,                                  # DCVC_ALIGN
) -> Tuple[torch.Tensor, Tuple[int,int]]:
    """
    Learned pack: [T,C,H,W] -> y_pad [T,3,Hp,Wp], returns orig (H, W).
    """
    assert x01.dim() == 4, f"expected [T,C,H,W], got {x01.shape}"
    T, C, H, W = x01.shape

    # run per frame; keep it batched over T
    y3 = pre_mlp(x01) + pre_unet(x01)           # [T,3,H,W]
    y01 = bound_pre(y3)                          # [T,3,H,W] in [0,1]

    # pad to multiples of align
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    if pad_h or pad_w:
        y_pad = F.pad(y01, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        y_pad = y01
    return y_pad, (H, W)


def sandwich_rgb_to_planes(
    y_hat: torch.Tensor,                          # [T,3,Hp,Wp], decoder output (float in [0,1])
    orig_size: Tuple[int,int],                    # (H, W) before pad (from sandwich_planes_to_rgb)
    post_unet: torch.nn.Module,                  # SmallUNet(3->C)
    post_mlp: torch.nn.Module,                   # MLP 1x1 (3->C)
    post_bound: Optional[torch.nn.Module] = None # optional BoundedProjector(C)
) -> torch.Tensor:
    """
    Learned unpack: crop pad -> postprocess to [T,C,H,W] (ideally still in [0,1]).
    """
    H, W = orig_size
    y = y_hat[..., :H, :W]                       # [T,3,H,W]

    x_rec = post_mlp(y) + post_unet(y)           # [T,C,H,W]
    if post_bound is not None:
        x_rec = post_bound(x_rec)                # keep it in [0,1] if you prefer
    else:
        # safe clamp for stability since codec can introduce tiny overshoots
        x_rec = x_rec.clamp_(0.0, 1.0)
    return x_rec


# ======================== FEATURE PLANES (C=12) ========================
def pack_planes_to_rgb(x: torch.Tensor, align: int = DCVC_ALIGN, mode: str = "flatten"):
    """
    x : [T, C, H, W], C == 12
    -> y_pad : [T, 3, H2_pad, W2_pad] ; orig : (H2_orig, W2_orig)

    modes:
      - "mosaic":   3 groups of 4 channels; F.pixel_shuffle(scale=2) per group -> concat as RGB
      - "flat4":    3 groups of 4 channels; tile each group into 2x2 mono -> concat as RGB
      - "flatten":  tile channels to a mono 3x4 grid -> repeat to RGB (legacy)
    """
    T, C, H, W = x.shape
    if mode not in ("mosaic", "flatten", "flat4"):
        raise ValueError(f"pack: unknown mode '{mode}'")
    if C != 12:
        raise ValueError(f"pack: C must be 12 (got {C})")

    if mode == "mosaic":
        xg = x.view(T, 3, 4, H, W)
        tiles = [F.pixel_shuffle(xg[:, g], 2) for g in range(3)]  # each [T,1,2H,2W]
        y = torch.cat(tiles[::-1], dim=1)  # [T,3,2H,2W]  (B,G,R)→RGB-ish
        h2, w2 = 2 * H, 2 * W

    elif mode == "flat4":
        # Tile each 4-ch group as 2x2 mono and map to one RGB channel
        xg = x.view(T, 3, 4, H, W)                                 # [T,3,4,H,W]
        mono = [rearrange(xg[:, g], 'T (r c) H W -> T 1 (r H) (c W)', r=2, c=2) for g in range(3)]
        y = torch.cat(mono[::-1], dim=1)                           # [T,3,2H,2W]
        h2, w2 = 2 * H, 2 * W

    else:  # "flatten"
        mono = rearrange(x, 'T (r c) H W -> T 1 (r H) (c W)', r=3, c=4)  # [T,1,3H,4W]
        y = mono.repeat(1, 3, 1, 1)                                      # [T,3,3H,4W]
        h2, w2 = 3 * H, 4 * W

    # pad
    pad_h = (align - h2 % align) % align
    pad_w = (align - w2 % align) % align
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')
    return y_pad, (h2, w2)


def unpack_rgb_to_planes(y_pad: torch.Tensor, C: int, orig_size: Tuple[int, int], mode: str = "flatten"):
    """
    Inverse of pack_planes_to_rgb (kept here unchanged; you’ll extend for flat4 in your next step).
    """
    if mode not in ("mosaic", "flatten"):
        # NOTE: you'll add "flat4" inverse in the follow-up step
        raise ValueError(f"unpack: unknown mode '{mode}'")

    H2, W2 = orig_size
    y = y_pad[..., :H2, :W2]

    if mode == "mosaic":
        if C != 12:
            raise ValueError(f"unpack(mosaic): C must be 12 (got {C})")
        b, g, r = y.split(1, dim=1)
        blocks = [F.pixel_unshuffle(ch, 2) for ch in (r, g, b)]
        return torch.cat(blocks, dim=1)  # [T,12,H,W]

    else:  # "flatten"
        if C != 12:
            raise ValueError(f"unpack(flatten): C must be 12 (got {C})")
        mono = y[:, :1]                      # [T,1,3H,4W]
        if H2 % 3 != 0 or W2 % 4 != 0:
            raise ValueError(f"unpack(flatten): orig_size {(H2,W2)} not divisible by (3,4)")
        H = H2 // 3
        W = W2 // 4
        x = rearrange(mono, 'T 1 (r H) (c W) -> T (r c) H W', r=3, c=4, H=H, W=W)
        return x


# ======================== DENSITY (Dz=192) ========================
def pack_density_to_rgb(d5: torch.Tensor, align: int = DCVC_ALIGN, mode: str = "flatten"):
    """
    d5: [1,1,Dy,Dx,Dz] (Dz must be 192 for 'mosaic'/'flat4')
    -> y_pad: [1,3,H2_pad,W2_pad]; orig: (H2_orig,W2_orig)

    modes:
      - "mosaic":
          • Map to [0,1], permute to [1,Dz,Dy,Dx]
          • Split to 3 groups of 64, pixel_shuffle(scale=8) per group -> [1,1,8Dy,8Dx]
          • Concat 3 groups as RGB -> [1,3,8Dy,8Dx]
      - "flat4":
          • Map to [0,1], permute to [1,Dz,Dy,Dx]
          • Split to 3 groups of 64, tile each group into 8x8 mono -> [1,1,8Dy,8Dx]
          • Concat 3 groups as RGB -> [1,3,8Dy,8Dx]
      - "flatten" (legacy, unchanged):
          • Map to [0,1], view as [1,C=Dy,H=Dx,W=Dz], row-wise tile to mono canvas -> repeat to RGB
    """
    assert d5.dim() == 5 and d5.shape[:2] == (1, 1), f"expected [1,1,Dy,Dx,Dz], got {tuple(d5.shape)}"
    _, _, Dy, Dx, Dz = d5.shape

    if mode not in ("flatten", "mosaic", "flat4"):
        raise ValueError(f"pack_density_to_rgb: unknown mode '{mode}'")

    if mode == "flatten":
        d01 = dens_to01(d5)                       # [1,1,Dy,Dx,Dz]
        d01_chw = d01.view(1, Dy, Dx, Dz)         # [1,C=Dy,H=Dx,W=Dz]
        mono, (Hc, Wc) = tile_1xCHW(d01_chw)      # [Hc,Wc]
        y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1,3,Hc,Wc]
        h2, w2 = Hc, Wc

    else:
        # both "mosaic" and "flat4" need Dz == 192 (3 * 8 * 8)
        if Dz != 192:
            raise ValueError(f"{mode} expects Dz=192, got Dz={Dz}")

        d01 = dens_to01(d5)                                     # [1,1,Dy,Dx,Dz]
        x = d01.permute(0, 1, 4, 2, 3).reshape(1, Dz, Dy, Dx)   # [1,Dz,Dy,Dx]
        xg = x.view(1, 3, 64, Dy, Dx)                           # 3 groups of 64

        if mode == "mosaic":
            # pixel shuffle (scale=8) per group: [1,64,Dy,Dx] -> [1,1,8Dy,8Dx]
            planes = [F.pixel_shuffle(xg[:, g], 8) for g in range(3)]
            y = torch.cat(planes[::-1], dim=1)                  # [1,3,8Dy,8Dx] (B,G,R)→RGB-ish)
        else:  # "flat4"
            # tile 8x8 channels -> [1,1,8Dy,8Dx]
            mono = [rearrange(xg[:, g], 'B (r c) H W -> B 1 (r H) (c W)', r=8, c=8) for g in range(3)]
            y = torch.cat(mono[::-1], dim=1)                    # [1,3,8Dy,8Dx]

        h2, w2 = 8 * Dy, 8 * Dx

    # pad to multiples of `align`
    pad_h = (align - h2 % align) % align
    pad_w = (align - w2 % align) % align
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')
    return y_pad, (h2, w2)


def unpack_rgb_to_planes(y_pad: torch.Tensor, C: int, orig_size: Tuple[int, int], mode: str = "flatten"):
    """
    y_pad : [T,3,H2_pad,W2_pad]  (from pack_planes_to_rgb)
    C     : must be 12
    orig_size : (H2_orig, W2_orig)  (returned by pack)
    -> x : [T,C,H,W] in [0,1]
    """
    if mode not in ("mosaic", "flatten", "flat4"):
        raise ValueError(f"unpack: unknown mode '{mode}'")
    if C != 12:
        raise ValueError(f"unpack expects C==12 (got {C})")

    H2, W2 = orig_size
    y = y_pad[..., :H2, :W2]  # remove padding

    if mode == "mosaic":
        # split RGB, unshuffle (scale=2), concat in channel order (r,g,b)
        b, g, r = y.split(1, dim=1)
        blocks = [F.pixel_unshuffle(ch, 2) for ch in (r, g, b)]  # each -> [T,4,H,W]
        return torch.cat(blocks, dim=1)                           # [T,12,H,W]

    if mode == "flat4":
        # inverse of: rearrange(..., 'T (r c) H W -> T 1 (r H) (c W)', r=2,c=2), RGB order was reversed at pack
        b, g, r = y.split(1, dim=1)  # [T,1,2H,2W]
        def inv_tile_2x2(ch):
            return rearrange(ch, 'T 1 (r H) (c W) -> T (r c) H W', r=2, c=2)
        groups = [inv_tile_2x2(r), inv_tile_2x2(g), inv_tile_2x2(b)]  # each [T,4,H,W]
        return torch.cat(groups, dim=1)  # [T,12,H,W]

    # mode == "flatten"
    # y is mono repeated 3×; take first channel and invert 3×4 tiling
    mono = y[:, :1]  # [T,1,3H,4W]
    if H2 % 3 != 0 or W2 % 4 != 0:
        raise ValueError(f"unpack(flatten): orig_size {(H2,W2)} not divisible by (3,4)")
    H = H2 // 3
    W = W2 // 4
    x = rearrange(mono, 'T 1 (r H) (c W) -> T (r c) H W', r=3, c=4, H=H, W=W)  # [T,12,H,W]
    return x


# -------------------------------------------------------
# Density: inverse of pack_density_to_rgb
# -------------------------------------------------------
def unpack_density_from_rgb(
    y_pad: torch.Tensor, Dy: int, Dx: int, Dz: int, orig_size: Tuple[int, int], mode: str = "flatten"
):
    """
    Inverse of pack_density_to_rgb.

    Args:
      y_pad:     [1,3,Hp,Wp] float01
      Dy,Dx,Dz:  target density dims (Dz must be 192 for 'mosaic'/'flat4')
      orig_size: (H2_orig, W2_orig) saved by the packer (crop size before pad)
      mode:      'flatten' | 'mosaic' | 'flat4'
    Returns:
      d5: [1,1,Dy,Dx,Dz] in raw domain (still in [0,1] if you haven’t called dens_from01)
           (call dens_from01 afterward to get [-5,30] range)
    """
    if mode not in ("flatten", "mosaic", "flat4"):
        raise ValueError(f"unpack_density_from_rgb: unknown mode '{mode}'")

    H2, W2 = orig_size
    y = y_pad[..., :H2, :W2]  # [1,3,H2,W2]

    if mode == "flatten":
        # y is 3× repeated mono canvas (H2,W2) that was built by row-wise tiling
        mono = y[:, 0].squeeze(0)                  # [H2,W2]
        d01_chw = untile_to_1xCHW(mono, Dy, Dx, Dz)  # [1,Dy,Dx,Dz]
        d5 = d01_chw.view(1, 1, Dy, Dx, Dz)
        return d5

    # For 'mosaic' and 'flat4', Dz must be 192 (3 groups * 64)
    if Dz != 192:
        raise ValueError(f"{mode} expects Dz=192, got Dz={Dz}")

    b, g, r = y.split(1, dim=1)  # [1,1,H2,W2] each

    if mode == "mosaic":
        # inverse of per-group pixel_shuffle(scale=8)
        def inv_shuffle_8(ch):
            return F.pixel_unshuffle(ch, 8)       # [1,64,Dy,Dx]
        g0 = inv_shuffle_8(r); g1 = inv_shuffle_8(g); g2 = inv_shuffle_8(b)  # order back to (0,1,2)
        x = torch.cat([g0, g1, g2], dim=1)        # [1,192,Dy,Dx]
    else:
        # flat4: inverse of 8x8 tiling
        def inv_tile_8x8(ch):
            return rearrange(ch, 'B 1 (r H) (c W) -> B (r c) H W', r=8, c=8)  # [1,64,Dy,Dx]
        g0 = inv_tile_8x8(r); g1 = inv_tile_8x8(g); g2 = inv_tile_8x8(b)
        x = torch.cat([g0, g1, g2], dim=1)        # [1,192,Dy,Dx]

    # Back to [1,1,Dy,Dx,Dz] (still in [0,1])
    d01_5 = x.view(1, 1, Dz, Dy, Dx).permute(0, 1, 3, 4, 2).contiguous()  # [1,1,Dy,Dx,Dz]
    return d01_5

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