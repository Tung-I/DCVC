import math
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- re-use your existing helpers from dcvc_wrapper.py ----
from TeTriRF.lib.dcvc_wrapper import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    _normalize_planes,
    _dens_to01, _dens_from01,
    _tile_1xCHW, _untile_to_1xCHW,
    _pad_to_align, _crop_from_align,
)
from TeTriRF.lib import utils


def _to_uint8_from_float01(img_f01_hw_or_hw3: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255] with rounding; accepts HxW (mono) or HxWx3 (BGR)."""
    img = np.clip(img_f01_hw_or_hw3, 0.0, 1.0)
    return np.rint(img * 255.0).astype(np.uint8)

def _to_float01_from_uint8(img_u8_hw_or_hw3: np.ndarray) -> np.ndarray:
    return img_u8_hw_or_hw3.astype(np.float32) / 255.0

def _jpeg_roundtrip_color(img_f01_bgr: np.ndarray, quality: int) -> Tuple[np.ndarray, int]:
    """
    JPEG encode/decode round-trip on CPU.
    img_f01_bgr: HxWx3 float in [0,1] (BGR order for OpenCV).
    Returns: (decoded_f01_bgr, encoded_bits)
    """
    img_u8 = _to_uint8_from_float01(img_f01_bgr)               # HxWx3 uint8 BGR
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg, ...) failed")
    bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)                   # HxWx3 uint8 BGR
    if dec is None:
        raise RuntimeError("cv2.imdecode failed")
    return _to_float01_from_uint8(dec), bits


class JPEGImageCodec(nn.Module):
    """
    CPU JPEG wrapper with the same public interface as DCVCImageCodec:

        forward(frame: [1,C,H,W]) -> (recon, bpp, psnr)
        forward_density(dens: [1,1,Dy,Dx,Dz]) -> (recon, bpp, psnr)

    Non-differentiable by nature; use with STE in the caller.
    """
    def __init__(self, cfg_jpeg, device="cuda", infer_mode=False):
        super().__init__()
        self.device       = torch.device(device)
        self.infer_mode   = infer_mode
        self.packing_mode = cfg_jpeg.packing_mode     # "flatten" or "mosaic"
        self.quant_mode   = cfg_jpeg.quant_mode       # "global" or "per_channel" (PSNR implemented for global)
        self.global_range = cfg_jpeg.global_range     # e.g. (-20.0, 20.0)
        self.in_channels  = cfg_jpeg.in_channels      # e.g. 12
        self.align        = int(getattr(cfg_jpeg, "align", 1))     # JPEG doesn't need alignment; keep for symmetry
        self.quality      = int(cfg_jpeg.quality)     # 1..100

    # --------------------------- tri-plane feature path ---------------------------

    @torch.no_grad()
    def forward(self, frame: torch.Tensor):
        """
        Args:
            frame: [1, C, H, W] float on any device.
        Returns:
            recon [1,C,H,W] float32 on input device
            bpp   scalar tensor (bits per padded pixel)
            psnr  scalar tensor (raw-domain, global peak)
        """
        assert frame.dim() == 4 and frame.shape[0] == 1, "Expected [1,C,H,W]"
        assert frame.shape[1] == self.in_channels, f"expected C={self.in_channels}, got {frame.shape[1]}"
        dev = frame.device

        # 1) Normalize to [0,1]
        x = frame.to(torch.float32)
        x01, c_min, scale = _normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)

        # 2) Pack to BGR image [1,3,H2p,W2p] with internal padding to 'align'
        y_pad, orig_hw = pack_planes_to_rgb(x01, align=self.align, mode=self.packing_mode)
        Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]
        H2, W2 = orig_hw

        # 3) JPEG round-trip on CPU
        #    (OpenCV uses BGR; our tensor is [1,3,H,W] with arbitrary channel semantics)
        y_cpu = y_pad[0].permute(1, 2, 0).contiguous().cpu().numpy()   # HxWx3 float01
        dec_f01_bgr, bits = _jpeg_roundtrip_color(y_cpu, quality=self.quality)
        dec_t = torch.from_numpy(dec_f01_bgr).permute(2, 0, 1)[None]   # [1,3,H,W] float01

        # 4) Crop to original, unpack back to planes, and de-normalize
        dec_t = _crop_from_align(dec_t, orig_hw)                       # [1,3,H2,W2]
        dec_t = dec_t.to(dev, non_blocking=True)    
        rec01 = unpack_rgb_to_planes(dec_t, x01.shape[1], orig_hw, mode=self.packing_mode)
        recon = (rec01 * scale + c_min).to(torch.float32)

        # 5) bpp over padded pixels (matches your DCVC convention)
        bpp = torch.tensor(float(bits) / float(Hp * Wp), device=dev, dtype=torch.float32)

        # 6) PSNR in raw domain (global mode peak)
        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])  # e.g., 40
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            psnr = utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            # implement per-channel PSNR if/when needed
            psnr = torch.tensor(0.0, device=dev, dtype=torch.float32)

        return recon, bpp, psnr

    # ------------------------------ density path ---------------------------------

    @torch.no_grad()
    def forward_density(self, density_1x1: torch.Tensor):
        """
        Args:
            density_1x1: [1,1,Dy,Dx,Dz] float32 on any device
        Returns:
            d_rec [1,1,Dy,Dx,Dz] float32 on input device
            bpp   scalar tensor (bits per padded pixel)
            psnr  scalar tensor (peak=35 for [-5,30] mapping)
        """
        assert density_1x1.dim() == 5 and density_1x1.shape[0] == 1 and density_1x1.shape[1] == 1
        dev = density_1x1.device
        _, _, Dy, Dx, Dz = density_1x1.shape

        # (i) map to [0,1]
        d01 = _dens_to01(density_1x1)                      # [1,1,Dy,Dx,Dz]

        # (ii) view as [1,C,H,W] with C=Dy, H=Dx, W=Dz and tile to mono
        d01_chw = d01.view(1, Dy, Dx, Dz)                  # [1,C,H,W]
        mono, (Hc, Wc) = _tile_1xCHW(d01_chw)              # [Hc,Wc] float01

        # (iii) repeat mono → 3ch image and pad to align
        y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0) # [1,3,Hc,Wc]
        y_pad, _ = _pad_to_align(y, align=self.align)      # [1,3,Hp,Wp]
        Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]

        # (iv) JPEG CPU round-trip
        y_cpu = y_pad[0].permute(1, 2, 0).contiguous().cpu().numpy()  # Hp×Wp×3 float01
        dec_f01_bgr, bits = _jpeg_roundtrip_color(y_cpu, quality=self.quality)
        dec_t = torch.from_numpy(dec_f01_bgr).permute(2, 0, 1)[None]  # [1,3,Hp,Wp]
        dec_t = dec_t[..., :Hc, :Wc].to(dev, non_blocking=True)        # [1,3,Hc,Wc] on dev
        mono_rec = dec_t[0, 0]                                         # <<< [Hc, Wc] 2-D canvas
        d01_rec_chw = _untile_to_1xCHW(mono_rec, Dy, Dx, Dz)           # [1,Dy,Dx,Dz]
        d_rec = _dens_from01(d01_rec_chw).view(1, 1, Dy, Dx, Dz)       # [1,1,Dy,Dx,Dz] on dev

        # bpp over padded pixels
        bpp = torch.tensor(float(bits) / float(Hp * Wp), device=dev, dtype=torch.float32)

        # PSNR in raw density domain (peak = 35)
        mse = F.mse_loss(d_rec, density_1x1.to(d_rec.dtype))
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr
