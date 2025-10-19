import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import io
import numpy as np
from typing import List, Optional
from fractions import Fraction
import cupy as cp
import pycuda.driver as cuda
import ctypes
import cv2


from src.utils.common import get_state_dict
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.models.dcvc_codec import DCVCImageCodec, DCVCVideoCodec
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb

from src.models.model_utils import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    pack_density_to_rgb, unpack_density_from_rgb,
    normalize_planes, DCVC_ALIGN,
    dens_to01, dens_from01,
    tile_1xCHW, untile_to_1xCHW,
    pad_to_align, crop_from_align,
    jpeg_roundtrip_color,
    sandwich_planes_to_rgb, sandwich_rgb_to_planes,
    hevc_video_roundtrip, av1_video_roundtrip, vp9_video_roundtrip,
)

import TeTriRF.lib.utils as tetrirf_utils
import PyNvVideoCodec as nvc

import av
from TeTriRF.lib.unet import SmallUNet, create_mlp, BoundedProjector, LinearPack


class VideoCodecWrapper(nn.Module):
    """
    Unified video codec wrapper (HEVC / AV1 / VP9) using PyAV in **QP mode**.
      CUDA: normalize/pack/pad
      CPU : encode/decode (in-memory)
      CUDA: crop/unpack/denorm (+ metrics)

    Optimization: if packing_mode == "flatten", we encode **grayscale** (1-plane) instead
    of three duplicated channels to reduce bitrate.
    """
    def __init__(self, cfg_codec, device="cuda"):
        super().__init__()
        self.device       = torch.device(device)
        self.in_channels  = int(cfg_codec.in_channels)
        self.packing_mode = cfg_codec.packing_mode
        self.align        = int(getattr(cfg_codec, "align", 32))
        self.quant_mode   = cfg_codec.quant_mode
        self.global_range = tuple(cfg_codec.global_range)

        # Backend choice
        name = str(cfg_codec.name).lower()
        if "hevc" in name:
            self.backend = "hevc"
        elif "av1" in name:
            self.backend = "av1"
        elif "vp9" in name:
            self.backend = "vp9"
        else:
            raise NotImplementedError(f"Unknown codec name '{cfg_codec.name}'")

        # Common video knobs (safe defaults)
        self.fps     = int(getattr(cfg_codec, "fps", None))
        self.gop     = int(getattr(cfg_codec, "gop", None))
        self.pix_fmt = str(getattr(cfg_codec, "pix_fmt", "yuv444p"))

        # QP knobs
        def _pick_qp(default_qp):
            if getattr(cfg_codec, "qp", None) is not None:
                return int(getattr(cfg_codec, "qp"))
            if self.backend == "hevc" and getattr(cfg_codec, "hevc_qp", None) is not None:
                return int(getattr(cfg_codec, "hevc_qp"))
            if self.backend == "av1" and getattr(cfg_codec, "av1_qp", None) is not None:
                return int(getattr(cfg_codec, "av1_qp"))
            if self.backend == "vp9" and getattr(cfg_codec, "vp9_qp", None) is not None:
                return int(getattr(cfg_codec, "vp9_qp"))
            return int(default_qp)

        if self.backend == "hevc":
            self.qp     = _pick_qp(None)
            self.preset = str(getattr(cfg_codec, "preset", getattr(cfg_codec, "hevc_preset", "medium")))
        elif self.backend == "av1":
            self.qp       = _pick_qp(None)
            self.cpu_used = str(getattr(cfg_codec, "preset", getattr(cfg_codec, "cpu_used", 6)))
        else:  # vp9
            self.qp       = _pick_qp(None)
            self.cpu_used = str(getattr(cfg_codec, "preset", getattr(cfg_codec, "cpu_used", 4)))

        self._last_bits = 0

    # ------------------------------ feature planes ------------------------------
    @torch.no_grad()
    def forward(self, frames: torch.Tensor):
        """
        frames: [T,C,H,W] float (any device). Returns recon [T,C,H,W] on input device.
        Uses grayscale path if packing_mode == "flatten".
        """
        assert frames.dim() == 4 and frames.shape[1] == self.in_channels
        in_dev = frames.device
        T, C, H, W = frames.shape

        # Normalize to [0,1], pack, pad
        x = frames.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        if self.quant_mode != "global":
            raise NotImplementedError("Only 'global' quant_mode supported.")
        scale = (self.global_range[1] - self.global_range[0])
        x01   = (x - self.global_range[0]) / scale
        canv_pad, orig_size = pack_planes_to_rgb(x01, align=self.align, mode=self.packing_mode)  # [T,3,Hp,Wp]
        H2, W2 = orig_size
        Hp, Wp = canv_pad.shape[-2:]

        use_gray = (self.packing_mode == "flatten")

        # CPU: codec round-trip
        if use_gray:
            mono_cpu = canv_pad[:, :1].detach().to("cpu", copy=True)  # [T,1,Hp,Wp]
            if self.backend == "hevc":
                rec_mono_cpu, bits = hevc_video_roundtrip(
                    mono_cpu, fps=self.fps, gop=self.gop, qp=self.qp, preset=self.preset,
                    pix_fmt=self.pix_fmt, grayscale=True
                )
            elif self.backend == "av1":
                # for q in [20, 26, 32, 38, 44]:
                #     _, bits = av1_video_roundtrip(mono_cpu, fps=30, gop=20, qp=q, cpu_used=6, pix_fmt="yuv444p", grayscale=True)
                #     print(q, bits)
                # raise Exception
                rec_mono_cpu, bits = av1_video_roundtrip(
                    mono_cpu, fps=self.fps, gop=self.gop, qp=self.qp, cpu_used=self.cpu_used,
                    pix_fmt=self.pix_fmt, grayscale=True
                )
            else:
                rec_mono_cpu, bits = vp9_video_roundtrip(
                    mono_cpu, fps=self.fps, gop=self.gop, qp=self.qp, cpu_used=self.cpu_used,
                    pix_fmt=self.pix_fmt, grayscale=True
                )
            self._last_bits = int(bits)
            # Expand back to 3 channels for unpacking
            rec_canv_cpu = rec_mono_cpu.repeat(1, 3, 1, 1)          # [T,3,Hp,Wp]
        else:
            canv_cpu = canv_pad.detach().to("cpu", copy=True)        # [T,3,Hp,Wp]
            if self.backend == "hevc":
                rec_canv_cpu, bits = hevc_video_roundtrip(
                    canv_cpu, fps=self.fps, gop=self.gop, qp=self.qp, preset=self.preset, pix_fmt=self.pix_fmt
                )
            elif self.backend == "av1":
                rec_canv_cpu, bits = av1_video_roundtrip(
                    canv_cpu, fps=self.fps, gop=self.gop, qp=self.qp, cpu_used=self.cpu_used, pix_fmt=self.pix_fmt
                )
            else:
                rec_canv_cpu, bits = vp9_video_roundtrip(
                    canv_cpu, fps=self.fps, gop=self.gop, qp=self.qp, cpu_used=self.cpu_used, pix_fmt=self.pix_fmt
                )
            self._last_bits = int(bits)

        # Back to CUDA: crop, unpack, de-normalize, metrics
        rec_canv = rec_canv_cpu.to(in_dev, non_blocking=True)[..., :H2, :W2]    # [T,3,H2,W2]
        rec01    = unpack_rgb_to_planes(rec_canv, C, (H2, W2), mode=self.packing_mode)  # [T,C,H,W]
        recon    = (rec01 * scale + self.global_range[0]).to(torch.float32)

        bpp_val = float(self._last_bits) / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        peak    = float(scale)
        psnr    = tetrirf_utils.mse2psnr_with_peak(F.mse_loss(recon, x), peak=peak)

        return recon, bpp, psnr

    # --------------------------------- density ----------------------------------
    @torch.no_grad()
    def forward_density(self, dens_seq: torch.Tensor):
        """
        dens_seq: [T,1,Dy,Dx,Dz] float (any device) -> recon same shape (on input device).
        Uses grayscale path naturally (monochrome canvases).
        """
        assert dens_seq.dim() == 5 and dens_seq.shape[1] == 1
        in_dev = dens_seq.device
        T, _, Dy, Dx, Dz = dens_seq.shape

        # CUDA: map to [0,1], tile -> mono canvases, pad
        d   = dens_seq.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        d01 = dens_to01(d)  # [T,1,Dy,Dx,Dz]

        mono_list = []
        Hc = Wc = None
        for t in range(T):
            chw  = d01[t].view(1, Dy, Dx, Dz)                      # [1,C,H,W]
            mono, (Hct, Wct) = tile_1xCHW(chw)                     # [Hc,Wc]
            if Hc is None:
                Hc, Wc = Hct, Wct
            mono_list.append(mono)
        mono_stack = torch.stack(mono_list, dim=0)                 # [T,Hc,Wc]
        canv       = mono_stack.unsqueeze(1)                        # [T,1,Hc,Wc]

        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        canv_pad = F.pad(canv, (0, pad_w, 0, pad_h), mode="replicate")   # [T,1,Hp,Wp]
        Hp, Wp = canv_pad.shape[-2:]

        # CPU: codec round-trip (grayscale)
        mono_cpu = canv_pad.detach().to("cpu", copy=True)           # [T,1,Hp,Wp]
        if self.backend == "hevc":
            rec_mono_cpu, bits = hevc_video_roundtrip(
                mono_cpu, fps=self.fps, gop=self.gop, qp=self.qp, preset=self.preset,
                pix_fmt=self.pix_fmt, grayscale=True
            )
        elif self.backend == "av1":
            rec_mono_cpu, bits = av1_video_roundtrip(
                mono_cpu, fps=self.fps, gop=self.gop, qp=self.qp, cpu_used=self.cpu_used,
                pix_fmt=self.pix_fmt, grayscale=True
            )
        else:
            rec_mono_cpu, bits = vp9_video_roundtrip(
                mono_cpu, fps=self.fps, gop=self.gop, qp=self.qp, cpu_used=self.cpu_used,
                pix_fmt=self.pix_fmt, grayscale=True
            )
        self._last_bits = int(bits)

        # Back to CUDA: crop, untile, de-normalize, metrics
        rec_mono = rec_mono_cpu.to(in_dev, non_blocking=True)[..., :Hc, :Wc]  # [T,1,Hc,Wc]
        d01_recs = []
        for t in range(T):
            rec_chw  = untile_to_1xCHW(rec_mono[t, 0], Dy, Dx, Dz)            # [1,Dy,Dx,Dz]
            d01_recs.append(rec_chw)
        d01_rec = torch.stack(d01_recs, dim=0).view(T, 1, Dy, Dx, Dz)
        d_rec   = dens_from01(d01_rec)

        bpp_val = float(self._last_bits) / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        mse     = F.mse_loss(d_rec, d)
        psnr    = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr


class DCVCVideoCodecWrapper(nn.Module):
    """
    Segment-wise DCVC wrapper for TriPlane features and density grids.

    Interfaces (time-major):
      - forward(frames):        frames shape [T, C, H, W]
      - forward_density(dseq):  dseq   shape [T, 1, Dy, Dx, Dz]

    Returns:
      recon     : [T, C, H, W]  (or [T, 1, Dy, Dx, Dz] for density)
      bpp       : scalar tensor (bits per padded pixel over the segment)
      psnr      : scalar tensor (raw-domain, global peak)
    """
    def __init__(self, cfg_dcvc, device="cuda"):
        super().__init__()
        self.device       = torch.device(device)
        self.packing_mode = cfg_dcvc.packing_mode
        self.pack_fn      = pack_planes_to_rgb
        self.unpack_fn    = unpack_rgb_to_planes
        self.qp           = int(cfg_dcvc.dcvc_qp)
        self.quant_mode   = cfg_dcvc.quant_mode
        self.global_range = tuple(cfg_dcvc.global_range)
        self.align        = int(getattr(cfg_dcvc, "align", DCVC_ALIGN))
        self.in_channels  = int(cfg_dcvc.in_channels)
        self.use_amp      = bool(getattr(cfg_dcvc, "use_amp", False))
        self.amp_dtype    = torch.float16

        # ---- instantiate the official video codec (I/P) ----
        self.codec = DCVCVideoCodec(
            i_frame_weight_path='/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar',
            p_frame_weight_path='/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_video.pth.tar',
            reset_interval=int(getattr(cfg_dcvc, "reset_interval", 32)),
            intra_period=int(getattr(cfg_dcvc, "intra_period", -1)),
            device=str(self.device),
            half_precision=self.use_amp,   # safe: nets converted after update()
        )

    # ------------------------------ feature planes ------------------------------

    def forward(self, frames: torch.Tensor):
        """
        frames : [T, C, H, W]  float32/float16 on any device (C should match cfg.in_channels)
        -> recon [T, C, H, W] on input device; bpp, psnr scalars
        """
        assert frames.dim() == 4, "Expected [T,C,H,W]"
        assert frames.shape[1] == self.in_channels, f"expected C={self.in_channels}, got {frames.shape[1]}"
        in_dev = frames.device
        T, C, H, W = frames.shape

        x = frames.to(torch.float32)

        # 1) Normalize to [0,1] over the same policy used in training
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)  # [T,C,H,W]

        # 2) Pack each time-step to 3ch (RGB-ish) + pad to align
        #    pack_planes_to_rgb works on [T,C,H,W]
        y_pad, orig_size = self.pack_fn(x01, align=self.align, mode=self.packing_mode)   # [T,3,H2p,W2p], (H2,W2)
        H2p, W2p = y_pad.shape[-2:]
        H2,  W2  = orig_size

        # 3) Run DCVC video codec on [B=1, T, 3, H2p, W2p]
        vid = y_pad.unsqueeze(0).to(self.device)                    # [1,T,3,H2p,W2p]
        # channels_last doesn't apply to 5D tensors—ensure contiguous
        vid = vid.contiguous()

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            enc = self.codec.compress(vid, self.qp)                 # dict with total_bits, shape meta
            dec = self.codec.decompress(enc)                        # [1,T,3,H2p,W2p] in [0,1]
        x_hat_3 = dec[:, :, :, :H2p, :W2p]                          # [1,T,3,H2p,W2p]
        x_hat_3 = x_hat_3.squeeze(0).to(torch.float32)              # [T,3,H2p,W2p]

        # 4) Crop to original pack size and unpack back to planes per frame
        x_hat_3 = x_hat_3[..., :H2, :W2]                            # [T,3,H2,W2]
        rec01   = self.unpack_fn(x_hat_3, C, (H2, W2), mode=self.packing_mode)  # [T,C,H,W]

        # 5) De-normalize to raw domain and move to caller device
        recon = (rec01 * scale + c_min).to(torch.float32).to(in_dev, non_blocking=True)

        # 6) Bits-per-padded-pixel over the segment
        total_bits = float(enc["total_bits"])
        bpp = torch.tensor(total_bits / float(T * H2p * W2p), device=in_dev, dtype=torch.float32)

        # 7) PSNR in raw domain (global mode)
        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])  # typically 40
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            psnr = tetrirf_utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            raise NotImplementedError("Per-channel PSNR not implemented for video wrapper.")

        return recon, bpp, psnr

    # --------------------------------- density ----------------------------------

    @torch.no_grad()
    def forward_density(self, dens_seq: torch.Tensor):
        """
        dens_seq : [T, 1, Dy, Dx, Dz]  float on any device
                   (If you prefer [1,T,Dy,Dx,Dz], pass dens_seq = dens_seq.squeeze(0).)
        Returns:
            d_rec : [T, 1, Dy, Dx, Dz]  on input device
            bpp   : scalar tensor
            psnr  : scalar tensor (peak=35 for [-5,30] mapping)
        """
        assert dens_seq.dim() == 5 and dens_seq.shape[1] == 1, "Expected [T,1,Dy,Dx,Dz]"
        in_dev = dens_seq.device
        T, _, Dy, Dx, Dz = dens_seq.shape

        # (i) map to [0,1]
        d01 = dens_to01(dens_seq)                                  # [T,1,Dy,Dx,Dz]

        # (ii) treat each t as [1,C,H,W] with C=Dy, H=Dx, W=Dz; tile → mono canvas
        #      Canvas size is time-invariant for fixed Dy,Dx,Dz.
        mono_list = []
        for t in range(T):
            chw = d01[t].view(1, Dy, Dx, Dz)                        # [1,C,H,W]
            mono, (Hc, Wc) = tile_1xCHW(chw)                        # [Hc,Wc]
            mono_list.append(mono)
        # stack & repeat to 3ch
        mono_stack = torch.stack(mono_list, dim=0)                  # [T,Hc,Wc]
        y3 = mono_stack.unsqueeze(1).repeat(1, 3, 1, 1)             # [T,3,Hc,Wc]

        # (iii) pad bottom/right to align (same for all frames)
        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        y_pad = F.pad(y3, (0, pad_w, 0, pad_h), mode="replicate")   # [T,3,Hp,Wp]
        Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]

        # (iv) run DCVC video codec
        vid = y_pad.unsqueeze(0).to(self.device)                    # [1,T,3,Hp,Wp]
        vid = vid.contiguous()
        enc = self.codec.compress(vid, self.qp)
        dec = self.codec.decompress(enc)                            # [1,T,3,Hp,Wp]
        x_hat = dec.squeeze(0).to(torch.float32)[..., :Hc, :Wc]     # [T,3,Hc,Wc]

        # (v) take mono channel back, untile each t → [1,Dy,Dx,Dz], stack back
        d01_recs = []
        for t in range(T):
            mono_rec = x_hat[t, 0]                                  # [Hc,Wc]
            rec_chw  = untile_to_1xCHW(mono_rec, Dy, Dx, Dz)        # [1,Dy,Dx,Dz]
            d01_recs.append(rec_chw)
        d01_rec = torch.stack(d01_recs, dim=0)                      # [T,1?,Dy,Dx,Dz] but rec_chw is [1,C,H,W]
        d01_rec = d01_rec.view(T, Dy, Dx, Dz).unsqueeze(1)          # [T,1,Dy,Dx,Dz]

        # (vi) map back to raw density and move to caller device
        d_rec = dens_from01(d01_rec).to(in_dev, non_blocking=True)  # [T,1,Dy,Dx,Dz]

        # bpp over padded pixels across T
        total_bits = float(enc["total_bits"])
        bpp = torch.tensor(total_bits / float(T * Hp * Wp), device=in_dev, dtype=torch.float32)

        # PSNR with fixed peak=35 (for [-5,30] mapping)
        mse = F.mse_loss(d_rec, dens_seq.to(d_rec.dtype))
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr

    

def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    # safe channel flip with positive strides
    return np.ascontiguousarray(img[..., ::-1])

def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[..., ::-1])

def _to_uint8_from_float01_bgr(img_f01_bgr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img_f01_bgr * 255.0), 0, 255).astype(np.uint8)

def _to_float01_from_uint8_bgr(img_u8_bgr: np.ndarray) -> np.ndarray:
    return img_u8_bgr.astype(np.float32) / 255.0

def _jpeg_roundtrip_bgr_float01(img_f01_bgr: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    img8 = _to_uint8_from_float01_bgr(img_f01_bgr)
    img8 = np.ascontiguousarray(img8)  # ensure contiguous for OpenCV
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", img8, params)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    bits = int(buf.size) * 8
    dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)  # BGR8
    if dec is None:
        raise RuntimeError("JPEG decode failed")
    if dec.ndim == 2:
        dec = cv2.cvtColor(dec, cv2.COLOR_GRAY2BGR)
    dec = np.ascontiguousarray(dec)  # be explicit
    return _to_float01_from_uint8_bgr(dec), bits


class JPEGImageCodecWrapper(torch.nn.Module):
    """
    CPU JPEG wrapper with the same public interface as DCVCImageCodecWrapper,
    using separate modes for planes vs grid. Non-differentiable (use STE outside).
    """
    def __init__(self, cfg_jpeg, device="cuda"):
        super().__init__()
        self.device       = torch.device(device)

        # === NEW: two modes ===
        self.plane_mode   = cfg_jpeg.plane_packing_mode
        self.grid_mode    = cfg_jpeg.grid_packing_mode

        self.quant_mode   = cfg_jpeg.quant_mode
        self.global_range = cfg_jpeg.global_range
        self.in_channels  = cfg_jpeg.in_channels
        self.align        = int(getattr(cfg_jpeg, "align", DCVC_ALIGN))  # keep for padding symmetry
        self.quality      = int(cfg_jpeg.quality)

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
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)

        # 2) Pack (+pad) with plane mode
        y_pad, orig_hw = pack_planes_to_rgb(x01, align=self.align, mode=self.plane_mode)
        Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]

        # 3) JPEG round-trip on CPU (BGR path like your script)
        y_cpu_rgb = y_pad[0].permute(1, 2, 0).contiguous().cpu().numpy()   # HxWx3 float01 RGB
        y_cpu_bgr = _rgb_to_bgr(y_cpu_rgb)                                 # RGB->BGR (contiguous)
        dec_bgr_f01, bits = _jpeg_roundtrip_bgr_float01(y_cpu_bgr, quality=self.quality)
        dec_rgb_f01 = _bgr_to_rgb(dec_bgr_f01)                             # BGR->RGB (contiguous)
        dec_t = torch.from_numpy(dec_rgb_f01).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

        # 4) Crop to original, unpack back to planes, and de-normalize
        dec_t = crop_from_align(dec_t, tuple(orig_hw))
        dec_t = dec_t.to(dev, non_blocking=True)
        rec01 = unpack_rgb_to_planes(dec_t, x01.shape[1], orig_hw, mode=self.plane_mode)
        recon = (rec01 * scale + c_min).to(torch.float32)

        # 5) bpp over padded pixels (matches DCVC convention)
        bpp = torch.tensor(float(bits) / float(Hp * Wp), device=dev, dtype=torch.float32)

        # 6) PSNR in raw domain (global mode peak)
        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])  # e.g., 40
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            psnr = tetrirf_utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            raise NotImplementedError("only implemented for global mode")

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
        Dy, Dx, Dz = density_1x1.shape[-3:]

        # (i) Pack (+pad) with grid mode (includes dens_to01 internally)
        y_pad, orig_hw = pack_density_to_rgb(density_1x1, align=self.align, mode=self.grid_mode)  # [1,3,Hp,Wp]
        Hp, Wp = y_pad.shape[-2:]

        # (ii) JPEG CPU round-trip in BGR
        y_cpu_rgb = y_pad[0].permute(1, 2, 0).contiguous().cpu().numpy()
        y_cpu_bgr = _rgb_to_bgr(y_cpu_rgb)
        dec_bgr_f01, bits = _jpeg_roundtrip_bgr_float01(y_cpu_bgr, quality=self.quality)
        dec_rgb_f01 = _bgr_to_rgb(dec_bgr_f01)
        dec_t = torch.from_numpy(dec_rgb_f01).permute(2, 0, 1)[None]

        # (iii) Crop back to pre-pad size and invert density pack
        dec_t = crop_from_align(dec_t, tuple(orig_hw))
        dec_t = dec_t.to(dev, non_blocking=True)
        d01 = unpack_density_from_rgb(dec_t, Dy, Dx, Dz, orig_size=orig_hw, mode=self.grid_mode)  # [1,1,Dy,Dx,Dz] in [0,1]
        d_rec = dens_from01(d01)

        # (iv) bpp over padded pixels
        bpp = torch.tensor(float(bits) / float(Hp * Wp), device=dev, dtype=torch.float32)

        # (v) PSNR in raw density domain (peak = 35)
        mse = F.mse_loss(d_rec, density_1x1.to(d_rec.dtype))
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr
