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


from src.utils.common import get_state_dict
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.models.dcvc_codec import DCVCImageCodec, DCVCVideoCodec
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb

from src.models.model_utils import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    normalize_planes, DCVC_ALIGN,
    dens_to01, dens_from01,
    tile_1xCHW, untile_to_1xCHW,
    pad_to_align, crop_from_align,
    jpeg_roundtrip_color
)
import TeTriRF.lib.utils as tetrirf_utils
import PyNvVideoCodec as nvc

import av


class HEVCVideoCodecWrapper(nn.Module):
    """
    PyAV (libx265) wrapper. Do as much as possible on CUDA:
      CUDA: normalize -> pack -> pad
      CPU : encode/decode (PyAV/libx265) using in-memory raw HEVC bitstream
      CUDA: crop -> unpack -> de-normalize (+ metrics)

    API (time-major):
      - forward(frames):        frames [T,C,H,W] -> (recon [T,C,H,W], bpp, psnr)
      - forward_density(dseq):  dseq   [T,1,Dy,Dx,Dz] -> (recon [T,1,Dy,Dx,Dz], bpp, psnr)
    """
    def __init__(self, cfg_codec, device="cuda"):
        super().__init__()
        self.device       = torch.device(device)
        self.in_channels  = int(cfg_codec.in_channels)
        self.packing_mode = cfg_codec.packing_mode
        self.align        = int(getattr(cfg_codec, "align", DCVC_ALIGN))
        self.quant_mode   = cfg_codec.quant_mode
        self.global_range = tuple(cfg_codec.global_range)

        # HEVC (x265) knobs
        self.crf     = int(getattr(cfg_codec, "crf", getattr(cfg_codec, "hevc_crf", 28)))
        # x265 presets: "ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo"
        self.preset  = str(getattr(cfg_codec, "preset", getattr(cfg_codec, "hevc_preset", "medium")))
        self.gop     = int(getattr(cfg_codec, "gop", getattr(cfg_codec, "hevc_gop", 10)))
        self.fps     = int(getattr(cfg_codec, "fps", 30))
        # We keep 4:4:4 where possible to preserve your packed planes
        self.pix_fmt = str(getattr(cfg_codec, "pix_fmt", "yuv444p"))

        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes

        self._last_bits = 0

    # ------------------------------ feature planes ------------------------------
    @torch.no_grad()
    def forward(self, frames: torch.Tensor):
        """
        frames: [T,C,H,W] float (any device). Returns recon [T,C,H,W] on input device.
        """
        assert frames.dim() == 4 and frames.shape[1] == self.in_channels
        in_dev = frames.device
        T, C, H, W = frames.shape

        # CUDA path: normalize, pack, pad
        x = frames.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)   # GPU
        canv_pad, orig_size = self.pack_fn(x01, align=self.align, mode=self.packing_mode)               # GPU [T,3,Hp,Wp]
        H2, W2 = orig_size
        Hp, Wp = canv_pad.shape[-2:]

        # CPU path: HEVC roundtrip (in-memory)
        canv_cpu = canv_pad.detach().to('cpu', copy=True)                                               # CPU
        rec_canv_cpu = self._hevc_roundtrip(canv_cpu, fps=self.fps, gop=self.gop,
                                            crf=self.crf, preset=self.preset, pix_fmt=self.pix_fmt)    # CPU [T,3,Hp,Wp]

        # Back to CUDA: crop, unpack, de-normalize, metrics
        rec_canv = rec_canv_cpu.to(in_dev, non_blocking=True)[..., :H2, :W2]                           # [T,3,H2,W2] GPU
        rec01    = self.unpack_fn(rec_canv, C, (H2, W2), mode=self.packing_mode)                       # [T,C,H,W] GPU
        recon    = (rec01 * scale + c_min).to(torch.float32)                                           # GPU

        # bpp & psnr on GPU
        bpp_val = self._last_bits / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])
            psnr = tetrirf_utils.mse2psnr_with_peak(F.mse_loss(recon, x), peak=peak)
        else:
            raise NotImplementedError("Only 'global' quant_mode supported.")

        return recon, bpp, psnr

    # --------------------------------- density ----------------------------------
    @torch.no_grad()
    def forward_density(self, dens_seq: torch.Tensor):
        """
        dens_seq: [T,1,Dy,Dx,Dz] float (any device) -> recon same shape (on input device).
        """
        assert dens_seq.dim() == 5 and dens_seq.shape[1] == 1
        in_dev = dens_seq.device
        T, _, Dy, Dx, Dz = dens_seq.shape

        # CUDA: map to [0,1], tile to canvases, pad
        d   = dens_seq.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        d01 = dens_to01(d)                                                                              # [T,1,Dy,Dx,Dz] GPU

        mono_list = []
        Hc = Wc = None
        for t in range(T):
            chw  = d01[t].view(1, Dy, Dx, Dz)                                                           # [1,C,H,W] GPU
            mono, (Hct, Wct) = tile_1xCHW(chw)                                                          # [Hc,Wc] GPU
            if Hc is None:
                Hc, Wc = Hct, Wct
            mono_list.append(mono)
        mono_stack = torch.stack(mono_list, dim=0)                                                      # [T,Hc,Wc] GPU
        canv       = mono_stack.unsqueeze(1).repeat(1, 3, 1, 1)                                         # [T,3,Hc,Wc] GPU

        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        canv_pad = F.pad(canv, (0, pad_w, 0, pad_h), mode="replicate")                                  # [T,3,Hp,Wp] GPU
        Hp, Wp = canv_pad.shape[-2:]

        # CPU: HEVC roundtrip
        rec_canv_cpu = self._hevc_roundtrip(canv_pad.detach().to('cpu', copy=True),
                                            fps=self.fps, gop=self.gop, crf=self.crf,
                                            preset=self.preset, pix_fmt=self.pix_fmt)                   # CPU [T,3,Hp,Wp]

        # Back to CUDA: crop, untile, de-normalize, metrics
        rec_canv = rec_canv_cpu.to(in_dev, non_blocking=True)[..., :Hc, :Wc]                            # [T,3,Hc,Wc] GPU

        d01_recs = []
        for t in range(T):
            mono_rec = rec_canv[t, 0]                                                                   # [Hc,Wc] GPU
            rec_chw  = untile_to_1xCHW(mono_rec, Dy, Dx, Dz)                                            # [1,Dy,Dx,Dz] GPU
            d01_recs.append(rec_chw)
        d01_rec = torch.stack(d01_recs, dim=0).view(T, 1, Dy, Dx, Dz)                                   # GPU
        d_rec   = dens_from01(d01_rec)                                                                  # GPU

        bpp_val = self._last_bits / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        mse  = F.mse_loss(d_rec, d)
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr

    # ------------------------------ PyAV HEVC core ------------------------------
    def _hevc_roundtrip(
        self,
        canvases: torch.Tensor,   # [T,3,H,W] float in [0,1], **CPU**
        fps: int, gop: int, crf: int, preset: str, pix_fmt: str
    ) -> torch.Tensor:
        """
        Encode canvases -> bytes (raw HEVC Annex-B in-memory) -> decode -> [T,3,H,W] float CPU.
        Accumulates total bitstream length in self._last_bits.
        """
        assert canvases.device.type == 'cpu', "PyAV expects CPU tensors"
        assert canvases.dim() == 4 and canvases.shape[1] == 3
        T, _, H, W = canvases.shape

        # Write raw HEVC elementary stream (Annex-B) to BytesIO
        out_buf = io.BytesIO()
        oc = av.open(out_buf, mode='w', format='hevc')  # raw H.265 stream

        stream = oc.add_stream('libx265', rate=fps)
        stream.width  = W
        stream.height = H
        stream.pix_fmt = pix_fmt                       # e.g., 'yuv444p'
        stream.time_base = Fraction(1, fps)
        stream.codec_context.time_base = Fraction(1, fps)
        stream.gop_size = gop

        # x265 options:
        # - repeat-headers=1: write VPS/SPS/PPS for easier decoding of raw streams
        # - keyint/min-keyint=gop: force fixed GOP
        # - scenecut=0: no scene-cut insertion (keeps cadence stable)
        # - bframes=0: (optional) avoid reordering if you want strict low latency
        x265_params = f"repeat-headers=1:keyint={gop}:min-keyint={gop}:scenecut=0"
        opts = {
            'crf': str(crf),
            'preset': str(preset),
            'x265-params': x265_params,
        }
        for k, v in opts.items():
            stream.codec_context.options[k] = v

        # Encode frames
        for t in range(T):
            frm = (canvases[t].clamp(0,1).permute(1,2,0).contiguous().numpy() * 255.0 + 0.5).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(frm, format='rgb24')
            # Only convert pixel format; size comes from stream
            frame = frame.reformat(format=pix_fmt)
            for packet in stream.encode(frame):
                oc.mux(packet)

        # Flush encoder
        for packet in stream.encode(None):
            oc.mux(packet)
        oc.close()

        bitstream = out_buf.getvalue()
        self._last_bits = len(bitstream) * 8

        # Decode from memory
        in_buf = io.BytesIO(bitstream)
        ic = av.open(in_buf, mode='r')   # auto-detect raw hevc

        rec_frames: List[torch.Tensor] = []
        for frame in ic.decode(video=0):
            rgb = frame.to_ndarray(format='rgb24')                                   # HxWx3 uint8
            rec_frames.append(torch.from_numpy(rgb).permute(2,0,1).float() / 255.0)  # CPU
        ic.close()

        if len(rec_frames) != T:
            raise RuntimeError(f"Decoded {len(rec_frames)} frames, expected {T} (got {len(rec_frames)})")

        out = torch.stack(rec_frames, dim=0)  # [T,3,H,W]
        if out.shape[-2:] != (H, W):
            raise RuntimeError(f"HEVC roundtrip size mismatch: in ({H},{W}) vs out {tuple(out.shape[-2:])}")
        return out


class PyNvVideoCodecWrapper(nn.Module):
    """
    NVENC/NVDEC (GPU) video codec via PyNvCodec (VPF).

    CUDA path:
      - normalize -> pack -> pad   (CUDA)
      - encode/decode (NVENC/NVDEC, GPU surfaces)
      - crop -> unpack -> denorm   (CUDA)
    """

    def __init__(self, cfg_codec, device: str = "cuda"):
        super().__init__()
        self.device       = torch.device(device)
        self.in_channels  = int(cfg_codec.in_channels)
        self.packing_mode = cfg_codec.packing_mode
        self.align        = int(getattr(cfg_codec, "align", DCVC_ALIGN))
        self.quant_mode   = cfg_codec.quant_mode
        self.global_range = tuple(cfg_codec.global_range)

        # NVENC knobs (safe defaults; tweak as needed)
        self.gpu_id   = int(getattr(cfg_codec, "gpu_id", 0))
        self.codec    = str(cfg_codec.nv_codec).lower()   # 'hevc' or 'av1' (if supported)
        self.fps      = int(cfg_codec.fps)
        self.gop      = int(cfg_codec.gop)
        self.preset   = str(cfg_codec.nv_preset)
        self.constqp  = int(cfg_codec.nv_constqp)

        fmt = str(getattr(cfg_codec, "fmt", None)).upper()
        if fmt not in {"NV12","YUV444"}:
            raise ValueError(f"Unsupported fmt {fmt} for PyNvVideoCodec")
        self.pix_fmt = fmt
        if self.pix_fmt == "NV12":
            self.profile  = "high" 
        elif self.pix_fmt == "YUV444":
            self.profile  = "high_444"
        else:
            raise ValueError(f"Unsupported pix_fmt {self.pix_fmt}")
        
        self._enc = None
        self._dec = None
        self._enc_w = self._enc_h = -1

        # tensor helpers
        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes
        self._last_bits = 0

    # ------------------------------ feature planes ------------------------------
    @torch.no_grad()
    def forward(self, frames: torch.Tensor):
        """
        frames: [T,C,H,W] float (any device). Returns recon [T,C,H,W] on input device.
        """
        assert frames.dim() == 4 and frames.shape[1] == self.in_channels
        in_dev = frames.device
        T, C, H, W = frames.shape

        # CUDA pre-processing
        x = frames.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)    # [T,C,H,W] CUDA
        canv_pad, orig_size = self.pack_fn(x01, align=self.align, mode=self.packing_mode)                # [T,3,Hp,Wp] CUDA
        H2, W2 = orig_size
        Hp, Wp = canv_pad.shape[-2:]

        # NVENC/NVDEC round-trip (GPU)
        rec_canv = self._nv_roundtrip_rgb(canv_pad)    # [T,3,Hp,Wp] CUDA

        # CUDA post-processing
        rec_canv = rec_canv[..., :H2, :W2]                                                               # [T,3,H2,W2]
        rec01    = self.unpack_fn(rec_canv, C, (H2, W2), mode=self.packing_mode)                         # [T,C,H,W]
        recon    = (rec01 * scale + c_min).to(torch.float32)                                             # [T,C,H,W] CUDA

        # metrics
        bpp_val = self._last_bits / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])
            psnr = tetrirf_utils.mse2psnr_with_peak(F.mse_loss(recon, x), peak=peak)
        else:
            raise NotImplementedError("Only 'global' quant_mode supported.")

        return recon, bpp, psnr

    # --------------------------------- density ----------------------------------
    @torch.no_grad()
    def forward_density(self, dens_seq: torch.Tensor):
        """
        dens_seq: [T,1,Dy,Dx,Dz] float (any device) -> recon same shape (on input device).
        """
        assert dens_seq.dim() == 5 and dens_seq.shape[1] == 1
        in_dev = dens_seq.device
        T, _, Dy, Dx, Dz = dens_seq.shape

        # CUDA: map to [0,1], tile to canvases, pad
        d   = dens_seq.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        d01 = dens_to01(d)                                                                               # [T,1,Dy,Dx,Dz] CUDA

        mono_list = []
        Hc = Wc = None
        for t in range(T):
            chw  = d01[t].view(1, Dy, Dx, Dz)                                                            # [1,C,H,W] CUDA
            mono, (Hct, Wct) = tile_1xCHW(chw)                                                           # [Hc,Wc]  CUDA
            if Hc is None:
                Hc, Wc = Hct, Wct
            mono_list.append(mono)
        mono_stack = torch.stack(mono_list, dim=0)                                                       # [T,Hc,Wc] CUDA
        canv       = mono_stack.unsqueeze(1).repeat(1, 3, 1, 1)                                          # [T,3,Hc,Wc] CUDA

        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        canv_pad = F.pad(canv, (0, pad_w, 0, pad_h), mode="replicate")                                   # [T,3,Hp,Wp] CUDA
        Hp, Wp = canv_pad.shape[-2:]

        # NVENC/NVDEC round-trip (GPU)
        rec_canv = self._nv_roundtrip_rgb(canv_pad)    # [T,3,Hp,Wp] CUDA
        rec_canv = rec_canv[..., :Hc, :Wc]                                                               # [T,3,Hc,Wc]

        # untile back
        d01_recs = []
        for t in range(T):
            mono_rec = rec_canv[t, 0]                                                                    # [Hc,Wc] CUDA
            rec_chw  = untile_to_1xCHW(mono_rec, Dy, Dx, Dz)                                             # [1,Dy,Dx,Dz] CUDA
            d01_recs.append(rec_chw)
        d01_rec = torch.stack(d01_recs, dim=0).view(T, 1, Dy, Dx, Dz)                                    # CUDA
        d_rec   = dens_from01(d01_rec)                                                                   # CUDA

        bpp_val = self._last_bits / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        mse  = F.mse_loss(d_rec, d)
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr

    # ------------------------------ NVENC/NVDEC core ------------------------------
    def _ensure_codec(self, W: int, H: int):
        if self._enc is not None and self._enc_w == W and self._enc_h == H:
            return
        self._enc = None; self._dec = None

        codec_enum_map = {
            'h264': nvc.cudaVideoCodec.H264,
            'hevc': nvc.cudaVideoCodec.HEVC,
            'av1' : nvc.cudaVideoCodec.AV1,
        }
        if self.codec not in codec_enum_map:
            raise ValueError(f"Unsupported nv_codec '{self.codec}'")
        codec_enum = codec_enum_map[self.codec]

        self._enc = nvc.CreateEncoder(
            width=W, height=H,
            fmt=self.pix_fmt,                 # "YUV444"
            usecpuinputbuffer=False,
            codec=self.codec,                 # "hevc" recommended for 4:4:4
            preset=str(self.preset),
            profile=str(self.profile),        # must be 4:4:4-capable
            rc="vbr",
            constqp=str(self.constqp),
            fps=str(self.fps),
            gop=str(self.gop),
            bf="0", b_adapt="0",              # optional but good for 1:1 cadence
        )

        self._dec = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=codec_enum,
            usedevicememory=True,
            maxwidth=W, maxheight=H,
            outputColorType=nvc.OutputColorType.RGB,   # decoder returns interleaved RGB on GPU
        )

        self._enc_w, self._enc_h = W, H

    def _nv_roundtrip_rgb(self, canv_pad_cuda: torch.Tensor) -> torch.Tensor:
        """
        Input:  [T,3,H,W] float in [0,1] (CUDA)
        Output: [T,3,H,W] float in [0,1] (CUDA)
        TEMP path: treat R,G,B as Y,U,V planes for YUV444 to validate device-input path.
                (Visuals will be off; this is only to unblock the encoder data path.)
        Encoder must be created with fmt="YUV444" and a 4:4:4-capable profile (e.g., "high_444" for HEVC).
        Decoder uses outputColorType=RGB so frames arrive as interleaved RGB on GPU.
        """

        # ----- CAI plane & AppFrame for YUV444: each plane is [H,W,1] uint8 -----
        class AppCAI:
            def __init__(self, t_hwc1_u8: torch.Tensor):
                # Expect [H,W,1] uint8 contiguous CUDA tensor
                assert (t_hwc1_u8.is_cuda and t_hwc1_u8.dtype == torch.uint8 and
                        t_hwc1_u8.is_contiguous() and t_hwc1_u8.ndim == 3 and t_hwc1_u8.shape[2] == 1)
                itemsize = t_hwc1_u8.element_size()  # 1
                self.__cuda_array_interface__ = {
                    "shape": (int(t_hwc1_u8.shape[0]), int(t_hwc1_u8.shape[1]), 1),  # (H,W,1)
                    "strides": (int(t_hwc1_u8.stride(0) * itemsize),
                                int(t_hwc1_u8.stride(1) * itemsize),
                                int(t_hwc1_u8.stride(2) * itemsize)),
                    "typestr": "|u1",
                    "data": (int(t_hwc1_u8.data_ptr()), False),
                    "version": 3,
                }
                self._keep = t_hwc1_u8  # keep alive

        class AppFrameYUV444:
            def __init__(self, Y_hwc1: torch.Tensor, U_hwc1: torch.Tensor, V_hwc1: torch.Tensor):
                self._planes = [AppCAI(Y_hwc1), AppCAI(U_hwc1), AppCAI(V_hwc1)]
            def cuda(self):
                return self._planes

        T, _, H, W = canv_pad_cuda.shape
        self._ensure_codec(W, H)

        rec_list = []
        total_bits = 0

        for t in range(T):
            # [3,H,W] float -> uint8 on CUDA
            rgb_u8 = (canv_pad_cuda[t].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).contiguous()

            # TEMP: map R,G,B -> Y,U,V (each plane must be [H,W,1] per your encoder’s expectation)
            Y_hwc1 = rgb_u8[0].unsqueeze(-1).contiguous()  # [H,W,1]
            U_hwc1 = rgb_u8[1].unsqueeze(-1).contiguous()  # [H,W,1]
            V_hwc1 = rgb_u8[2].unsqueeze(-1).contiguous()  # [H,W,1]

            # Device-input encode via AppFrameYUV444 (3 CAI planes shaped [H,W,1])
            bitstream = self._enc.Encode(AppFrameYUV444(Y_hwc1, U_hwc1, V_hwc1))
            if bitstream:
                total_bits += len(bitstream) * 8
                pkt = nvc.PacketData(bitstream)
                for frame in self._dec.Decode(pkt):
                    frm = torch.from_dlpack(frame)  # GPU zero-copy (interleaved RGB)
                    if frm.ndim == 3 and frm.shape[-1] == 3:
                        rec_list.append(frm.permute(2, 0, 1).contiguous())  # [3,H,W]
                    elif frm.ndim == 3 and frm.shape[0] == 3:
                        rec_list.append(frm)
                    else:
                        raise RuntimeError(f"Unexpected decoded shape {tuple(frm.shape)}")

        # Flush & drain
        tail = self._enc.EndEncode()
        if tail:
            total_bits += len(tail) * 8
            pkt = nvc.PacketData(tail)
            for frame in self._dec.Decode(pkt):
                frm = torch.from_dlpack(frame)
                if frm.ndim == 3 and frm.shape[-1] == 3:
                    rec_list.append(frm.permute(2, 0, 1).contiguous())
                elif frm.ndim == 3 and frm.shape[0] == 3:
                    rec_list.append(frm)
                else:
                    raise RuntimeError(f"Unexpected decoded shape {tuple(frm.shape)}")

        if len(rec_list) < T:
            raise RuntimeError(f"Decoded fewer frames ({len(rec_list)}) than encoded ({T}).")

        rec = torch.stack(rec_list[:T], dim=0).to(torch.float32) / 255.0
        self._last_bits = int(total_bits)
        return rec

class AV1VideoCodecWrapper(nn.Module):
    """
    PyAV (libaom-av1) wrapper. Do as much as possible on CUDA:
      CUDA: normalize -> pack -> pad
      CPU : encode/decode (PyAV)
      CUDA: crop -> unpack -> de-normalize (+ metrics)
    """
    def __init__(self, cfg_codec, device="cuda"):
        super().__init__()
        self.device       = torch.device(device)
        self.in_channels  = int(cfg_codec.in_channels)
        self.packing_mode = cfg_codec.packing_mode
        self.align        = int(getattr(cfg_codec, "align", DCVC_ALIGN))
        self.quant_mode   = cfg_codec.quant_mode
        self.global_range = tuple(cfg_codec.global_range)

        # AV1 knobs
        self.crf     = int(getattr(cfg_codec, "crf", cfg_codec.crf))
        self.preset  = str(getattr(cfg_codec, "preset", "4"))        # libaom "cpu-used"
        self.gop     = int(getattr(cfg_codec, "gop", cfg_codec.gop))
        self.fps     = int(getattr(cfg_codec, "fps", 30))
        self.pix_fmt = str(getattr(cfg_codec, "pix_fmt", "yuv444p"))

        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes
        self._last_bits = 0

    # ------------------------------ feature planes ------------------------------
    @torch.no_grad()
    def forward(self, frames: torch.Tensor):
        """
        frames: [T,C,H,W] float (any device). Returns recon [T,C,H,W] on input device.
        """
        assert frames.dim() == 4 and frames.shape[1] == self.in_channels
        in_dev = frames.device
        T, C, H, W = frames.shape

        # CUDA path: normalize, pack, pad
        x = frames.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)   # GPU
        canv_pad, orig_size = self.pack_fn(x01, align=self.align, mode=self.packing_mode)               # GPU [T,3,Hp,Wp]
        H2, W2 = orig_size
        Hp, Wp = canv_pad.shape[-2:]

        # CPU path: AV1 roundtrip
        canv_cpu = canv_pad.detach().to('cpu', copy=True)                                               # CPU
        rec_canv_cpu = self._av1_roundtrip(canv_cpu, fps=self.fps, gop=self.gop,
                                           crf=self.crf, cpu_used=self.preset, pix_fmt=self.pix_fmt)   # CPU [T,3,Hp,Wp]

        # Back to CUDA: crop, unpack, de-normalize, metrics
        rec_canv = rec_canv_cpu.to(in_dev, non_blocking=True)                                           # GPU
        rec_canv = rec_canv[..., :H2, :W2]                                                              # [T,3,H2,W2]
        rec01    = self.unpack_fn(rec_canv, C, (H2, W2), mode=self.packing_mode)                        # [T,C,H,W] GPU
        recon    = (rec01 * scale + c_min).to(torch.float32)                                            # GPU

        # bpp & psnr on GPU
        bpp_val = self._last_bits / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])
            psnr = tetrirf_utils.mse2psnr_with_peak(F.mse_loss(recon, x), peak=peak)
        else:
            raise NotImplementedError("Only 'global' quant_mode supported.")

        return recon, bpp, psnr

    # --------------------------------- density ----------------------------------
    @torch.no_grad()
    def forward_density(self, dens_seq: torch.Tensor):
        """
        dens_seq: [T,1,Dy,Dx,Dz] float (any device) -> recon same shape (on input device).
        """
        assert dens_seq.dim() == 5 and dens_seq.shape[1] == 1
        in_dev = dens_seq.device
        T, _, Dy, Dx, Dz = dens_seq.shape

        # CUDA: map to [0,1], tile to canvases, pad
        d   = dens_seq.to(dtype=torch.float32, device=in_dev, non_blocking=True)
        d01 = dens_to01(d)                                                                              # [T,1,Dy,Dx,Dz] GPU

        # tile per t on GPU
        mono_list = []
        Hc = Wc = None
        for t in range(T):
            chw  = d01[t].view(1, Dy, Dx, Dz)                                                           # [1,C,H,W] GPU
            mono, (Hct, Wct) = tile_1xCHW(chw)                                                          # [Hc,Wc] GPU
            if Hc is None:
                Hc, Wc = Hct, Wct
            mono_list.append(mono)
        mono_stack = torch.stack(mono_list, dim=0)                                                      # [T,Hc,Wc] GPU
        canv       = mono_stack.unsqueeze(1).repeat(1, 3, 1, 1)                                         # [T,3,Hc,Wc] GPU

        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        canv_pad = F.pad(canv, (0, pad_w, 0, pad_h), mode="replicate")                                  # [T,3,Hp,Wp] GPU
        Hp, Wp = canv_pad.shape[-2:]

        # CPU: AV1 roundtrip
        rec_canv_cpu = self._av1_roundtrip(canv_pad.detach().to('cpu', copy=True),
                                           fps=self.fps, gop=self.gop, crf=self.crf,
                                           cpu_used=self.preset, pix_fmt=self.pix_fmt)                  # CPU

        # Back to CUDA: crop, untile, de-normalize, metrics
        rec_canv = rec_canv_cpu.to(in_dev, non_blocking=True)[..., :Hc, :Wc]                            # [T,3,Hc,Wc] GPU

        d01_recs = []
        for t in range(T):
            mono_rec = rec_canv[t, 0]                                                                   # [Hc,Wc] GPU
            rec_chw  = untile_to_1xCHW(mono_rec, Dy, Dx, Dz)                                            # [1,Dy,Dx,Dz] GPU
            d01_recs.append(rec_chw)
        d01_rec = torch.stack(d01_recs, dim=0).view(T, 1, Dy, Dx, Dz)                                   # GPU
        d_rec   = dens_from01(d01_rec)                                                                  # GPU

        bpp_val = self._last_bits / float(T * Hp * Wp)
        bpp     = torch.tensor(bpp_val, device=in_dev, dtype=torch.float32)
        mse  = F.mse_loss(d_rec, d)
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr

    # ------------------------------ PyAV AV1 core ------------------------------
    def _av1_roundtrip(
        self,
        canvases: torch.Tensor,   # [T,3,H,W] float in [0,1], **CPU**
        fps: int, gop: int, crf: int, cpu_used: str, pix_fmt: str
    ) -> torch.Tensor:
        """
        Encode canvases -> bytes (IVF in-memory) -> decode -> [T,3,H,W] float CPU.
        Accumulates total bitstream length in self._last_bits.
        """
        assert canvases.device.type == 'cpu', "PyAV expects CPU tensors"
        assert canvases.dim() == 4 and canvases.shape[1] == 3
        T, _, H, W = canvases.shape

        out_buf = io.BytesIO()
        oc = av.open(out_buf, mode='w', format='ivf')

        stream = oc.add_stream('libaom-av1', rate=fps)
        stream.width  = W
        stream.height = H
        stream.pix_fmt = pix_fmt
        stream.time_base = Fraction(1, fps)
        stream.codec_context.time_base = Fraction(1, fps)
        stream.gop_size = gop

        opts = {
            'crf': str(crf),
            'cpu-used': str(cpu_used),
            'enable-chroma-deltaq': '0',
        }
        for k, v in opts.items():
            stream.codec_context.options[k] = v

        # Encode
        for t in range(T):
            # ensure contiguous uint8 HxWx3
            frm = (canvases[t].clamp(0,1).permute(1,2,0).contiguous().numpy() * 255.0 + 0.5).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(frm, format='rgb24')
            frame = frame.reformat(format=pix_fmt)
            for packet in stream.encode(frame):
                oc.mux(packet)
        for packet in stream.encode(None):
            oc.mux(packet)
        oc.close()

        bitstream = out_buf.getvalue()
        self._last_bits = len(bitstream) * 8

        # Decode
        in_buf = io.BytesIO(bitstream)
        ic = av.open(in_buf, mode='r')

        rec_frames: List[torch.Tensor] = []
        for frame in ic.decode(video=0):
            rgb = frame.to_ndarray(format='rgb24')                                   # HxWx3 uint8
            rec_frames.append(torch.from_numpy(rgb).permute(2,0,1).float() / 255.0)  # CPU
        ic.close()

        if len(rec_frames) != T:
            raise RuntimeError(f"Decoded {len(rec_frames)} frames, expected {T}")

        out = torch.stack(rec_frames, dim=0)                                         # [T,3,H,W] CPU
        if out.shape[-2:] != (H, W):
            raise RuntimeError(f"AV1 roundtrip size mismatch: in ({H},{W}) vs out {tuple(out.shape[-2:])}")
        return out


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


class DCVCImageCodecWrapper(torch.nn.Module):
    '''
    Autograd: 
        - Calling the codec in .eval() mode does not disable gradients; 
        - it only toggles internal behaviors like dropout/batchnorm. 
        - Avoid torch.no_grad() around the codec if you want gradients.
    '''
    def __init__(
            self, cfg_dcvc, device, 
            weight_path="/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar"):
        super().__init__()
        self.device = device
        self.weight_path = weight_path

        self.codec_wrapper = DCVCImageCodec(
            weight_path=self.weight_path,
            require_grad=False
            )

        self.packing_mode = cfg_dcvc.packing_mode
        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes
        self.qp = cfg_dcvc.dcvc_qp
        self.quant_mode = cfg_dcvc.quant_mode
        self.global_range = cfg_dcvc.global_range
        self.align = DCVC_ALIGN
        self.in_channels = cfg_dcvc.in_channels
        self.cfg_dcvc = cfg_dcvc
        self.use_amp = cfg_dcvc.use_amp
        self.amp_dtype = torch.float16  # keep TriPlane in fp32, only DCVC in fp16

        self.use_gradbpp_est = getattr(cfg_dcvc, "gradbpp", False)
        self.bpp_estimator = None
        if self.use_gradbpp_est:
            self.bpp_estimator = DMCI()  
            self.bpp_estimator.load_state_dict(get_state_dict(self.weight_path))
            self.bpp_estimator.update(0.12)
            self.bpp_estimator = self.bpp_estimator.to(self.device)
            # Freeze params so grads flow to input only
            for p in self.bpp_estimator.parameters():
                p.requires_grad_(False)
            self.bpp_estimator.eval()
            # self.bpp_estimator.half()

    def compute_bpp(
        self,
        image_tensor: torch.Tensor   # (B,3,H,W) in [0,1]
    ) -> torch.Tensor:
        """
        Returns differentiable bpp (scalar tensor) w.r.t. image_tensor.
        """
        assert self.bpp_estimator is not None, "BPP estimator not initialized"
        T, C, H, W = image_tensor.shape
        x_in = rgb2ycbcr(image_tensor)
        bpp = self.bpp_estimator.estimate_bpp(x_in, qp=self.qp)
        return bpp
    
    def estimate_bpp_only(self, planes_1xCHW: torch.Tensor) -> torch.Tensor:
        """
        planes_1xCHW: [1,C,H,W] float on any device, raw-domain (same input you pass to .forward())
        Returns differentiable bpp scalar (bits per padded pixel).
        """
        assert self.use_gradbpp_est and self.bpp_estimator is not None, "BPP estimator not enabled"
        dev = planes_1xCHW.device
        # 1) normalize to [0,1]
        x01, _, _ = normalize_planes(planes_1xCHW, mode=self.quant_mode, global_range=self.global_range)  # [1,C,H,W] on dev
        # 2) pack + pad to align (what codec sees)
        canv_pad, _ = self.pack_fn(x01, align=self.align, mode=self.packing_mode)  # [1,3,Hp,Wp]
        # 3) DMCI estimator (works on [0,1] RGB-like)
        return self.compute_bpp(canv_pad)  # differentiable scalar tensor

    def estimate_bpp_density_only(self, density_1x1: torch.Tensor) -> torch.Tensor:
        """
        density_1x1: [1,1,Dy,Dx,Dz] float on any device.
        Returns differentiable bpp scalar for density.
        """
        assert self.use_gradbpp_est and self.bpp_estimator is not None, "BPP estimator not enabled"
        dev = density_1x1.device
        # map to [0,1]
        d01 = dens_to01(density_1x1)                             # [1,1,Dy,Dx,Dz]
        Dy, Dx, Dz = d01.shape[2:]
        chw = d01.view(1, Dy, Dx, Dz)                            # [1,C,H,W] with C=Dy
        mono, (Hc, Wc) = tile_1xCHW(chw)                         # [Hc,Wc]
        canv = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)    # [1,3,Hc,Wc]
        # pad to align
        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        canv_pad = F.pad(canv, (0, pad_w, 0, pad_h), mode="replicate")  # [1,3,Hp,Wp]
        # entropy estimator on [0,1]
        return self.compute_bpp(canv_pad)                        # differentiable scalar tensor

    def forward(self, frame: torch.Tensor):
        """
        Args:
            frame: [1, C, H, W] float on any device.
        Returns:
            recon [1,C,H,W] float32 on input device
            bpp   scalar tensor (bits per padded pixel)
            psnr  scalar tensor (raw-domain, global peak)
        """
        x = frame
        assert x.shape[1] == self.in_channels, f"expected C={self.in_channels}, got {x.shape[1]}"

        # Quantize feature planes to 0-1
        x01, c_min, scale = normalize_planes(
            x, mode=self.quant_mode, global_range=self.global_range)

        # Pack the quantized feature planes to 3 channels (and padding)
        y_pad, orig_size = self.pack_fn(x01, mode=self.packing_mode)
        H2p, W2p = y_pad.shape[-2:]
        y_pad = y_pad.to(device=self.device)

        # Optimize memory layout
        try:
            y_pad = y_pad.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass

        # Run DCVC coding
        y_half = y_pad.to(torch.float16)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):

                enc_result = self.codec_wrapper.compress(y_half, self.qp)
                bits = self.codec_wrapper.measure_size(enc_result, self.qp)
                dec_result  = self.codec_wrapper.decompress(enc_result)
                x_hat_half = dec_result[..., :H2p, :W2p]

                # mimic training bpp for consistency (bits / padded pixels)
                bits = self.codec_wrapper.measure_size(enc_result, self.qp)
                bpp = torch.tensor(float(bits) / float(H2p * W2p), device=y_pad.device, dtype=torch.float32)

        # Exit AMP, cast to fp32 for numerics and to match TriPlane later
        x_hat32 = x_hat_half.to(torch.float32)

        # Unpack the reconstructed feature planes and crop to ori size
        rec01 = self.unpack_fn(x_hat32, x01.shape[1], orig_size, mode=self.packing_mode)

        # Rescale to original range
        recon = (rec01 * scale + c_min).to(torch.float32) 

        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])  # = 40.0
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            plane_psnr = tetrirf_utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            raise NotImplementedError("mse2psnr_with_peak only implemented for global mode")

        return recon, bpp, plane_psnr

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
        _, _, Dy, Dx, Dz = density_1x1.shape

        # Map to [0,1], pack as mono canvas, then 3ch by repetition
        d01 = dens_to01(density_1x1)                # [1,1,Dy,Dx,Dz]
        d01_chw = d01.view(1, Dy, Dx, Dz)                 # [1,C,H,W] with C=Dy,H=Dx,W=Dz
        mono, (Hc, Wc) = tile_1xCHW(d01_chw)        # [Hc,Wc]
        y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)# [1,3,Hc,Wc]

        # Align to DCVC stride
        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode="replicate").to(self.device)

        # AMP + codec forward
        y_half = y_pad.to(torch.float16).contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            enc_result = self.codec_wrapper.compress(y_half, self.qp)
            dec_result = self.codec_wrapper.decompress(enc_result)
            x_hat_half = dec_result[..., :Hc, :Wc]

            # mimic training bpp for consistency (bits / padded pixels)
            Hp, Wp = y_pad.shape[-2:]
            bits = self.codec_wrapper.measure_size(enc_result, self.qp)
            bpp = torch.tensor(float(bits) / float(Hp * Wp), device=y_pad.device, dtype=torch.float32)

        x_hat = x_hat_half.to(torch.float32)

        # Take one channel back to mono canvas (any of the three; they should match)
        mono_rec = x_hat[:, 0].squeeze(0)                 # [Hc,Wc] fp32

        # Untile → [1,C,H,W] → [1,1,Dy,Dx,Dz]
        d01_rec_chw = untile_to_1xCHW(mono_rec, Dy, Dx, Dz)   # [1,Dy,Dx,Dz]
        d_rec = dens_from01(d01_rec_chw).view(1, 1, Dy, Dx, Dz)

        # PSNR in raw density domain (peak = 35)
        mse = F.mse_loss(d_rec, density_1x1)
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        # ---- NEW: switch bpp source ----
        if self.use_gradbpp_est and self.bpp_estimator is not None:
            # Use packed/padded canvas (what the codec sees) for an apples-to-apples bpp
            # y_pad is (B=1, 3, Hp, Wp) in [0,1].
            bpp_est = self.compute_bpp(image_tensor=y_pad)
            bpp = bpp_est.to(y_pad.device, dtype=torch.float32)

        return d_rec, bpp, psnr
    

class JPEGImageCodecWrapper(torch.nn.Module):
    """
    CPU JPEG wrapper with the same public interface as DCVCImageCodec:

        forward(frame: [1,C,H,W]) -> (recon, bpp, psnr)
        forward_density(dens: [1,1,Dy,Dx,Dz]) -> (recon, bpp, psnr)

    Non-differentiable by nature; use with STE in the caller.
    """
    def __init__(self, cfg_jpeg, device="cuda"):
        super().__init__()
        self.device       = torch.device(device)
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
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)

        # 2) Pack to BGR image [1,3,H2p,W2p] with internal padding to 'align'
        y_pad, orig_hw = pack_planes_to_rgb(x01, align=self.align, mode=self.packing_mode)
        Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]
        H2, W2 = orig_hw

        # 3) JPEG round-trip on CPU
        #    (OpenCV uses BGR; our tensor is [1,3,H,W] with arbitrary channel semantics)
        y_cpu = y_pad[0].permute(1, 2, 0).contiguous().cpu().numpy()   # HxWx3 float01
        dec_f01_bgr, bits = jpeg_roundtrip_color(y_cpu, quality=self.quality)
        dec_t = torch.from_numpy(dec_f01_bgr).permute(2, 0, 1)[None]   # [1,3,H,W] float01

        # 4) Crop to original, unpack back to planes, and de-normalize
        dec_t = crop_from_align(dec_t, orig_hw)                       # [1,3,H2,W2]
        dec_t = dec_t.to(dev, non_blocking=True)    
        rec01 = unpack_rgb_to_planes(dec_t, x01.shape[1], orig_hw, mode=self.packing_mode)
        recon = (rec01 * scale + c_min).to(torch.float32)

        # 5) bpp over padded pixels (matches your DCVC convention)
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
        _, _, Dy, Dx, Dz = density_1x1.shape

        # (i) map to [0,1]
        d01 = dens_to01(density_1x1)                      # [1,1,Dy,Dx,Dz]

        # (ii) view as [1,C,H,W] with C=Dy, H=Dx, W=Dz and tile to mono
        d01_chw = d01.view(1, Dy, Dx, Dz)                  # [1,C,H,W]
        mono, (Hc, Wc) = tile_1xCHW(d01_chw)              # [Hc,Wc] float01

        # (iii) repeat mono → 3ch image and pad to align
        y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0) # [1,3,Hc,Wc]
        y_pad, _ = pad_to_align(y, align=self.align)      # [1,3,Hp,Wp]
        Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]

        # (iv) JPEG CPU round-trip
        y_cpu = y_pad[0].permute(1, 2, 0).contiguous().cpu().numpy()  # Hp×Wp×3 float01
        dec_f01_bgr, bits = jpeg_roundtrip_color(y_cpu, quality=self.quality)
        dec_t = torch.from_numpy(dec_f01_bgr).permute(2, 0, 1)[None]  # [1,3,Hp,Wp]
        dec_t = dec_t[..., :Hc, :Wc].to(dev, non_blocking=True)        # [1,3,Hc,Wc] on dev
        mono_rec = dec_t[0, 0]                                         # <<< [Hc, Wc] 2-D canvas
        d01_rec_chw = untile_to_1xCHW(mono_rec, Dy, Dx, Dz)           # [1,Dy,Dx,Dz]
        d_rec = dens_from01(d01_rec_chw).view(1, 1, Dy, Dx, Dz)       # [1,1,Dy,Dx,Dz] on dev

        # bpp over padded pixels
        bpp = torch.tensor(float(bits) / float(Hp * Wp), device=dev, dtype=torch.float32)

        # PSNR in raw density domain (peak = 35)
        mse = F.mse_loss(d_rec, density_1x1.to(d_rec.dtype))
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        return d_rec, bpp, psnr