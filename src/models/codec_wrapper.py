import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from src.utils.common import get_state_dict
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.models.dcvc_codec import DCVCImageCodec, DCVCVideoCodec

from src.models.model_utils import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    normalize_planes, DCVC_ALIGN,
    dens_to01, dens_from01,
    tile_1xCHW, untile_to_1xCHW,
    pad_to_align, crop_from_align,
    jpeg_roundtrip_color
)
import TeTriRF.lib.utils as tetrirf_utils


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
            self, cfg_dcvc, device):
        super().__init__()
        self.device = device

        self.codec_wrapper = DCVCImageCodec(
            weight_path="/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar", 
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