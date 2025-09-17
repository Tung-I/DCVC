import os
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_utils import (
    pack_planes_to_rgb, unpack_rgb_to_planes,
    normalize_planes, DCVC_ALIGN,
    dens_to01, dens_from01,
    tile_1xCHW, untile_to_1xCHW,
    pad_to_align, crop_from_align,
    jpeg_roundtrip_color,
    sandwich_planes_to_rgb, sandwich_rgb_to_planes,
)

from src.models.ste_dvgo_image import STE_DVGO_Image

class STE_DVGO_Sandwich_Image(STE_DVGO_Image):
    def __init__(self, frameids, xyz_min, xyz_max, cfg=None, device='cuda'):
        super().__init__(frameids, xyz_min, xyz_max, cfg=cfg, device=device)

        # Extra caches for codec canvases (per axis)
        self._canvas_codec = {ax: None for ax in ('xy','xz','yz')}   # y_codec [1,3,Hp,Wp] (detached)
        self._canvas_size  = {ax: None for ax in ('xy','xz','yz')}   # (H2_orig, W2_orig)

    def _maybe_refresh_codec_cache(self, frameid, global_step):
        """As before, but also cache y_codec + orig_size for each axis."""
        planes  = self._gather_planes_one_frame(frameid)
        density = self._gather_density_one_frame(frameid)

        need_refresh = (
            self._codec_cache_step < 0 or
            self.codec_refresh_k <= 1 or
            (global_step is not None and (global_step - self._codec_cache_step) >= self.codec_refresh_k) or
            self._planes_or_density_changed(planes, density)
        )
        if not need_refresh:
            return

        # ---- feature planes ----
        for ax in ('xy','xz','yz'):
            x = planes[ax]  # [1,C,H,W]
            recon, bpp, plane_psnr, y_pad, y_codec, orig_size = \
                self.codec.forward_with_canvases_axis(ax, x)

            self._codec_cache[ax]       = recon.detach()
            self._codec_cache_bpp[ax]   = bpp.detach() if torch.is_tensor(bpp) else torch.tensor(float(bpp), device=x.device)
            self._codec_cache_psnr[ax]  = plane_psnr.detach() if torch.is_tensor(plane_psnr) else torch.tensor(float(plane_psnr), device=x.device)
            self._codec_cache_rawsnap[ax] = x.detach()

            self._canvas_codec[ax] = y_codec.detach()
            self._canvas_size[ax]  = tuple(orig_size)

        # ---- density grid path unchanged ----
        d = density
        d_recon, d_bpp, d_psnr = self.codec.forward_density(d)
        self._dens_cache          = d_recon.detach()
        self._dens_cache_bpp      = d_bpp.detach() if torch.is_tensor(d_bpp) else torch.tensor(float(d_bpp), device=d.device)
        self._dens_cache_psnr     = d_psnr.detach() if torch.is_tensor(d_psnr) else torch.tensor(float(d_psnr), device=d.device)
        self._dens_cache_rawsnap  = d.detach()

        self._codec_cache_step = int(global_step if global_step is not None else 0)

    def _ste_overrides(self, frameid):
        """
        Build *codec-aware* overrides using:
           post( y_sur ), with y_sur = y_codec_cached.detach() + (y_pad_now - y_pad_now.detach())
        This lets gradients flow into pre/post (and tri-planes) while
        the forward value reflects true codec corruption.
        """
        planes = self._gather_planes_one_frame(frameid)

        overrides = {}
        for ax, name in [('xy','k0.xy_plane'), ('xz','k0.xz_plane'), ('yz','k0.yz_plane')]:
            raw = planes[ax]
            y_codec = self._canvas_codec[ax]
            orig_sz = self._canvas_size[ax]
            assert y_codec is not None and orig_sz is not None

            x01, c_min, scale = normalize_planes(raw, mode=self.codec.quant_mode, global_range=self.codec.global_range)

            # axis-aware pre
            y_pad_now, _ = self.codec.pack_axis(ax, x01, align=self.codec.align)

            # STE through codec
            y_sur = y_codec + (y_pad_now - y_pad_now.detach())

            # axis-aware post
            rec01 = self.codec.unpack_axis(ax, y_sur, x01.shape[1], orig_sz)

            # rescale to raw
            recon = (rec01 * scale + c_min).to(torch.float32)
            overrides[name] = recon


        # Density path unchanged (still cached STE)
        raw_d = self._gather_density_one_frame(frameid)
        rec_d = self._dens_cache
        overrides['density.grid'] = rec_d + (raw_d - raw_d.detach())

        return overrides

    def forward(self, rays_o, rays_d, viewdirs, frame_ids, global_step=None, mode='feat', **render_kwargs):
        # Refresh canvases if needed
        frame_ids_unique = torch.unique(frame_ids, sorted=True).cpu().int().numpy().tolist()
        assert len(frame_ids_unique) == 1
        frameid = frame_ids_unique[0]

        self._maybe_refresh_codec_cache(frameid, global_step)

        # Build codec-aware overrides (pre/post are active here)
        dvgo = self.dvgos[str(frameid)]
        param_map  = dict(dvgo.named_parameters())
        buffer_map = dict(dvgo.named_buffers())
        overrides  = self._ste_overrides(frameid)

        param_overrides  = {k: v for k, v in overrides.items() if k in param_map}
        buffer_overrides = {k: v for k, v in overrides.items() if k in buffer_map}

        ret_frame = torch.func.functional_call(
            dvgo,
            {**param_map, **buffer_map, **param_overrides, **buffer_overrides},
            (rays_o, rays_d, viewdirs),
            {'shared_rgbnet': self.rgbnet, 'global_step': global_step, 'mode': mode, **render_kwargs}
        )

        # ----- bpp: prefer differentiable estimator on current y_pad -----
        use_grad_bpp = bool(getattr(self.cfg.codec, "gradbpp", False))
        if use_grad_bpp and hasattr(self.codec, "compute_bpp"):
            planes_now = self._gather_planes_one_frame(frameid)
            bpps = []
            for ax in ('xy','xz','yz'):
                x01, _, _ = normalize_planes(planes_now[ax], mode=self.codec.quant_mode, global_range=self.codec.global_range)
                y_pad_now, _ = self.codec.pack_axis(ax, x01, align=self.codec.align)
                bpps.append(self.codec.compute_bpp(y_pad_now.to(self.codec.device, dtype=torch.float32)))
            dens_now = self._gather_density_one_frame(frameid)
            dens_bpp = self.codec.estimate_bpp_density_only(dens_now) if hasattr(self.codec, "estimate_bpp_density_only") else torch.tensor(0.0, device=y_pad_now.device)
            avg_bpp = (sum(bpps) + dens_bpp)
        else:
            # fallback: cached scalars
            bpps_planes = [self._codec_cache_bpp[ax] for ax in ('xy','xz','yz')]
            dens_bpp    = self._dens_cache_bpp
            device = rays_o.device
            def _as_scalar(x): 
                if x is None: return torch.tensor(0.0, device=device)
                return x if torch.is_tensor(x) else torch.tensor(float(x), device=device)
            bpps_planes = [_as_scalar(b) for b in bpps_planes]
            dens_bpp    = _as_scalar(dens_bpp)
            avg_bpp = (sum(bpps_planes) + dens_bpp)

        # For logging: still report cached PSNR (detached)
        psnr_by_axis = {ax: self._codec_cache_psnr[ax] for ax in ('xy','xz','yz')}
        psnr_by_axis['density'] = self._dens_cache_psnr

        return ret_frame, avg_bpp, psnr_by_axis
