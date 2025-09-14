import os
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from TeTriRF.lib.dvgo import DirectVoxGO
from TeTriRF.lib.dmpigo import DirectMPIGO

from TeTriRF.lib.dvgo_video import RGB_Net, RGB_SH_Net
from src.models.codec_wrapper import DCVCImageCodecWrapper, JPEGImageCodecWrapper


class STE_DVGO_Image(torch.nn.Module):
    def __init__(self, frameids, xyz_min, xyz_max, 
                 cfg=None, device='cuda'):
        super(STE_DVGO_Image, self).__init__()

        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.frameids = frameids
        self.cfg = cfg
        self.dvgos = nn.ModuleDict()
        self.viewbase_pe = cfg.fine_model_and_render.viewbase_pe
        self.fixed_frame = []
        self._initial_models()
        self.codec_type = cfg.codec.name

        if len(self.frameids) == 1:
            if self.codec_type == 'JPEGImageCodec':
                self.codec = JPEGImageCodecWrapper(self.cfg.codec, device)
            elif self.codec_type == 'DCVCImageCodec':
                self.codec = DCVCImageCodecWrapper(self.cfg.codec, device)
            else:
                raise NotImplementedError(f"Codec type {self.codec_type} not implemented.")
        else:
            raise RuntimeError("DCVCImageCodec only supports single frame input.")
          
        # --- Cache & refresh knobs ---
        self.codec_refresh_k = int(getattr(self.cfg.codec, "codec_refresh_k", 1))
        self.refresh_trigger_eps = float(getattr(self.cfg.codec, "refresh_trigger_eps", 0.0))
        self.bpp_refresh_k        = int(getattr(self.cfg.codec, "bpp_refresh_k", 1))  # NEW
        self._codec_cache_step = -1

        self._codec_cache = {ax: None for ax in ('xy','xz','yz')}          # detached recon (1,C,H,W)
        self._codec_cache_bpp = {ax: None for ax in ('xy','xz','yz')}      # float tensor
        self._codec_cache_psnr = {ax: None for ax in ('xy','xz','yz')}     # float tensor
        self._codec_cache_rawsnap = {ax: None for ax in ('xy','xz','yz')}  # snapshot of raw planes at cache time, for change detection
        
        self._dens_cache        = None   # [1,1,Dy,Dx,Dz] (detached)
        self._dens_cache_bpp    = None   # scalar tensor
        self._dens_cache_psnr   = None   # scalar tensor
        self._dens_cache_rawsnap= None   # [1,1,Dy,Dx,Dz] (detached)

    def forward(self, rays_o, rays_d, viewdirs, frame_ids, global_step=None, mode='feat', **render_kwargs):
        frame_ids_unique = torch.unique(frame_ids, sorted=True).cpu().int().numpy().tolist()
        assert len(frame_ids_unique) == 1, "Expect a single frame per batch"
        frameid = frame_ids_unique[0]
        dvgo = self.dvgos[str(frameid)]

        # Refresh cache if needed (runs DCVC at most every K steps)
        self._maybe_refresh_codec_cache(frameid, global_step)

        # Build param overrides using STE from cache
        param_map  = dict(dvgo.named_parameters())
        buffer_map = dict(dvgo.named_buffers())
        overrides = self._ste_overrides(frameid)

        param_overrides  = {k: v for k, v in overrides.items() if k in param_map}
        buffer_overrides = {k: v for k, v in overrides.items() if k in buffer_map}
        merged_params  = {**param_map,  **param_overrides}
        merged_buffers = {**buffer_map, **buffer_overrides}

        # Render with functional_call
        ret_frame = torch.func.functional_call(
            dvgo,
            {**merged_params, **merged_buffers},
            (rays_o, rays_d, viewdirs),
            {'shared_rgbnet': self.rgbnet, 'global_step': global_step, 'mode': mode, **render_kwargs}
        )

        # ----- BPP path: gradient-preserved every bpp_refresh_k steps -----
        use_grad_bpp = bool(getattr(self.cfg.codec, "gradbpp", False))
        step = int(global_step if global_step is not None else 0)
        do_grad_bpp = (use_grad_bpp and (self.bpp_refresh_k <= 1 or (step % self.bpp_refresh_k) == 0))

        if do_grad_bpp:
            # Differentiable bpp from CURRENT planes/density (slow path, but periodic)
            planes_now = self._gather_planes_one_frame(frameid)
            bpps_planes = [
                self.codec.estimate_bpp_only(planes_now['xy']),
                self.codec.estimate_bpp_only(planes_now['xz']),
                self.codec.estimate_bpp_only(planes_now['yz']),
            ]
            dens_now = self._gather_density_one_frame(frameid)
            dens_bpp = self.codec.estimate_bpp_density_only(dens_now)

            avg_bpp = (sum(bpps_planes) + dens_bpp) / 4.0  # 3 planes + density

            # For logging, still report cached PSNR (detached)
            psnr_by_axis = {ax: self._codec_cache_psnr[ax] for ax in ('xy','xz','yz')}
            psnr_by_axis['density'] = self._dens_cache_psnr
        else:
            # Fast path: use detached cached bpp (no gradient)
            bpps_planes = [self._codec_cache_bpp[ax] for ax in ('xy','xz','yz')]
            dens_bpp    = self._dens_cache_bpp

            # Safeguard in case cache not ready (shouldn’t happen after _maybe_refresh…)
            device = rays_o.device
            def _as_scalar(x):
                if x is None: return torch.tensor(0.0, device=device)
                return x if torch.is_tensor(x) else torch.tensor(float(x), device=device)
            bpps_planes = [_as_scalar(b) for b in bpps_planes]
            dens_bpp    = _as_scalar(dens_bpp)

            avg_bpp = (sum(bpps_planes) + dens_bpp) / 4.0  # 3 planes + density

            psnr_by_axis = {ax: self._codec_cache_psnr[ax] for ax in ('xy','xz','yz')}
            psnr_by_axis['density'] = self._dens_cache_psnr
        
        return ret_frame, avg_bpp, psnr_by_axis

    def _gather_planes_one_frame(self, frameid):
        k0 = self.dvgos[str(frameid)].k0
        return {
            'xy': getattr(k0, 'xy_plane'),  # (1,C,H,W) nn.Parameter
            'xz': getattr(k0, 'xz_plane'),
            'yz': getattr(k0, 'yz_plane'),
        }
    
    def _gather_density_one_frame(self, frameid):
        dvgo = self.dvgos[str(frameid)]
        return getattr(dvgo.density, 'grid')

    def _encode_decode_one_frame(self, frameid):
        """Functional encode-decode that returns recon planes for THIS frame only, w/o mutation."""
        planes = self._gather_planes_one_frame(frameid)  # dict of (C,H,W)
        recon_by_axis = {}
        bpp_by_axis   = {}
        psnr_by_axis  = {}

        for ax in ('xy','xz','yz'):
            # Pack (C,H,W)->image-like for your wrapper if needed (your codec already expects (T,C,H,W))
            x = planes[ax] # (1,C,H,W)
            recon, bpp, plane_psnr = self.codec(x)  # must be differentiable (no .detach() inside)
            recon_by_axis[ax] = recon        
            bpp_by_axis[ax]   = bpp
            psnr_by_axis[ax]  = plane_psnr

        return recon_by_axis, bpp_by_axis, psnr_by_axis

    def _invalidate_codec_cache(self):
        self._codec_cache_step = -1
        for ax in ('xy','xz','yz'):
            self._codec_cache[ax] = None
            self._codec_cache_bpp[ax] = None
            self._codec_cache_psnr[ax] = None
            self._codec_cache_rawsnap[ax] = None

        self._dens_cache         = None
        self._dens_cache_bpp     = None
        self._dens_cache_psnr    = None
        self._dens_cache_rawsnap = None


    @torch.no_grad()
    def _changed_too_much(self, cur: torch.Tensor, snap: torch.Tensor) -> bool:
        if snap is None or (cur.shape != snap.shape):
            return True
        num = (cur - snap).float().pow(2).sum()
        den = snap.float().pow(2).sum().clamp_min(1e-12)
        rel = (num / den).sqrt()
        return rel.item() > self.refresh_trigger_eps
    
    @torch.no_grad()
    def _planes_or_density_changed(self, planes: dict, density: torch.Tensor) -> bool:
        if self.refresh_trigger_eps <= 0:
            return False
        # planes
        for ax, p in planes.items():
            if self._changed_too_much(p, self._codec_cache_rawsnap[ax]):
                return True
        # density
        if self._changed_too_much(density, self._dens_cache_rawsnap):
            return True
        return False

    def _maybe_refresh_codec_cache(self, frameid, global_step):
        """Run DCVC if cache is stale; store *detached* recons/bpp for planes + density."""
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
            x = planes[ax]
            recon, bpp, plane_psnr = self.codec(x)        # same as before
            self._codec_cache[ax]       = recon.detach()
            self._codec_cache_bpp[ax]   = (bpp.detach() if torch.is_tensor(bpp)
                                           else torch.tensor(float(bpp), device=x.device))
            self._codec_cache_psnr[ax]  = (plane_psnr.detach() if torch.is_tensor(plane_psnr)
                                           else torch.tensor(float(plane_psnr), device=x.device))
            self._codec_cache_rawsnap[ax] = x.detach()

        # ---- NEW: density grid ----
        d = density
        d_recon, d_bpp, d_psnr = self.codec.forward_density(d)
        self._dens_cache          = d_recon.detach()
        self._dens_cache_bpp      = d_bpp.detach() if torch.is_tensor(d_bpp) else torch.tensor(float(d_bpp), device=d.device)
        self._dens_cache_psnr     = d_psnr.detach() if torch.is_tensor(d_psnr) else torch.tensor(float(d_psnr), device=d.device)
        self._dens_cache_rawsnap  = d.detach()

        self._codec_cache_step = int(global_step if global_step is not None else 0)

    def _ste_overrides(self, frameid):
        """Build STE substituted planes for functional_call from cached recons."""
        planes = self._gather_planes_one_frame(frameid)
        density = self._gather_density_one_frame(frameid)

        overrides = {}
        for ax, name in [('xy','k0.xy_plane'), ('xz','k0.xz_plane'), ('yz','k0.yz_plane')]:
            raw = planes[ax]                # (1,C,H,W) param
            rec = self._codec_cache[ax]     # (1,C,H,W) detached tensor
            assert rec is not None, "Codec cache is empty—call _maybe_refresh_codec_cache() first."
            # Straight-through estimator: forward==rec, grad wrt raw is identity
            ste = rec + (raw - raw.detach())
            overrides[name] = ste

        assert self._dens_cache is not None, "Density cache empty—refresh first."
        raw_d = density
        rec_d = self._dens_cache
        overrides['density.grid'] = rec_d + (raw_d - raw_d.detach())

        return overrides

    def _initial_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_kwargs = copy.deepcopy(self.cfg.fine_model_and_render)
        num_voxels = model_kwargs.pop('num_voxels')
        cfg_train = self.cfg.fine_train

        coarse_ckpt_path = None
        
        if len(cfg_train.pg_scale):
            num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

        for frameid in self.frameids:
            coarse_ckpt_path = os.path.join(self.cfg.basedir, self.cfg.expname, f'coarse_last_{frameid}.tar')
            if not os.path.isfile(coarse_ckpt_path):
                coarse_ckpt_path = None
            frameid = str(frameid)
            print(f'model create: frame{frameid}')

            #k0_config = {'factor': self.cfg.fine_model_and_render.plane_scale}
            if self.cfg.data.ndc:
                self.dvgos[frameid] = DirectMPIGO(
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    num_voxels=num_voxels, 
                    mask_cache_path=coarse_ckpt_path,  
                    **model_kwargs)
            else:
                self.dvgos[frameid] = DirectVoxGO(
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    num_voxels=num_voxels, 
                    mask_cache_path=coarse_ckpt_path, rgb_model = self.cfg.fine_model_and_render.RGB_model,
                    **model_kwargs)
            self.dvgos[frameid] = self.dvgos[frameid].to(device)

        if self.cfg.fine_model_and_render.RGB_model=='MLP':
            dim0 = (3+3*self.viewbase_pe*2) + self.cfg.fine_model_and_render.rgbnet_dim
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            self.rgbnet = RGB_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth)
        elif self.cfg.fine_model_and_render.RGB_model =='SH':
            dim0 = self.cfg.fine_model_and_render.rgbnet_dim
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            self.rgbnet = RGB_SH_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth, deg=2)
        
        print('*** models creation completed.',self.frameids)

    def get_kwargs(self):
        return {
            'frameids': self.frameids,
            'xyz_min': self.xyz_min,
            'xyz_max': self.xyz_max,
            'viewbase_pe': self.viewbase_pe,
        }

    def load_checkpoints(self):

        cfg = self.cfg
        ret = []

        for frameid in self.frameids:
            try:
                frameid = str(frameid)
                last_ckpt_path = os.path.join(cfg.basedir, cfg.ckptname, f'fine_last_{frameid}.tar')
                if not os.path.isfile(last_ckpt_path):
                    print(f"Frame {frameid}'s checkpoint doesn't exist")
                    # Always try to load the pre-trained triplane model
                    raise FileNotFoundError(f"Checkpoint for frame {frameid} not found at {last_ckpt_path}")

                # allowlist numpy._core.multiarray._reconstruct
                ckpt = torch.load(last_ckpt_path, weights_only=False)

                model_kwargs = ckpt['model_kwargs']
                if self.cfg.data.ndc:
                    self.dvgos[frameid] = DirectMPIGO(**model_kwargs)
                else:
                    self.dvgos[frameid] = DirectVoxGO(**model_kwargs)
                self.dvgos[frameid].load_state_dict(ckpt['model_state_dict'], strict=True)
                self.dvgos[frameid] = self.dvgos[frameid].cuda()
                print(f"Frame {frameid}'s checkpoint loaded.")
                ret.append(int(frameid))
            except Exception as e:
                print(f"Error loading checkpoint for frame {frameid}: {e}")

        try:
            beg = self.frameids[0]
            eend = self.frameids[-1]
            rgbnet_file = os.path.join(cfg.basedir, cfg.ckptname, f'rgbnet_{beg}_{eend}.tar')
            checkpoint =torch.load(rgbnet_file)
            self.rgbnet.load_state_dict(checkpoint['model_state_dict']) 
        except Exception as e:
            print(f"Error loading RGBNet checkpoint: {e}")
        return ret

    def save_checkpoints(self):
        cfg = self.cfg
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid = str(frameid)
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last_{frameid}.tar')
            ckpt = {
                'model_state_dict': self.dvgos[frameid].state_dict(),
                'model_kwargs': self.dvgos[frameid].get_kwargs(),
            }
            torch.save(ckpt, ckpt_path)
            print(f"Frame {frameid}'s checkpoint saved to {ckpt_path}")


        beg = self.frameids[0]
        eend = self.frameids[-1]

        if self.cfg.fine_model_and_render.dynamic_rgbnet:
            rgbnet_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
        else:
            rgbnet_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'rgbnet.tar')
        rgbnet_ckpt = {
            'model_state_dict': self.rgbnet.state_dict(),
            # Add any other necessary information to the checkpoint dictionary
            'model_kwargs': self.rgbnet.get_kwargs(),
        }
        torch.save(rgbnet_ckpt, rgbnet_ckpt_path)
        print(f"RGBNet checkpoint saved to {rgbnet_ckpt_path}")

    def scale_volume_grid(self, scale):
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid = str(frameid)
            self.dvgos[frameid].scale_volume_grid(scale)

    def density_total_variation_add_grad(self,  weight, dense_mode, frameids):
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        frameid = frame_ids_unique[0]
        frameid = str(frameid)
        self.dvgos[frameid].density_total_variation_add_grad(weight, dense_mode)

    def k0_total_variation_add_grad(self,  weight, dense_mode, frameids):
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        frameid = frame_ids_unique[0]
        frameid = str(frameid)
        self.dvgos[frameid].k0_total_variation_add_grad(weight, dense_mode)

    def compute_k0_l1_loss(self, frameids):
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        loss = 0
        N =0
        frameid = frame_ids_unique[0]
        if (str(frameid-1) in self.dvgos):
            frameid2 = str(frameid-1)
            # print(f"Compute k0 l1 loss between {frameid} and {frameid2}")
            if not self.dvgos[str(frameid)].k0.xy_plane.size() == self.dvgos[frameid2].k0.xy_plane.size():
                xy_plane, xz_plane, yz_plane = self.dvgos[frameid2].k0.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)

                loss += 2*F.l1_loss(self.dvgos[str(frameid)].k0.xy_plane, xy_plane)
                loss += 2*F.l1_loss(self.dvgos[str(frameid)].k0.xz_plane, xz_plane)
                loss += 2*F.l1_loss(self.dvgos[str(frameid)].k0.yz_plane, yz_plane)
                N=N+3
            else:
                loss += F.l1_loss(self.dvgos[str(frameid)].k0.xy_plane, self.dvgos[frameid2].k0.xy_plane)
                loss += F.l1_loss(self.dvgos[str(frameid)].k0.xz_plane, self.dvgos[frameid2].k0.xz_plane)
                loss += F.l1_loss(self.dvgos[str(frameid)].k0.yz_plane, self.dvgos[frameid2].k0.yz_plane)
                loss += 5*F.l1_loss(self.dvgos[str(frameid)].density.grid, self.dvgos[frameid2].density.grid)
                N+=4
        if str(frameid+1) in self.dvgos:
            frameid2 = str(frameid+1)
            # print(f"Compute k0 l1 loss between {frameid} and {frameid2}")
            loss += F.l1_loss(self.dvgos[str(frameid)].k0.xy_plane, self.dvgos[frameid2].k0.xy_plane)
            loss += F.l1_loss(self.dvgos[str(frameid)].k0.xz_plane, self.dvgos[frameid2].k0.xz_plane)
            loss += F.l1_loss(self.dvgos[str(frameid)].k0.yz_plane, self.dvgos[frameid2].k0.yz_plane)
            loss += 5*F.l1_loss(self.dvgos[str(frameid)].density.grid, self.dvgos[frameid2].density.grid)
            N+=4
        if N == 0:
            return loss
        time.sleep(0.5)
        return loss/N

    def update_occupancy_cache(self):
        res = []
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid = str(frameid)
            res.append(self.dvgos[frameid].update_occupancy_cache())
        return np.mean(res)
    
    def _set_requires_grad_module(self, module: torch.nn.Module, flag: bool):
        for p in module.parameters(recurse=True):
            p.requires_grad_(flag)