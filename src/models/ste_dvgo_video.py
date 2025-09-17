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
from src.models.codec_wrapper import DCVCVideoCodecWrapper, HEVCVideoCodecWrapper, AV1VideoCodecWrapper, PyNvVideoCodecWrapper

class STE_DVGO_Video(nn.Module):
    def __init__(self, frameids, xyz_min, xyz_max, cfg=None, device='cuda'):
        super().__init__()
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.frameids = list(map(int, frameids))  # support multi-frame
        self.cfg = cfg
        self.dvgos = nn.ModuleDict()
        self.viewbase_pe = cfg.fine_model_and_render.viewbase_pe
        self.fixed_frame = []
        self._initial_models()

        # ---- Codec (video only) ----
        if cfg.codec.name == 'DCVCVideoCodec':
            self.codec = DCVCVideoCodecWrapper(self.cfg.codec, device)
        elif cfg.codec.name == 'AV1VideoCodec':
            self.codec = AV1VideoCodecWrapper(self.cfg.codec, device)
        elif cfg.codec.name == 'PyNvVideoCodecWrapper':
            self.codec = PyNvVideoCodecWrapper(self.cfg.codec, device)
        elif cfg.codec.name == 'HEVCVideoCodec':
            self.codec = HEVCVideoCodecWrapper(self.cfg.codec, device)
        else:
            raise NotImplementedError(f"Unknown codec {cfg.codec.name}")

        # ---- Cache policy ----
        self.codec_refresh_k     = int(getattr(self.cfg.codec, "codec_refresh_k", 1))
        self.refresh_trigger_eps = float(getattr(self.cfg.codec, "refresh_trigger_eps", 0.0))
        self._codec_cache_step   = -1

        # Per-plane, per-frame caches
        self._codec_cache         = {ax: {fid: None for fid in self.frameids} for ax in ('xy','xz','yz')}  # recon (1,C,H,W)
        self._codec_cache_bpp     = {ax: {fid: None for fid in self.frameids} for ax in ('xy','xz','yz')}  # scalar tensor
        self._codec_cache_psnr    = {ax: {fid: None for fid in self.frameids} for ax in ('xy','xz','yz')}  # scalar tensor
        self._codec_cache_rawsnap = {ax: {fid: None for fid in self.frameids} for ax in ('xy','xz','yz')}  # snapshot at cache time

        # Density caches per-frame
        self._dens_cache          = {fid: None for fid in self.frameids}   # [1,1,Dy,Dx,Dz]
        self._dens_cache_bpp      = {fid: None for fid in self.frameids}   # scalar tensor
        self._dens_cache_psnr     = {fid: None for fid in self.frameids}   # scalar tensor
        self._dens_cache_rawsnap  = {fid: None for fid in self.frameids}   # [1,1,Dy,Dx,Dz]

        # Shared RGB head (same as image class)
        if self.cfg.fine_model_and_render.RGB_model == 'MLP':
            dim0 = (3 + 3*self.viewbase_pe*2) + self.cfg.fine_model_and_render.rgbnet_dim
            rw = self.cfg.fine_model_and_render.rgbnet_width
            rd = self.cfg.fine_model_and_render.rgbnet_depth
            self.rgbnet = RGB_Net(dim0=dim0, rgbnet_width=rw, rgbnet_depth=rd)
        else:
            dim0 = self.cfg.fine_model_and_render.rgbnet_dim
            rw = self.cfg.fine_model_and_render.rgbnet_width
            rd = self.cfg.fine_model_and_render.rgbnet_depth
            self.rgbnet = RGB_SH_Net(dim0=dim0, rgbnet_width=rw, rgbnet_depth=rd, deg=2)

    # -------------------------------------------------------------------------
    # Forward (assumes one frame_id in the batch — identical contract)
    # -------------------------------------------------------------------------
    def forward(self, rays_o, rays_d, viewdirs, frame_ids, global_step=None, mode='feat', **render_kwargs):
        frame_ids_unique = torch.unique(frame_ids, sorted=True).cpu().int().numpy().tolist()
        assert len(frame_ids_unique) == 1, "Expect a single frame per batch"
        frameid = int(frame_ids_unique[0])
        dvgo = self.dvgos[str(frameid)]

        # Refresh segment cache if needed (runs DCVC video over all frames when refreshing)
        self._maybe_refresh_codec_cache(frameid, global_step)

        # Build param overrides using STE from per-frame cache
        param_map  = dict(dvgo.named_parameters())
        buffer_map = dict(dvgo.named_buffers())

        overrides = self._ste_overrides(frameid)

        param_overrides  = {k: v for k, v in overrides.items() if k in param_map}
        buffer_overrides = {k: v for k, v in overrides.items() if k in buffer_map}
        merged_params  = {**param_map,  **param_overrides}
        merged_buffers = {**buffer_map, **buffer_overrides}

        # Render
        ret_frame = torch.func.functional_call(
            dvgo,
            {**merged_params, **merged_buffers},
            (rays_o, rays_d, viewdirs),
            {'shared_rgbnet': self.rgbnet, 'global_step': global_step, 'mode': mode, **render_kwargs}
        )

        # BPP & diagnostics — *per current frame*, same formula as image class
        bpps_planes = [self._codec_cache_bpp[ax][frameid] for ax in ('xy','xz','yz')]
        # guard & move to right device
        bpps_planes = [b if (torch.is_tensor(b) and b.device == rays_o.device)
                       else torch.tensor(float(b or 0.0), device=rays_o.device) for b in bpps_planes]
        d_bpp = self._dens_cache_bpp[frameid]
        d_bpp = d_bpp if (torch.is_tensor(d_bpp) and d_bpp.device == rays_o.device) \
                else torch.tensor(float(d_bpp or 0.0), device=rays_o.device)

        avg_bpp = (sum(bpps_planes) + d_bpp)

        psnr_by_axis = {ax: (self._codec_cache_psnr[ax][frameid]
                             if (torch.is_tensor(self._codec_cache_psnr[ax][frameid]) and
                                 self._codec_cache_psnr[ax][frameid].device == rays_o.device)
                             else torch.tensor(float(self._codec_cache_psnr[ax][frameid] or 0.0),
                                               device=rays_o.device))
                        for ax in ('xy','xz','yz')}
        p = self._dens_cache_psnr[frameid]
        psnr_by_axis['density'] = p if (torch.is_tensor(p) and p.device == rays_o.device) \
                                  else torch.tensor(float(p or 0.0), device=rays_o.device)

        return ret_frame, avg_bpp, psnr_by_axis

    # -------------------------------------------------------------------------
    # Gatherers (unchanged)
    # -------------------------------------------------------------------------
    def _gather_planes_one_frame(self, frameid):
        k0 = self.dvgos[str(frameid)].k0
        return {
            'xy': getattr(k0, 'xy_plane'),
            'xz': getattr(k0, 'xz_plane'),
            'yz': getattr(k0, 'yz_plane'),
        }

    def _gather_density_one_frame(self, frameid):
        dvgo = self.dvgos[str(frameid)]
        return getattr(dvgo.density, 'grid')

    # -------------------------------------------------------------------------
    # Cache utils
    # -------------------------------------------------------------------------
    def _invalidate_codec_cache(self):
        self._codec_cache_step = -1
        for ax in ('xy','xz','yz'):
            for fid in self.frameids:
                self._codec_cache[ax][fid] = None
                self._codec_cache_bpp[ax][fid] = None
                self._codec_cache_psnr[ax][fid] = None
                self._codec_cache_rawsnap[ax][fid] = None
        for fid in self.frameids:
            self._dens_cache[fid]         = None
            self._dens_cache_bpp[fid]     = None
            self._dens_cache_psnr[fid]    = None
            self._dens_cache_rawsnap[fid] = None

    @torch.no_grad()
    def _changed_too_much(self, cur: torch.Tensor, snap: torch.Tensor) -> bool:
        if snap is None or (cur.shape != snap.shape):
            return True
        num = (cur - snap).float().pow(2).sum()
        den = snap.float().pow(2).sum().clamp_min(1e-12)
        rel = (num / den).sqrt()
        return rel.item() > self.refresh_trigger_eps

    @torch.no_grad()
    def _frame_changed(self, frameid: int) -> bool:
        """Check only the requested frame for minimal change logic."""
        if self.refresh_trigger_eps <= 0:
            return False
        planes = self._gather_planes_one_frame(frameid)
        for ax, p in planes.items():
            if self._changed_too_much(p, self._codec_cache_rawsnap[ax][frameid]):
                return True
        density = self._gather_density_one_frame(frameid)
        if self._changed_too_much(density, self._dens_cache_rawsnap[frameid]):
            return True
        return False

    def _maybe_refresh_codec_cache(self, frameid, global_step):
        """
        If requested frame is stale / first time / step gap exceeded:
        run the **video codec once over all frames** to refresh the whole segment cache.
        """
        need_refresh = (
            self._codec_cache_step < 0 or
            self.codec_refresh_k <= 1 or
            (global_step is not None and (global_step - self._codec_cache_step) >= self.codec_refresh_k) or
            self._frame_changed(frameid)
        )
        if not need_refresh:
            return

        # ---------- Feature planes (per axis) ----------
        # Build [T,C,H,W] for each axis in the configured frame order
        planes_by_axis = {ax: [] for ax in ('xy','xz','yz')}
        with torch.no_grad():
            for fid in self.frameids:
                p = self._gather_planes_one_frame(fid)
                for ax in ('xy','xz','yz'):
                    planes_by_axis[ax].append(p[ax].detach())
            for ax in ('xy','xz','yz'):
                planes_by_axis[ax] = torch.cat(planes_by_axis[ax], dim=0)  # [T,C,H,W]

        # Encode/decode once per axis using the **video** wrapper
        for ax in ('xy','xz','yz'):
            xseg = planes_by_axis[ax]                               # [T,C,H,W]
            rec, bpp, psnr = self.codec(xseg)                       # rec [T,C,H,W], scalar bpp/psnr
            # distribute per frame
            for t, fid in enumerate(self.frameids):
                x_raw = xseg[t:t+1]
                self._codec_cache[ax][fid]       = rec[t:t+1].detach()
                self._codec_cache_bpp[ax][fid]   = (bpp.detach() if torch.is_tensor(bpp)
                                                    else torch.tensor(float(bpp), device=x_raw.device))
                self._codec_cache_psnr[ax][fid]  = (psnr.detach() if torch.is_tensor(psnr)
                                                    else torch.tensor(float(psnr), device=x_raw.device))
                self._codec_cache_rawsnap[ax][fid] = x_raw.detach()

        # ---------- Density sequence ----------
        # Build [T,1,Dy,Dx,Dz]
        with torch.no_grad():
            d_list = [self._gather_density_one_frame(fid).detach() for fid in self.frameids]
            d_seq  = torch.cat(d_list, dim=0)                       # [T,1,Dy,Dx,Dz]
        d_rec, d_bpp, d_psnr = self.codec.forward_density(d_seq)
        for t, fid in enumerate(self.frameids):
            d_raw = d_seq[t:t+1]
            self._dens_cache[fid]         = d_rec[t:t+1].detach()
            self._dens_cache_bpp[fid]     = (d_bpp.detach() if torch.is_tensor(d_bpp)
                                             else torch.tensor(float(d_bpp), device=d_raw.device))
            self._dens_cache_psnr[fid]    = (d_psnr.detach() if torch.is_tensor(d_psnr)
                                             else torch.tensor(float(d_psnr), device=d_raw.device))
            self._dens_cache_rawsnap[fid] = d_raw.detach()

        self._codec_cache_step = int(global_step if global_step is not None else 0)

    def _ste_overrides(self, frameid):
        """Build STE substitutions from per-frame cache (planes + density)."""
        planes  = self._gather_planes_one_frame(frameid)
        density = self._gather_density_one_frame(frameid)

        overrides = {}
        for ax, name in [('xy','k0.xy_plane'), ('xz','k0.xz_plane'), ('yz','k0.yz_plane')]:
            raw = planes[ax]                                   # (1,C,H,W) param
            rec = self._codec_cache[ax][frameid]               # (1,C,H,W) cached recon
            assert rec is not None, "Codec cache is empty—refresh first."
            overrides[name] = rec + (raw - raw.detach())       # STE

        rec_d = self._dens_cache[frameid]
        assert rec_d is not None, "Density cache empty—refresh first."
        overrides['density.grid'] = rec_d + (density - density.detach())
        return overrides

    # -------------------------------------------------------------------------
    # Model instantiation (same as your image class)
    # -------------------------------------------------------------------------
    def _initial_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_kwargs = copy.deepcopy(self.cfg.fine_model_and_render)
        num_voxels = model_kwargs.pop('num_voxels')
        cfg_train = self.cfg.fine_train
        if len(cfg_train.pg_scale):
            num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

        for frameid in self.frameids:
            ckpt = os.path.join(self.cfg.basedir, self.cfg.expname, f'coarse_last_{frameid}.tar')
            if not os.path.isfile(ckpt):
                ckpt = None
            key = str(frameid)
            if self.cfg.data.ndc:
                self.dvgos[key] = DirectMPIGO(
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    num_voxels=num_voxels, mask_cache_path=ckpt, **model_kwargs)
            else:
                self.dvgos[key] = DirectVoxGO(
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    num_voxels=num_voxels, mask_cache_path=ckpt,
                    rgb_model=self.cfg.fine_model_and_render.RGB_model, **model_kwargs)
            self.dvgos[key] = self.dvgos[key].to(device)


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