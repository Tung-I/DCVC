import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call

from torch_scatter import segment_coo
import ipdb
from . import grid
from torch.utils.cpp_extension import load
import copy
from .dvgo import DirectVoxGO
from .dmpigo import DirectMPIGO
from .sh import eval_sh
import time
from torch.serialization import safe_globals
from .dvgo_video import RGB_Net, RGB_SH_Net
from .dcvc_wrapper import DCVCPlaneCodec, DCVCImageCodec, DCVCImageCodecInfer

# Assuming these are already available in your codebase:
# - DirectVoxGO / DirectMPIGO creation via self.initial_models()
# - DCVCImageCodec with .forward(feature_planes) and .forward_density(density_grid)

class DCVC_DVGO_E2E(torch.nn.Module):
    def __init__(self, frameids, xyz_min, xyz_max, cfg=None, device='cuda', infer_mode=False):
        super().__init__()

        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.frameids = frameids
        self.cfg = cfg
        self.dvgos = nn.ModuleDict()
        self.viewbase_pe = cfg.fine_model_and_render.viewbase_pe
        self.fixed_frame = []

        # create per-frame dvgo models and the shared rgbnet, same as before
        self.initial_models()

        self.infer_mode = infer_mode

        # codec (single-frame setting)
        if len(self.frameids) == 1:
            in_channels = int(cfg.fine_model_and_render.rgbnet_dim / 3)
            self.codec = DCVCImageCodec(self.cfg.dcvc, device, infer_mode=infer_mode)
        else:
            raise RuntimeError("DCVCImageCodec only supports single frame input.")

    # ---------------------- small helpers ----------------------

    def _gather_planes_one_frame(self, frameid):
        k0 = self.dvgos[str(frameid)].k0
        return {
            'xy': getattr(k0, 'xy_plane'),  # (1,C,H,W) nn.Parameter
            'xz': getattr(k0, 'xz_plane'),
            'yz': getattr(k0, 'yz_plane'),
        }

    def _gather_density_one_frame(self, frameid):
        dvgo = self.dvgos[str(frameid)]
        return getattr(dvgo.density, 'grid')  # (1,1,Dy,Dx,Dz) nn.Parameter

    def _encode_decode_planes_and_density(self, frameid):
        """
        Run codec on current planes + density (no detach, no cache, no STE).
        Returns:
          recon_by_axis: dict[str->Tensor]  (1,C,H,W)
          bpp_by_axis:   dict[str->Tensor or float]
          psnr_by_axis:  dict[str->Tensor or float]
          d_recon:       Tensor (1,1,Dy,Dx,Dz)
          d_bpp:         Tensor or float
          d_psnr:        Tensor or float
        """
        planes = self._gather_planes_one_frame(frameid)
        recon_by_axis, bpp_by_axis, psnr_by_axis = {}, {}, {}

        for ax in ('xy', 'xz', 'yz'):
            x = planes[ax]                               # (1,C,H,W), Parameter
            # codec forward must remain differentiable here
            recon, bpp, plane_psnr = self.codec(x)
            recon_by_axis[ax] = recon
            bpp_by_axis[ax]   = bpp
            psnr_by_axis[ax]  = plane_psnr

        density = self._gather_density_one_frame(frameid)   # (1,1,Dy,Dx,Dz), Parameter
        d_recon, d_bpp, d_psnr = self.codec.forward_density(density)

        return recon_by_axis, bpp_by_axis, psnr_by_axis, d_recon, d_bpp, d_psnr

    # ---------------------- main forward ----------------------

    def forward(self, rays_o, rays_d, viewdirs, frame_ids, global_step=None, mode='feat', **render_kwargs):
        # 1) single-frame per batch invariant
        frame_ids_unique = torch.unique(frame_ids, sorted=True).cpu().int().numpy().tolist()
        assert len(frame_ids_unique) == 1, "Expect a single frame per batch"
        frameid = frame_ids_unique[0]
        dvgo = self.dvgos[str(frameid)]

        # 2) run codec on current planes + density (no caching, no STE)
        (recon_by_axis, bpp_by_axis, psnr_by_axis,
         d_recon, d_bpp, d_psnr) = self._encode_decode_planes_and_density(frameid)

        # 3) build functional overrides with the reconstructed tensors
        #    (cast to the registered param dtype/device when necessary)
        param_map  = dict(dvgo.named_parameters())
        buffer_map = dict(dvgo.named_buffers())

        overrides = {}

        # planes
        for ax, name in [('xy','k0.xy_plane'), ('xz','k0.xz_plane'), ('yz','k0.yz_plane')]:
            like = param_map.get(name, None)
            rec  = recon_by_axis[ax]
            if like is not None:
                rec = rec.to(device=like.device, dtype=like.dtype)
            overrides[name] = rec

        # density
        dens_name = 'density.grid'
        like_dens = param_map.get(dens_name, None)
        if like_dens is not None:
            d_recon = d_recon.to(device=like_dens.device, dtype=like_dens.dtype)
        overrides[dens_name] = d_recon

        # split overrides into param/buffer dicts for robust merge
        param_overrides  = {k: v for k, v in overrides.items() if k in param_map}
        buffer_overrides = {k: v for k, v in overrides.items() if k in buffer_map}

        merged_params  = {**param_map,  **param_overrides}
        merged_buffers = {**buffer_map, **buffer_overrides}

        # 4) render using reconstructed weights via functional_call
        ret_frame = functional_call(
            dvgo,
            {**merged_params, **merged_buffers},
            (rays_o, rays_d, viewdirs),
            {'shared_rgbnet': self.rgbnet, 'global_step': global_step, 'mode': mode, **render_kwargs}
        )

        # 5) bitrate accounting: planes + density, averaged over 4 components
        bpps_planes = []
        for ax in ('xy','xz','yz'):
            b = bpp_by_axis[ax]
            bpps_planes.append(b if torch.is_tensor(b) else torch.tensor(float(b), device=d_recon.device))
        d_bpp_t = d_bpp if torch.is_tensor(d_bpp) else torch.tensor(float(d_bpp), device=d_recon.device)

        avg_bpp = (bpps_planes[0] + bpps_planes[1] + bpps_planes[2] + d_bpp_t) / 4.0

        # 6) diagnostics (PSNR by axis + density)
        out_psnr = {ax: (psnr_by_axis[ax] if torch.is_tensor(psnr_by_axis[ax])
                         else torch.tensor(float(psnr_by_axis[ax]), device=d_recon.device))
                    for ax in ('xy','xz','yz')}
        out_psnr['density'] = d_psnr if torch.is_tensor(d_psnr) \
                              else torch.tensor(float(d_psnr), device=d_recon.device)

        return ret_frame, avg_bpp, out_psnr
    

    def initial_models(self):
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

    def save_infer_checkpoints(self, qp):
        cfg = self.cfg
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid = str(frameid)
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'compressed_{qp}_fine_last_{frameid}.tar')
            ckpt = {
                'model_state_dict': self.dvgos[frameid].state_dict(),
                'model_kwargs': self.dvgos[frameid].get_kwargs(),
            }
            torch.save(ckpt, ckpt_path)
            print(f"Frame {frameid}'s checkpoint saved to {ckpt_path}")

    def set_fixedframe(self, ids):
        """Set the fixed frame ids for the model.
        """
        self.fixed_frame = ids
        
        if len(ids)>0:
            frameid = -1
            for frameid in self.frameids:
                if frameid not in self.fixed_frame:
                    break
            assert frameid!=-1
            source_id = ids[0]
            xy_plane, xz_plane, yz_plane = self.dvgos[str(source_id)].k0.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)
            
            density_grid  = self.dvgos[str(source_id)].density.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)
            for frameid in self.frameids:
                if frameid in self.fixed_frame:
                    continue
                device = self.dvgos[str(frameid)].k0.xy_plane.device
                if self.cfg.fine_train.initialize_feature:
                    self.dvgos[str(frameid)].k0.xy_plane = nn.Parameter(xy_plane.clone()).to(device)
                    self.dvgos[str(frameid)].k0.xz_plane = nn.Parameter(xz_plane.clone()).to(device)
                    self.dvgos[str(frameid)].k0.yz_plane = nn.Parameter(yz_plane.clone()).to(device)
                if self.cfg.fine_train.initialize_density:
                    self.dvgos[str(frameid)].density.grid = nn.Parameter((density_grid.clone()*1.0 + 0*torch.randn_like(density_grid))).to(device) 

                print(f'Initialize  frame:{frameid}')




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
