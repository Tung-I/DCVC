import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from .plane_codec_dcvc import DCVCPlaneCodec

class DCVC_DVGO_Video(torch.nn.Module):
    def __init__(self, frameids, xyz_min, xyz_max, cfg=None, device='cuda'):
        super(DCVC_DVGO_Video, self).__init__()

        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.frameids = frameids
        self.cfg = cfg
        self.dvgos = nn.ModuleDict()
        self.viewbase_pe = cfg.fine_model_and_render.viewbase_pe
        self.fixed_frame = []

        self.initial_models()

        self.last_bpp   = torch.tensor(0., device=device)
        self.planes_are_dirty = True   # set True whenever you *update* planes

    @torch.enable_grad()            # stay in graph
    def run_codec_once(self):
        """Encode-decode xy/xz/yz planes; store distorted planes & bpp."""
        planes_by_axis = {'xy': [], 'xz': [], 'yz': []}
        for fid in self.frameids:
            k0 = self.dvgos[str(fid)].k0
            for ax in planes_by_axis:
                planes_by_axis[ax].append(getattr(k0, f'{ax}_plane')[0])

        total_bpp = 0.
        for ax, seq_list in planes_by_axis.items():
            seq = torch.stack(seq_list, dim=0)         # [T,C,H,W]
            recon, bpp = self.codec(seq)
            total_bpp += bpp
            # write back
            idx = 0
            for fid in self.frameids:
                k0 = self.dvgos[str(fid)].k0
                getattr(k0, f'{ax}_plane').data = recon[idx]; idx += 1

        self.last_bpp = total_bpp / 3        # average over axes
        self.planes_are_dirty = False        # we’re now in sync


    def forward(self, rays_o, rays_d, viewdirs, frame_ids, global_step=None, mode='feat', **render_kwargs):
        
        frame_ids_unique = torch.unique(frame_ids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        frameid = frame_ids_unique[0]
        ret_frame = self.dvgos[str(frameid)](rays_o, rays_d, viewdirs, shared_rgbnet= self.rgbnet, global_step=global_step, mode=mode, **render_kwargs)

        return ret_frame

    def get_kwargs(self):
        return {
            'frameids': self.frameids,
            'xyz_min': self.xyz_min,
            'xyz_max': self.xyz_max,
            'viewbase_pe': self.viewbase_pe,
        }

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


    def load_checkpoints(self):

        cfg = self.cfg
        ret = []

        for frameid in self.frameids:
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
            # break

        beg = self.frameids[0]
        eend = self.frameids[-1]
        rgbnet_file = os.path.join(cfg.basedir, cfg.ckptname, f'rgbnet_{beg}_{eend}.tar')
        
        checkpoint =torch.load(rgbnet_file)
        self.rgbnet.load_state_dict(checkpoint['model_state_dict']) 
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

    

       

        