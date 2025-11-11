from __future__ import annotations
import os, time, copy, random, argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
import imageio
from mmengine.config import Config
import wandb
import math

from src.utils.common import setup_unique_torch_extensions_dir
from src.models.ste_dvgo_video import STE_DVGO_Video
# setup_unique_torch_extensions_dir()

from TeTriRF.lib.utils import debug_print_param_status
from TeTriRF.lib import dvgo, dvgo_video, dcvc_dvgo_video, utils      # unchanged
from TeTriRF.lib.load_data import load_data
from torch_efficient_distloss import flatten_eff_distloss

WANDB=True

"""
Usage:
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/dcvc_qp48.py --frame_ids 0 1 2 3 4 5 6 7 8 9
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/av1_qp20.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/av1_qp44_mosaic_affine_tv.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/hevc_qp36_flatten_affine.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/av1_qp52_k16.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/av1_qp26_tv.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/dcvc_qp24_flat4_absmax_tv.py --frame_ids 0 1 2 3 4 5 6 7 8 9 
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/av1_qp62_gop10_tv.py --frame_ids 0 1 2 3 4 5 6 7 8 9 
    python train_codec_nerf_video.py --config configs/nhr_sport1/av1_qp44.py --frame_ids 0 1 2 3 4 5 6 7 8 9 
    python train_codec_nerf_video.py --config configs/dynerf_flame_steak/hevc_qp44_gop10_tv.py --frame_ids 0 1 2 3 4 5 6 7 8 9 
    python train_codec_nerf_video.py --config configs/nhr_sport1/dcvc_qp24.py --frame_ids 0 1 2 3 4 5 6 7 8 9 
    python train_codec_nerf_video.py --config configs/dynerf_sear_steak/av1_qp50.py --frame_ids 0 1 2 3 4 5 6 7 8 9
    """

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--frame_ids', nargs='+', type=int, help='List of frame IDs')
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--training_mode", type=int, default=1)
    # misc I/O
    p.add_argument("--i_print", type=int, default=250)
    p.add_argument("--i_log", type=int, default=1000)
    p.add_argument("--render_only", action='store_true')
    p.add_argument("--no_reload", action='store_true')
    p.add_argument("--no_reload_optimizer", action='store_true')
    # (eval flags left unchanged)
    p.add_argument("--dump_images", action='store_true')
    p.add_argument("--eval_ssim", action='store_true')
    p.add_argument("--eval_lpips_alex", action='store_true')
    p.add_argument("--eval_lpips_vgg", action='store_true')
    p.add_argument("--resume", action='store_true')
    return p

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def load_dataset(cfg: Config) -> Dict:
    data = load_data(cfg.data)
    keep = {'hwf','HW','Ks','near','far','near_clip',
            'i_train','i_val','i_test','irregular_shape',
            'poses','render_poses','images','frame_ids','masks'}
    for k in list(data.keys()):
        if k not in keep:
            data.pop(k)

    if data['irregular_shape']:
        data['images'] = [torch.FloatTensor(im, device='cpu') for im in data['images']]
    else:
        data['images'] = torch.FloatTensor(data['images'], device='cpu')
    data['poses']  = torch.Tensor(data['poses'])        # stays CUDA by default
    return data

@dataclass
class Trainer:
    cfg: Config
    args: argparse.Namespace
    data: Dict
    device: torch.device = field(init=False)
    model: dvgo_video.DirectVoxGO_Video = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    ray_loader: DataLoader = field(init=False)
    batch_sampler: Callable = field(init=False)
    _psnr_buffer: List[float] = field(default_factory=list, init=False)
    _tic: float = field(default_factory=time.time, init=False)

    def __post_init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model_and_opt()

        # debug_print_param_status(self.model, self.optimizer)
        # raise Exception
    
        self._build_rays()
        self.lambda_bpp = self.qp_to_lambda()
        
        """
        # for _qp in [0, 12, 24, 48]:
        #     print(f"lambda_bpp[{_qp}] = {self.qp_to_lambda(_qp)}")
        # raise Exception('')
        lambda_bpp[0] = 1.0
        lambda_bpp[12] = 3.544807152675064
        lambda_bpp[24] = 12.565657749656294
        lambda_bpp[48] = 157.8957546814973
        """

        if WANDB:
            wandb.watch(self.model, log="all", log_freq=self.args.i_log)

    def _build_model_and_opt(self):
        xyz_min = torch.tensor(self.cfg.data.xyz_min)
        xyz_max = torch.tensor(self.cfg.data.xyz_max)
        ids      = torch.unique(self.data['frame_ids']).cpu().tolist()
       
        if self.cfg.codec.train_mode == 'ste':
            self.model = STE_DVGO_Video(ids, xyz_min, xyz_max, self.cfg).to(self.device)
            codecs_params = None
            sandwich_params = None
        else:
            raise NotImplementedError(f"train_mode {self.cfg.codec.train_mode} not implemented")
        
        if self.cfg.ckptname:
            _ = self.model.load_checkpoints()
            
        self.optimizer = utils.create_optimizer_or_freeze_model_dcvc_triplane(
                self.model, self.cfg.fine_train, global_step=0, codec_params=codecs_params, sandwich_params=sandwich_params
        )

    # -------------------------------------------------------------------------
    # Dataset → DataLoader
    # -------------------------------------------------------------------------
    def _build_rays(self):
        (self.rgb_l, self.ro_l, self.rd_l,
        self.vd_l, self.sz_l, self.fid_l,
        sampler_fn) = self._gather_training_rays()

        self.batch_sampler = sampler_fn            # -> (camera_id , sel_idx)
        from torch.utils.data import DataLoader
        class _Dummy(torch.utils.data.Dataset):
            def __len__(self): return 1
            def __getitem__(self, i): return 0
        self.ray_loader = DataLoader(_Dummy(), batch_size=1)

    def _gather_training_rays(self, tmasks=None):
        cfg, data, model = self.cfg, self.data, self.model
        HW, Ks, poses, frame_ids = data['HW'], data['Ks'], data['poses'], data['frame_ids']
        uniq_ids = torch.unique(frame_ids, sorted=True).cpu().tolist()
        device   = self.device                                 # == cuda

        rgb_l, ro_l, rd_l, vd_l, imsz_l, fid_l = [], [], [], [], [], []
        for fid in uniq_ids:
            if fid in model.fixed_frame:
                continue

            # pick the training views that belong to this frame id
            mask     = (frame_ids == fid)[data['i_train']]
            t_train  = np.array(data['i_train'])[mask]
            rgb_ori  = data['images'][t_train].to('cpu' if cfg.data.load2gpu_on_the_fly
                                                else device)
            pmasks = None
            if tmasks is not None:
                pmasks = torch.from_numpy(tmasks[t_train]).to(
                            'cpu' if cfg.data.load2gpu_on_the_fly else device)

            # Each entry in the returned lists corresponds to a single view 
            # (have the same fid)
            rgb, ro, rd, vd, imsz, fids = dvgo.get_training_rays_multi_frame(
                rgb_tr_ori=rgb_ori,
                train_poses=poses[t_train],
                HW=HW[t_train], Ks=Ks[t_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                frame_ids=frame_ids[t_train],
                model=model.dvgos[str(fid)],
                masks=pmasks, render_kwargs={},             # same as original
                flatten=(cfg.fine_train.ray_sampler == 'flatten')
            )
            
            rgb_l+=rgb; ro_l+=ro; rd_l+=rd; vd_l+=vd; imsz_l+=imsz; fid_l+=fids

        sampler = dvgo.batch_indices_generator_MF(rgb_l, cfg.fine_train.N_rand)
        return rgb_l, ro_l, rd_l, vd_l, imsz_l, fid_l, lambda: next(sampler)

    # -------------------------------------------------------------------------
    # Training utilities
    # -------------------------------------------------------------------------
    def _progressive_grow(self, step: int):
        cfg_t = self.cfg.fine_train
        if step in cfg_t.pg_scale + cfg_t.pg_scale2:
            n_left = (len(cfg_t.pg_scale + cfg_t.pg_scale2)
                      - (cfg_t.pg_scale + cfg_t.pg_scale2).index(step) - 1)
            vox = int(self.cfg.fine_model_and_render.num_voxels / (2 ** n_left))
            self.model.scale_volume_grid(vox)
            # self.optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(
            #     self.model, cfg_t, global_step=0)
            self.optimizer = utils.create_optimizer_or_freeze_model_dcvc_triplane(
                self.model, self.cfg.fine_train, global_step=step
            )
            print(f'Progressive growing to {vox} voxels at step {step}.')
            for fid in self.model.dvgos.keys():
                if int(fid) in self.model.fixed_frame: continue
                self.model.dvgos[fid].act_shift -= cfg_t.decay_after_scale / (2 if step in cfg_t.pg_scale2 else 1)
            torch.cuda.empty_cache()

    def qp_to_lambda(self):
        # assert that only one of dcvc_qp and jpeg.quality is set
        # dcvc_qp = self.cfg.codec.dcvc_qp
        # quality = self.cfg.codec.quality
        # lambda_min = self.cfg.fine_train.lambda_min
        # lambda_max = self.cfg.fine_train.lambda_max


        # if dcvc_qp is not None:
        #     lambda_val = math.log(lambda_min) + dcvc_qp / (64 - 1) * (
        #             math.log(lambda_max) - math.log(lambda_min))
        #     lambda_val = math.pow(math.e, lambda_val)
        # elif quality is not None:
        #     qp = quality
        #     lambda_val = math.log(lambda_min) + qp / (100 - 1) * (
        #             math.log(lambda_max) - math.log(lambda_min))
        #     lambda_val = math.pow(math.e, lambda_val)
        # else:
        #     lambda_val = 1.0

        # return lambda_val
        return 0.0

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------
    def _train_step(self, step: int):
        # every batch currently contains rays from exactly one camera view of one frame id.
        cam, sel_np = self.batch_sampler()
        sel = torch.from_numpy(sel_np)
        target  = self.rgb_l[cam][sel]
        ro      = self.ro_l[cam][sel]
        rd      = self.rd_l[cam][sel]
        vd      = self.vd_l[cam][sel]
        fid_b   = self.fid_l[cam][sel]

        if self.cfg.data.load2gpu_on_the_fly:
            to_dev = lambda x: x.to(self.device)
            target, ro, rd, vd = map(to_dev, (target, ro, rd, vd))
        
        # Encode and decode Tri-Planes, compute rendering loss
        render, avg_bpp, psnr_by_axis = self.model(ro, rd, vd, frame_ids=fid_b, global_step=step,
                            mode='feat',
                            near=self.data['near'], far=self.data['far'],
                            bg=1 if self.cfg.data.white_bkgd else 0,
                            rand_bkgd=self.cfg.data.rand_bkgd,
                            stepsize=self.cfg.fine_model_and_render.stepsize,
                            inverse_y=self.cfg.data.inverse_y,
                            flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y)

        rec_loss, bpp_loss = self._compute_loss(render, target, step, fid_b, avg_bpp)
        self.optimizer.zero_grad(set_to_none=True)
        loss = rec_loss + bpp_loss
        loss.backward()

        # Total-variation regularization on voxel grids
        cfg_train = self.cfg.fine_train
        if step<cfg_train.tv_before and step>cfg_train.tv_after and step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                self.model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(ro), step<cfg_train.tv_dense_before, fid_b)
            if cfg_train.weight_tv_k0>0:
                self.model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(ro), step<cfg_train.tv_dense_before, fid_b)

        self.optimizer.step()
        return loss, psnr_by_axis, avg_bpp, rec_loss, bpp_loss

    def _compute_loss(self, render, target, step, fid_batch, avg_bpp):
        cfg_t = self.cfg.fine_train
        rec_loss = cfg_t.weight_main * F.mse_loss(render['rgb_marched'], target)
        psnr = utils.mse2psnr(rec_loss.detach()); self._psnr_buffer.append(psnr.item())

        if cfg_t.weight_entropy_last > 0:
            p = render['alphainv_last'].clamp(1e-6,1-1e-6)
            rec_loss += cfg_t.weight_entropy_last * (-(p*torch.log(p)+(1-p)*torch.log(1-p))).mean()

        if cfg_t.weight_distortion > 0:
            rec_loss += cfg_t.weight_distortion * flatten_eff_distloss(
                render['weights'], render['s'], 1/render['n_max'], render['ray_id'])

        # loss += self.cfg.fine_train.weight_l1_loss * self.model.compute_k0_l1_loss(fid_batch)


        bpp_loss = self.lambda_bpp * avg_bpp

        return rec_loss, bpp_loss

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(self):
        cfg_t = self.cfg.fine_train
        os.makedirs(os.path.join(self.cfg.basedir, self.cfg.expname), exist_ok=True)
        for step in trange(1, cfg_t.N_iters+1):
            if (step+500) % 1000 == 0:
                self.model.update_occupancy_cache()
            self._progressive_grow(step)
            loss, plane_psnr_dict, avg_bpp, rec_loss, bpp_loss  = self._train_step(step)

            # LR decay
            decay = 0.1 ** (1 / (cfg_t.lrate_decay*1000)) # cfg_t.lrate_decay = 20
            for g in self.optimizer.param_groups: g['lr'] *= decay

            if step % self.args.i_print == 0 or step == 1:
                dt   = time.time() - self._tic
                psnr = np.mean(self._psnr_buffer); self._psnr_buffer.clear()
                tqdm.write(f'[step {step:6d}] loss {loss.item():.4e}  psnr {psnr:5.2f} '
                           f'elapsed {dt/3600:02.0f}:{dt/60%60:02.0f}:{dt%60:02.0f}')
                
                # raise Exception("Stop here")

            if step % self.args.i_log == 0 and WANDB:
                wandb.log({
                    "train/psnr": float(psnr),
                    "train/loss": float(loss.item()),
                    "train/feature_plane_psnr": float(plane_psnr_dict['xy']),
                    "train/density_plane_psnr": float(plane_psnr_dict['density']),
                    # "train/yz_plane_psnr": float(plane_psnr_dict['yz']),
                    "train/rec_loss": float(rec_loss.item()),
                    "train/bpp_loss": float(bpp_loss.item()),
                    "train/avg_bpp": float(avg_bpp),
                    "time/elapsed_s": dt,
                }, step=step)

            if step % cfg_t.save_every == 0 and step >= cfg_t.save_after:
                self.model.save_checkpoints()

        print('Training finished.')

# ------------------------------------------------------------------------------
# 5. Entry point
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    cfg  = Config.fromfile(args.config)
    cfg.data.frame_ids = args.frame_ids

    # initialize wandb
    if WANDB:
        wandb.init(
        project=cfg.wandbprojectname,
        name=cfg.expname,
        config=vars(args)
        )

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    seed_everything(args.seed)

    data = load_dataset(cfg)
    if not args.render_only:
        Trainer(cfg, args, data).train()