# run_multiframe_refactor.py
# ──────────────────────────────────────────────────────────────────────────────
"""End-to-end training script for TeTriRF (multi-frame).  Exactly reproduces the
behaviour of the original `run_multiframe.py` but is easier to extend (e.g. to
DCVC-integrated tri-planes, multi-GPU, AMP, etc.)."""

# ------------------------------------------------------------------------------
# 1. Imports & utils
# ------------------------------------------------------------------------------
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

from TeTriRF.lib import dvgo, dmpigo, dvgo_video, dcvc_dvgo_video, utils      # unchanged
from TeTriRF.lib.load_data import load_data
from torch_efficient_distloss import flatten_eff_distloss


"""
Usage:
    python train_dcvc_triplane.py --config TeTriRF/configs/N3D/flame_steak_dcvc.py --frame_ids 0 1 2 3 4 5 6 7 8 9 --training_mode 1
"""

# ------------------------------------------------------------------------------
# 2. Argument & config handling
# ------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--frame_ids', nargs='+', type=int, help='List of frame IDs')
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--training_mode", type=int, default=0)
    # misc I/O
    p.add_argument("--i_print", type=int, default=100)
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


# ------------------------------------------------------------------------------
# 3. Data helpers
# ------------------------------------------------------------------------------
def load_dataset(cfg: Config) -> Dict:
    """Wrapper around TeTriRF.lib.load_data with post-processing identical to the
    original script (trim unused keys, move to CPU if irregular)."""
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


# ------------------------------------------------------------------------------
# 4. Core trainer
# ------------------------------------------------------------------------------
@dataclass
class Trainer:
    """Stateful wrapper around the whole TeTriRF fine-stage training loop."""
    cfg: Config
    args: argparse.Namespace
    data: Dict
    device: torch.device = field(init=False)
    model: dvgo_video.DirectVoxGO_Video = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    ray_loader: DataLoader = field(init=False)
    batch_sampler: Callable = field(init=False)

    # bookkeeping
    _psnr_buffer: List[float] = field(default_factory=list, init=False)
    _tic: float = field(default_factory=time.time, init=False)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model_and_opt()
        
        wandb.watch(self.model, log="all", log_freq=self.args.i_print)

        self._build_rays()

    def _build_model_and_opt(self):
        xyz_min = torch.tensor(self.cfg.data.xyz_min)
        xyz_max = torch.tensor(self.cfg.data.xyz_max)
        ids      = torch.unique(self.data['frame_ids']).cpu().tolist()

        # self.model = dvgo_video.DirectVoxGO_Video(ids, xyz_min, xyz_max, self.cfg).to(self.device)
        self.model = dcvc_dvgo_video.DCVC_DVGO_Video(ids, xyz_min, xyz_max, self.cfg, dcvc_qp = self.cfg.fine_train.qp).to(self.device)
        ret = self.model.load_checkpoints()
        # if args.resume:
        #     ret = self.model.load_checkpoints()
        #     self.model.set_fixedframe(ret)         # identical to original side-effect
        if (not self.cfg.fine_model_and_render.dynamic_rgbnet
            and self.args.training_mode > 0):
            self.cfg.fine_train.lrate_rgbnet = 0
        self.optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(
            self.model, self.cfg.fine_train, global_step=0)

    # -------------------------------------------------------------------------
    # Dataset → DataLoader
    # -------------------------------------------------------------------------
    def _build_rays(self):
        """Collect rays exactly the same way the original script does ― keep the
        six Python lists in memory and return an index-generator.  We still create
        a dummy DataLoader so later refactor (multi-GPU, etc.) is easy, but we no
        longer rely on a non-existent `get_by_indices()` method."""
        (self.rgb_l, self.ro_l, self.rd_l,
        self.vd_l, self.sz_l, self.fid_l,
        sampler_fn) = self._gather_training_rays()

        self.batch_sampler = sampler_fn            # -> (camera_id , sel_idx)
        # keep a DL in case you want to push the loop into torch.compile later
        from torch.utils.data import DataLoader
        class _Dummy(torch.utils.data.Dataset):
            def __len__(self): return 1
            def __getitem__(self, i): return 0
        self.ray_loader = DataLoader(_Dummy(), batch_size=1)

    def _gather_training_rays(self, tmasks=None):
        """Bit-for-bit port of the inline helper in run_multiframe.py."""
        cfg, data, model = self.cfg, self.data, self.model
        HW, Ks, poses, frame_ids = data['HW'], data['Ks'], data['poses'], data['frame_ids']
        uniq_ids = torch.unique(frame_ids, sorted=True).cpu().tolist()
        device   = self.device                                 # == cuda

        rgb_l, ro_l, rd_l, vd_l, imsz_l, fid_l = [], [], [], [], [], []
        for fid in uniq_ids:
            if fid in model.fixed_frame:
                continue
            mask     = (frame_ids == fid)[data['i_train']]
            t_train  = np.array(data['i_train'])[mask]
            rgb_ori  = data['images'][t_train].to('cpu' if cfg.data.load2gpu_on_the_fly
                                                else device)

            pmasks = None
            if tmasks is not None:
                pmasks = torch.from_numpy(tmasks[t_train]).to(
                            'cpu' if cfg.data.load2gpu_on_the_fly else device)

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
            self.optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(
                self.model, cfg_t, global_step=0)
            for fid in self.model.dvgos.keys():
                if int(fid) in self.model.fixed_frame: continue
                self.model.dvgos[fid].act_shift -= cfg_t.decay_after_scale / (2 if step in cfg_t.pg_scale2 else 1)
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------
    def _train_step(self, step: int):
        cam, sel_np = self.batch_sampler()
        sel = torch.from_numpy(sel_np)

        # identical slicing to original script
        target  = self.rgb_l[cam][sel]
        ro      = self.ro_l[cam][sel]
        rd      = self.rd_l[cam][sel]
        vd      = self.vd_l[cam][sel]
        fid_b   = self.fid_l[cam][sel]

        if self.cfg.data.load2gpu_on_the_fly:
            to_dev = lambda x: x.to(self.device)
            target, ro, rd, vd = map(to_dev, (target, ro, rd, vd))
        
        if step > 1:
            self.model.run_codec_once()

        render = self.model(ro, rd, vd, frame_ids=fid_b, global_step=step,
                            mode='feat',
                            near=self.data['near'], far=self.data['far'],
                            bg=1 if self.cfg.data.white_bkgd else 0,
                            rand_bkgd=self.cfg.data.rand_bkgd,
                            stepsize=self.cfg.fine_model_and_render.stepsize,
                            inverse_y=self.cfg.data.inverse_y,
                            flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y)

        loss = self._compute_loss(render, target, step, fid_b)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss

    def _compute_loss(self, render, target, step, fid_batch):
        cfg_t = self.cfg.fine_train
        loss = cfg_t.weight_main * F.mse_loss(render['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach()); self._psnr_buffer.append(psnr.item())

        if cfg_t.weight_entropy_last > 0:
            p = render['alphainv_last'].clamp(1e-6,1-1e-6)
            loss += cfg_t.weight_entropy_last * (-(p*torch.log(p)+(1-p)*torch.log(1-p))).mean()

        if cfg_t.weight_distortion > 0:
            loss += cfg_t.weight_distortion * flatten_eff_distloss(
                render['weights'], render['s'], 1/render['n_max'], render['ray_id'])


        loss += self.cfg.fine_train.weight_l1_loss * self.model.compute_k0_l1_loss(fid_batch)
       
        # loss += cfg_t.lambda_bpp * self.model.last_bpp
        return loss

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
            loss = self._train_step(step)

            # LR decay
            decay = 0.1 ** (1 / (cfg_t.lrate_decay*1000))
            for g in self.optimizer.param_groups: g['lr'] *= decay

            if step % self.args.i_print == 0 or step == 1:
                dt   = time.time() - self._tic
                psnr = np.mean(self._psnr_buffer); self._psnr_buffer.clear()
                tqdm.write(f'[step {step:6d}] loss {loss.item():.4e}  psnr {psnr:5.2f}  '
                           f'elapsed {dt/3600:02.0f}:{dt/60%60:02.0f}:{dt%60:02.0f}')

                wandb.log({
                    "train/psnr": float(psnr),
                    "train/loss": float(loss.item()),
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
    wandb.init(
      project="image_dcvc_flame_steak",
      name=cfg.wandbname,
      config=vars(args)
    )

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    seed_everything(args.seed)

    data = load_dataset(cfg)
    if not args.render_only:
        Trainer(cfg, args, data).train()
