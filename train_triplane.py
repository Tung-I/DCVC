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
import shutil, json, glob
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
import imageio
from mmengine.config import Config
import wandb

from TeTriRF.lib import dvgo, dmpigo, dvgo_video, utils      # unchanged
from TeTriRF.lib.load_data import load_data
from torch_efficient_distloss import flatten_eff_distloss

"""
Usage:
    python train_triplane.py --config configs/dynerf_flame_steak/video_z14.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --training_mode 1
    python train_triplane.py --config configs/dynerf_sear_steak/image_l.py --frame_ids 0  --training_mode 1
    python train_triplane.py --config TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9  --training_mode 1
    """

WANDB = True

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
    p.add_argument("--i_print", type=int, default=500)
    p.add_argument("--render_only", action='store_true')
    p.add_argument("--no_reload", action='store_true')
    p.add_argument("--no_reload_optimizer", action='store_true')
    # (eval flags left unchanged)
    p.add_argument("--dump_images", action='store_true')
    p.add_argument("--eval_ssim", action='store_true')
    p.add_argument("--eval_lpips_alex", action='store_true')
    p.add_argument("--eval_lpips_vgg", action='store_true')
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

    # in Trainer dataclass fields (or set in __post_init__)
    _best_val_psnr: float = field(default_factory=lambda: float("-inf"), init=False)
    _best_val_step: int   = field(default=-1, init=False)


    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model_and_opt()
        if WANDB:
            wandb.watch(self.model, log="all", log_freq=self.args.i_print)
        self._build_rays()

        # validation cadence (add to cfg.fine_train if you like; default here)
        self.val_every = int(getattr(self.cfg.fine_train, "val_every", 1000))

        # where checkpoints are being written by your model
        self.out_dir = os.path.join(self.cfg.basedir, self.cfg.expname)
        os.makedirs(self.out_dir, exist_ok=True)

    def _build_model_and_opt(self):
        xyz_min = torch.tensor(self.cfg.data.xyz_min)
        xyz_max = torch.tensor(self.cfg.data.xyz_max)
        ids      = torch.unique(self.data['frame_ids']).cpu().tolist()

        self.model = dvgo_video.DirectVoxGO_Video(ids, xyz_min, xyz_max, self.cfg).to(self.device)
        if self.cfg.ckptname:
            ret = self.model.load_checkpoints()
            # self.model.dvgos['0'].reset_occupancy_cache()
            # self.model.set_fixedframe(ret)         # identical to original side-effect
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
        psnr = utils.mse2psnr(loss.detach())
        self._psnr_buffer.append(psnr.item())

        if cfg_t.weight_entropy_last > 0:
            p = render['alphainv_last'].clamp(1e-6,1-1e-6)
            loss += cfg_t.weight_entropy_last * (-(p*torch.log(p)+(1-p)*torch.log(1-p))).mean()

        if cfg_t.weight_distortion > 0:
            loss += cfg_t.weight_distortion * flatten_eff_distloss(
                render['weights'], render['s'], 1/render['n_max'], render['ray_id'])


        loss += self.cfg.fine_train.weight_l1_loss * self.model.compute_k0_l1_loss(fid_batch)
        return loss
    
    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(self):
        cfg_t = self.cfg.fine_train
        os.makedirs(self.out_dir, exist_ok=True)
        for step in trange(1, cfg_t.N_iters + 1):
            if (step + 500) % 1000 == 0:
                self.model.update_occupancy_cache()
            self._progressive_grow(step)
            loss = self._train_step(step)

            # LR decay
            decay = 0.1 ** (1 / (cfg_t.lrate_decay * 1000))
            for g in self.optimizer.param_groups:
                g['lr'] *= decay

            # --- print/log ---
            if step % self.args.i_print == 0 or step == 1:
                dt = time.time() - self._tic
                psnr = float(np.mean(self._psnr_buffer)); self._psnr_buffer.clear()
                tqdm.write(f'[step {step:6d}] loss {loss.item():.4e}  psnr {psnr:5.2f}  '
                        f'elapsed {dt/3600:02.0f}:{dt/60%60:02.0f}:{dt%60:02.0f}')
                if WANDB:
                    wandb.log({
                        "train/psnr": psnr,
                        "train/loss": float(loss.item()),
                        "time/elapsed_s": dt,
                    }, step=step)

            # --- periodic validation on test scene ---
            if (step % self.val_every == 0) or (step == cfg_t.N_iters):
                # current trainer uses single-frame training for DCVCImageCodec
                uniq_fids = torch.unique(self.data['frame_ids']).cpu().numpy().tolist()
                fid = int(uniq_fids[0]) if len(uniq_fids) else 0

                val_psnr = self._eval_test_psnr(fid)
                if val_psnr is not None:
                    if WANDB:
                        wandb.log({"val/psnr": val_psnr}, step=step)
                    tqdm.write(f'[step {step:6d}] VALIDATION  psnr {val_psnr:5.2f} dB')

                    # best-checkpointing
                    if val_psnr > self._best_val_psnr:
                        self._best_val_psnr = val_psnr
                        self._best_val_step = step
                        tqdm.write(f'  ➜ new BEST val psnr {val_psnr:.3f} at step {step}; saving snapshot…')
                        self._save_best_snapshot(step, val_psnr)

            # --- (optional) periodic normal checkpointing, if you still want it ---
            if step % cfg_t.save_every == 0 and step >= cfg_t.save_after:
                self.model.save_checkpoints()

        print('Training finished.')
        if self._best_val_step >= 0:
            print(f'Best validation PSNR {self._best_val_psnr:.3f} dB at step {self._best_val_step}. '
                f'Checkpoint copied under {os.path.join(self.out_dir, "best_psnr")}.')

    # -------------------------------------------------------------------------
    # helper to render a single view and compute PSNR
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _render_one_view_psnr(self, H, W, K, c2w, gt, frame_id, mask=None):
        # rays
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H, W, K, torch.as_tensor(c2w),
            self.cfg.data.ndc, inverse_y=self.cfg.data.inverse_y,
            flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y
        )
        rays_o = rays_o.flatten(0, -2); rays_d = rays_d.flatten(0, -2); viewdirs = viewdirs.flatten(0, -2)

        # chunked render (same chunk size as your eval)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        chunks = [
            {k: v for k, v in self.model(ro, rd, vd, frame_ids=frame_id,
                                         global_step=-1,
                                        near=self.data['near'], far=self.data['far'],
                                        bg=1 if self.cfg.data.white_bkgd else 0,
                                        rand_bkgd=self.cfg.data.rand_bkgd,
                                        stepsize=self.cfg.fine_model_and_render.stepsize,
                                        inverse_y=self.cfg.data.inverse_y,
                                        flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y
                                        ).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(150480, 0),
                                rays_d.split(150480, 0),
                                viewdirs.split(150480, 0))
        ]
        rgb = torch.cat([c['rgb_marched'] for c in chunks]).reshape(H, W, -1).cpu().numpy()


        if mask is not None:
            m = (mask[..., 0] > 0.5)
            mse = np.mean(np.square(rgb[m] - gt[m]))
        else:
            mse = np.mean(np.square(rgb - gt))
        psnr = -10.0 * np.log10(max(mse, 1e-12))
        return float(psnr)
    
    @torch.no_grad()
    def _eval_test_psnr(self, frame_id: int) -> Optional[float]:
        # collect indices of this frame within i_test
        i_test = self.data['i_test']
        frame_ids = self.data['frame_ids']
        mask = (frame_ids == frame_id)[i_test]

        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        t_test = np.array(i_test)[mask]
        if len(t_test) == 0:
            return None  # no test views for this frame

        HW = self.data['HW']; Ks = self.data['Ks']; poses = self.data['poses']
        imgs = self.data['images']
        masks = self.data.get('masks', None)

        # make frame_id become tensor
        frame_id = torch.tensor([frame_id])
        psnrs = []
        for idx in t_test:
            H, W = HW[idx]
            K = Ks[idx]
            c2w = poses[idx]
            gt = imgs[idx].cpu().numpy()
            ms = None if masks is None else masks[idx]
            psnrs.append(self._render_one_view_psnr(H, W, K, c2w, gt, frame_id, mask=ms))
        return float(np.mean(psnrs))

    def _save_best_snapshot(self, step: int, val_psnr: float):
        # first save current checkpoints the usual way
        self.model.save_checkpoints()

        best_dir = os.path.join(self.out_dir, "best_psnr")
        os.makedirs(best_dir, exist_ok=True)

        # copy all tar files that the model just wrote
        for p in glob.glob(os.path.join(self.out_dir, "*.tar")):
            shutil.copy2(p, os.path.join(best_dir, os.path.basename(p)))

        # also keep a note
        meta = {"best_step": step, "best_val_psnr": val_psnr}
        with open(os.path.join(best_dir, "best_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

# ------------------------------------------------------------------------------
# 5. Entry point
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    cfg  = Config.fromfile(args.config)
    cfg.data.frame_ids = args.frame_ids

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
