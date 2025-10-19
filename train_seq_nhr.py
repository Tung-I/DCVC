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

from TeTriRF.lib import dvgo, dmpigo, dvgo_video, utils      # unchanged
from TeTriRF.lib.load_data import load_data
from torch_efficient_distloss import flatten_eff_distloss

from src.data_loader.sampler import MultiBucketCycleSampler

"""
Usage:
    python train_seq_nhr.py --config configs/NHR/sport1.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --training_mode 1 

"""

WANDB = True

# ------------------------------------------------------------------------------
# 2. Argument & config handling
# ------------------------------------------------------------------------------
def build_arg_parser():
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
        if WANDB:
            wandb.watch(self.model, log="all", log_freq=self.args.i_print)
        self._build_rays()

    def _build_model_and_opt(self):
        xyz_min = torch.tensor(self.cfg.data.xyz_min)
        xyz_max = torch.tensor(self.cfg.data.xyz_max)
        ids      = torch.unique(self.data['frame_ids']).cpu().tolist()

        self.model = dvgo_video.DirectVoxGO_Video(ids, xyz_min, xyz_max, self.cfg).to(self.device)
        if self.cfg.ckptname:
            ret = self.model.load_checkpoints()
        if (not self.cfg.fine_model_and_render.dynamic_rgbnet
            and self.args.training_mode > 0):
            self.cfg.fine_train.lrate_rgbnet = 0
        self.optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(
            self.model, self.cfg.fine_train, global_step=0)

    # -------------------------------------------------------------------------
    # Dataset → DataLoader
    # -------------------------------------------------------------------------
    def _build_rays(self):
        (self.rgb_l, self.ro_l, self.rd_l,
        self.vd_l, self.sz_l, self.fid_l,
        sampler_fn) = self._gather_training_rays()

        self.batch_sampler = sampler_fn
        from torch.utils.data import DataLoader
        class _Dummy(torch.utils.data.Dataset):
            def __len__(self): return 1
            def __getitem__(self, i): return 0
        self.ray_loader = DataLoader(_Dummy(), batch_size=1)

    def _gather_training_rays(self, tmasks=None):
        cfg, data, model = self.cfg, self.data, self.model
        HW, Ks, poses, frame_ids_all = data['HW'], data['Ks'], data['poses'], data['frame_ids']
        uniq_ids = torch.unique(frame_ids_all, sorted=True).cpu().tolist()
        device   = self.device

        # default stepsize if missing
        step_size = float(cfg.fine_model_and_render.get('stepsize', 1.0))
        render_kwargs = {
            'near': float(data['near']),
            'far':  float(data['far']),
            'stepsize': step_size,
        }

        rgb_l, ro_l, rd_l, vd_l, imsz_l, fid_l = [], [], [], [], [], []

        for fid in uniq_ids:
            if fid in model.fixed_frame:
                continue

            i_train_t = torch.as_tensor(self.data['i_train'], dtype=torch.long, device=frame_ids_all.device)
            sel       = (frame_ids_all[i_train_t] == fid)                      # torch.bool, length == len(i_train)
            t_train   = i_train_t[sel].cpu().numpy()                           # numpy indices for downstream code
            if t_train.size == 0:
                continue

            rgb_ori  = data['images'][t_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

            pmasks = None
            if tmasks is not None:
                pmasks = torch.from_numpy(tmasks[t_train]).to('cpu' if cfg.data.load2gpu_on_the_fly else device)

            # build per-view buckets for this fid
            rgb, ro, rd, vd, imsz, fids = dvgo.get_training_rays_multi_frame(
                rgb_tr_ori=rgb_ori,
                train_poses=poses[t_train],
                HW=HW[t_train], Ks=Ks[t_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                frame_ids=frame_ids_all[t_train],
                model=model.dvgos[str(fid)],
                masks=pmasks, render_kwargs=render_kwargs,
                flatten=(cfg.fine_train.ray_sampler == 'flatten')
            )
            rgb_l+=rgb; ro_l+=ro; rd_l+=rd; vd_l+=vd; imsz_l+=imsz; fid_l+=fids

        # len(rgb_l) = n_buckets (one bucket = one (frame, view) ray set)
        bucket_lengths = [len(x) for x in rgb_l]
        BS = int(cfg.fine_train.N_rand)

        # Original sampler (may mix frame_ids across buckets in a batch)
        base_sampler = MultiBucketCycleSampler(bucket_lengths, BS,
                                            shuffle_across_buckets=False,
                                            shuffle_within_bucket=True)

        # --- Wrap the sampler to remove mixed-fid outliers and pad to BS ---
        # Build a bucket -> fid map once.
        bucket2fid = []
        for f_b in fid_l:
            # each bucket's fid list is constant; take the first element
            if isinstance(f_b, torch.Tensor):
                bucket2fid.append(int(f_b[0].item()))
            else:
                # rare path if not tensor
                bucket2fid.append(int(np.asarray(f_b)[0]))

        import numpy as np
        from collections import defaultdict

        rng = np.random.RandomState(getattr(cfg, "seed", 0))

        def frame_safe_sampler():
            """
            1) Call base_sampler() to get a batch (which may mix frame_ids).
            2) If mixed, keep the dominant fid's rays and pad to BS by sampling with
            replacement from the kept buckets. Return (bucket_ids, index_lists)
            in the SAME types as base_sampler (numpy index arrays).
            """
            bucket_ids, index_lists = base_sampler()  # index_lists: List[np.ndarray], one per bucket id

            # Determine the fid of each selected bucket in this batch
            batch_fids = [bucket2fid[b] for b in bucket_ids]
            unique_fids = list(set(batch_fids))

            if len(unique_fids) <= 1:
                # already frame-safe
                return bucket_ids, index_lists

            # --- Mixed batch: keep only dominant fid and pad ---
            # Group pairs by fid and count rays
            by_fid = defaultdict(list)
            counts = {}
            for b, sel in zip(bucket_ids, index_lists):
                f = bucket2fid[b]
                by_fid[f].append((b, sel))
            for f, pairs in by_fid.items():
                counts[f] = sum(int(len(sel)) for _, sel in pairs)

            # dominant fid = the one contributing the most rays in this batch
            dom_fid = max(counts.items(), key=lambda kv: kv[1])[0]
            kept_pairs = by_fid[dom_fid]

            kept = sum(int(len(sel)) for _, sel in kept_pairs)
            if kept == 0:
                # pathological, resample a fresh batch until not mixed (few tries), then return
                for _ in range(8):
                    bucket_ids, index_lists = base_sampler()
                    batch_fids = [bucket2fid[b] for b in bucket_ids]
                    if len(set(batch_fids)) <= 1:
                        return bucket_ids, index_lists
                # ultimate fallback: take the first pair and duplicate to BS
                b0, sel0 = bucket_ids[0], index_lists[0]
                need = BS - len(sel0)
                if need > 0:
                    extra = rng.randint(0, bucket_lengths[b0], size=need, dtype=np.int64)
                    sel0 = np.concatenate([sel0, extra], axis=0)
                return [b0], [sel0.astype(np.int64, copy=False)]

            # Pad the batch to BS with replacement from the kept buckets
            need = BS - kept
            if need > 0:
                # choose buckets to draw from, weighted by their total size (stable choice)
                kept_bs = [b for b, _ in kept_pairs]
                weights = np.asarray([bucket_lengths[b] for b in kept_bs], dtype=np.float64)
                if weights.sum() == 0:
                    weights = np.ones_like(weights, dtype=np.float64)
                weights /= weights.sum()

                # aggregate additions per bucket
                add_counts = defaultdict(int)
                draws = rng.choice(kept_bs, size=need, p=weights)
                for b in draws:
                    add_counts[int(b)] += 1

                # append new indices (with replacement) for each affected bucket
                new_pairs = []
                for b, sel in kept_pairs:
                    c = add_counts.get(int(b), 0)
                    if c > 0:
                        extra = rng.randint(0, bucket_lengths[b], size=c, dtype=np.int64)
                        sel = np.concatenate([sel, extra], axis=0)
                    new_pairs.append((b, sel.astype(np.int64, copy=False)))

                kept_pairs = new_pairs

            # Done; return in the same style as base sampler
            out_bucket_ids   = [b for b, _ in kept_pairs]
            out_index_lists  = [sel for _, sel in kept_pairs]
            return out_bucket_ids, out_index_lists

        # return buckets + our frame-safe sampler wrapper
        return rgb_l, ro_l, rd_l, vd_l, imsz_l, fid_l, frame_safe_sampler


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
        bucket_ids, index_lists = self.batch_sampler()
        rgb_chunks, ro_chunks, rd_chunks, vd_chunks, fid_chunks = [], [], [], [], []
        for b, sel in zip(bucket_ids, index_lists):
            rgb_chunks.append(self.rgb_l[b][sel])
            ro_chunks.append(self.ro_l[b][sel])
            rd_chunks.append(self.rd_l[b][sel])
            vd_chunks.append(self.vd_l[b][sel])
            fid_chunks.append(self.fid_l[b][sel])

        target = torch.cat(rgb_chunks, dim=0)
        ro     = torch.cat(ro_chunks,  dim=0)
        rd     = torch.cat(rd_chunks,  dim=0)
        vd     = torch.cat(vd_chunks,  dim=0)
        fid_b  = torch.cat(fid_chunks, dim=0).long()

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
                if WANDB:
                    wandb.log({
                        "train/psnr": float(psnr),
                        "train/loss": float(loss.item()),
                        "time/elapsed_s": dt,
                    }, step=step)

                # raise Exception("Stop here")

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
