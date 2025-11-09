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
from TeTriRF.lib.load_data_NHR import load_data
from torch_efficient_distloss import flatten_eff_distloss

from src.data_loader.sampler import MultiBucketCycleSampler

WANDB=True

"""
Usage:
    python train_codec_nerf_video_nhr.py --config configs/nhr_sport1/av1_qp44.py --frame_ids 0 1 2 3 4 5 6 7 8 9 --training_mode 1
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

        # --- Build bucket -> fid map once (each bucket has a constant fid) ---
        bucket2fid = []
        for f_b in fid_l:
            bucket2fid.append(int(f_b[0].item()) if isinstance(f_b, torch.Tensor) else int(np.asarray(f_b)[0]))

        # ------------------ Frame-stable buffering sampler -------------------
        # Never drop or duplicate rays. We buffer what base_sampler returns
        # and only emit a batch once a single fid has >= BS rays available.
        import numpy as np
        from collections import defaultdict, deque

        # pending[fid][bucket] = deque of np.ndarray chunks of indices to use (FIFO)
        pending = { }
        pending_count = defaultdict(int)  # total rays buffered per fid

        def _ensure_fid(fid):
            if fid not in pending:
                pending[fid] = defaultdict(deque)

        def _append_chunk(fid, b, idx_np):
            if idx_np.size == 0:
                return
            _ensure_fid(fid)
            pending[fid][b].append(idx_np)          # keep chunk boundaries as produced
            pending_count[fid] += int(idx_np.size)

        def _pop_batch_for_fid(fid, target_bs):
            """Assemble one homogeneous batch for this fid (no replacement, no drop)."""
            by_bucket = pending[fid]
            need = target_bs
            out_bucket_ids, out_index_lists = [], []

            # round-robin over buckets of this fid while need > 0
            for b in list(by_bucket.keys()):
                while need > 0 and by_bucket[b]:
                    chunk = by_bucket[b][0]  # peek
                    take  = min(need, chunk.shape[0])
                    sel   = chunk[:take]
                    out_bucket_ids.append(b)
                    out_index_lists.append(sel.astype(np.int64, copy=False))

                    if take == chunk.shape[0]:
                        by_bucket[b].popleft()
                    else:
                        by_bucket[b][0] = chunk[take:]  # keep remainder

                    need -= take
                    pending_count[fid] -= int(take)

                    # early-exit if batch complete
                    if need == 0:
                        break

            # Note: It’s possible a few small chunks leave some buckets empty; that’s fine.
            # We return exactly target_bs rays, all from one fid, assembled strictly from the buffered deck.
            return out_bucket_ids, out_index_lists

        def frame_stable_sampler():
            # Keep pulling from base_sampler until SOME fid has enough to form a full batch.
            while True:
                # Is there any fid ready?
                for fid_ready, tot in list(pending_count.items()):
                    if tot >= BS:
                        return _pop_batch_for_fid(fid_ready, BS)

                # Not ready yet → pull another mixed batch from the original sampler and buffer it.
                bucket_ids, index_lists = base_sampler()  # index_lists: List[np.ndarray]
                for b, sel in zip(bucket_ids, index_lists):
                    fid_here = bucket2fid[b]
                    _append_chunk(fid_here, b, np.asarray(sel, dtype=np.int64))

                # loop and re-check readiness

        # return buckets + our frame-stable sampler wrapper
        return rgb_l, ro_l, rd_l, vd_l, imsz_l, fid_l, frame_stable_sampler

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
        return 0.0

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