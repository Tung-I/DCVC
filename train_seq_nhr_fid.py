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
from TeTriRF.lib.load_data_NHR import load_data
from torch_efficient_distloss import flatten_eff_distloss

from src.data_loader.sampler import MultiBucketCycleSampler

"""
Usage:
    python train_seq_nhr_fid.py --config configs/NHR/sport1.py \
        --frame_ids 0 1 2 3 4 5 6 7 8 9  \
            --training_mode 1  --eval_train

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
    # For testing
    p.add_argument("--i_eval",  type=int, default=100, help="evaluate test PSNR every N steps")
    p.add_argument("--eval_factor", type=int, default=0,  help="downsample factor for eval (0 = fullres)")
    p.add_argument("--save_best_only", action="store_true",
                   help="save checkpoints only when test PSNR improves")
    p.add_argument("--eval_train", action="store_true",
                   help="use train PSNR (EMA) as the metric for saving the best ckpt")
    p.add_argument("--train_psnr_ema", type=float, default=0.98,
                   help="EMA decay for train PSNR used when --eval_train is on")
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

        self.best_test_psnr = float("-inf")
        self.best_step = -1

        self.best_train_psnr = float("-inf")
        self.train_psnr_ema  = None

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

        # NEW ↓ keep a smooth training metric for saving best
        v = float(psnr.item())
        if self.train_psnr_ema is None:
            self.train_psnr_ema = v
        else:
            m = float(getattr(self.args, "train_psnr_ema", 0.98))
            self.train_psnr_ema = m * self.train_psnr_ema + (1.0 - m) * v

        if cfg_t.weight_entropy_last > 0:
            p = render['alphainv_last'].clamp(1e-6,1-1e-6)
            loss += cfg_t.weight_entropy_last * (-(p*torch.log(p)+(1-p)*torch.log(1-p))).mean()

        if cfg_t.weight_distortion > 0:
            loss += cfg_t.weight_distortion * flatten_eff_distloss(
                render['weights'], render['s'], 1/render['n_max'], render['ray_id'])


        loss += self.cfg.fine_train.weight_l1_loss * self.model.compute_k0_l1_loss(fid_batch)
        return loss


    # -------------------------------------------------------------------------
    # Testing loop
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _eval_test_psnr(self, factor: int = 0, eval_all_fids: bool = False) -> float:
        """
        Evaluate mean PSNR on test views.
        - If eval_all_fids=False (default): mimic render.py by evaluating ONLY the
        "current" frame (take the last id in cfg.data.frame_ids).
        - If eval_all_fids=True: average across all test views for all fids.
        """
        self.model.eval()

        data, cfg, device = self.data, self.cfg, self.device
        i_test = np.asarray(data['i_test'])
        if i_test.size == 0:
            self.model.train(); return float("-inf")

        frame_ids_all = data['frame_ids']
        # ----- choose which fids to evaluate -----
        if eval_all_fids:
            test_mask = np.ones_like(i_test, dtype=bool)
        else:
            # match render.py: evaluate the "current" frame_id (last in the list)
            current_id = int(cfg.data.frame_ids[-1])
            test_mask = (frame_ids_all[i_test].cpu().numpy() == current_id)

        i_test = i_test[test_mask]
        if i_test.size == 0:
            self.model.train(); return float("-inf")

        # Assemble per-view inputs
        HW    = np.asarray(data['HW'])[i_test].copy()
        Ks    = np.asarray(data['Ks'])[i_test].copy()
        poses = data['poses'][i_test]
        fids  = frame_ids_all[i_test].cpu().numpy()

        if factor and factor > 0:
            HW = (HW / factor).astype(int)
            Ks[:, :2, :3] /= factor

        stepsize = float(cfg.fine_model_and_render.stepsize)
        render_kwargs = dict(
            near=data['near'],
            far=data['far'],
            bg=1 if cfg.data.white_bkgd else 0,
            stepsize=stepsize,
            inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x,
            flip_y=cfg.data.flip_y,
            render_depth=True,
            rand_bkgd=False,                 # <-- IMPORTANT: deterministic background for eval
        )
        # If your model exposes a shared rgbnet, you can pass it here too:
        if hasattr(self.model, "shared_rgbnet") and self.model.shared_rgbnet is not None:
            render_kwargs["shared_rgbnet"] = self.model.shared_rgbnet

        psnrs = []
        chunk = 150480
        keys = ['rgb_marched']

        for i in range(len(i_test)):
            H, W = int(HW[i, 0]), int(HW[i, 1])
            K    = torch.tensor(Ks[i], dtype=torch.float32, device=device)
            c2w  = poses[i].to(device)

            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc,
                inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y
            )
            ro = rays_o.flatten(0, -2)
            rd = rays_d.flatten(0, -2)
            vd = viewdirs.flatten(0, -2)

            fid = torch.full((ro.shape[0],), int(fids[i]), device=device, dtype=torch.long)

            outs = []
            for s in range(0, ro.shape[0], chunk):
                ret = self.model(
                    ro[s:s+chunk], rd[s:s+chunk], vd[s:s+chunk],
                    frame_ids=fid[s:s+chunk], mode='feat', **render_kwargs
                )
                outs.append(torch.stack([ret[k] for k in keys], dim=0))
            rgb = torch.cat([o[0] for o in outs], dim=0).view(H, W, -1)

            # Ground truth
            gt = data['images'][i_test[i]]
            if isinstance(gt, torch.Tensor):
                gt = gt.cpu().numpy()
            pred = rgb.clamp(0, 1).detach().cpu().numpy()

            # Optional mask
            if data.get('masks', None) is not None:
                m = data['masks'][i_test[i]]
                if isinstance(m, torch.Tensor):
                    m = m.cpu().numpy()
                sel = (m[..., 0] > 0.5)
                mse = np.mean((pred[sel] - gt[sel])**2) if sel.any() else np.mean((pred - gt)**2)
            else:
                mse = np.mean((pred - gt)**2)

            psnrs.append(-10.0 * np.log10(max(mse, 1e-10)))

        self.model.train()
        return float(np.mean(psnrs)) if len(psnrs) else float("-inf")

    def _maybe_eval_and_save(self, step: int):
        """Depending on args.eval_train, save best by train PSNR (EMA) or test PSNR."""
        if self.args.i_eval <= 0:
            return
        if step % self.args.i_eval != 0:
            return

        if self.args.eval_train:
            # --- Use TRAIN PSNR (EMA) as metric ---
            metric = float(self.train_psnr_ema) if self.train_psnr_ema is not None else float("-inf")
            if WANDB:
                wandb.log({"train/psnr_ema": metric}, step=step)

            if metric > self.best_train_psnr:
                self.best_train_psnr = metric
                self.best_step = step
                self.model.save_checkpoints()

                outdir = os.path.join(self.cfg.basedir, self.cfg.expname)
                with open(os.path.join(outdir, "BEST_TRAIN_PSNR.txt"), "w") as f:
                    f.write(f"step={step} psnr_ema={metric:.4f}\n")
                if WANDB:
                    wandb.run.summary["best_train_psnr"] = metric
                    wandb.run.summary["best_step"] = int(step)
        else:
            # --- Use TEST PSNR (your original behavior) ---
            test_psnr = self._eval_test_psnr(factor=self.args.eval_factor)
            if WANDB:
                wandb.log({"test/psnr": float(test_psnr)}, step=step)

            if test_psnr > self.best_test_psnr:
                self.best_test_psnr = test_psnr
                self.best_step = step
                self.model.save_checkpoints()

                outdir = os.path.join(self.cfg.basedir, self.cfg.expname)
                with open(os.path.join(outdir, "BEST_PSNR.txt"), "w") as f:
                    f.write(f"step={step} psnr={test_psnr:.4f}\n")
                if WANDB:
                    wandb.run.summary["best_test_psnr"] = float(test_psnr)
                    wandb.run.summary["best_step"] = int(step)

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

                if WANDB and self.train_psnr_ema is not None:
                    wandb.log({"train/psnr_ema": float(self.train_psnr_ema)}, step=step)


            # Evaluate & save best (only)
            self._maybe_eval_and_save(step)

            # # Optional: keep an occasional snapshot regardless of performance
            # if (not self.args.save_best_only) and (step % cfg_t.save_every == 0) and step >= cfg_t.save_after:
            #     self.model.save_checkpoints()

        # final eval/save
        self._maybe_eval_and_save(cfg_t.N_iters)
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
