from __future__ import annotations
import sys
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
from src.models import STE_DVGO_Sandwich_Image
# setup_unique_torch_extensions_dir()

from TeTriRF.lib.utils import debug_print_param_status
from TeTriRF.lib import dvgo, dvgo_video, dcvc_dvgo_video, utils      # unchanged
from TeTriRF.lib.load_data import load_data
from torch_efficient_distloss import flatten_eff_distloss

from src.models.model_utils import normalize_planes


WANDB=True

"""
Usage:
    python train_codec_nerf_sandwich.py --config configs/dynerf_sear_steak/dcvc_qp36_gradbpp_sandwich.py --frame_ids 0  --debug_dump_once
    python train_codec_nerf_sandwich.py --config configs/dynerf_sear_steak/dcvc_qp36_gradbpp_sandwich.py --frame_ids 0  --debug_dump_after_backward
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
    p.add_argument("--debug_dump_once", action="store_true",
               help="Dump trainability/grad info to txt and exit at the next step")
    p.add_argument("--debug_dump_after_backward", action="store_true",
                help="Also compute a backward to record grad norms before exiting")
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
        # Build model first; we’ll create the optimizer with PHASE control below
        self._build_model()     # <— split out building model only (see below)
        self._build_rays()
        self.lambda_bpp = self.qp_to_lambda()

        # Warmup + base LRs
        self.warmup_steps = int(getattr(self.cfg.fine_train, "sandwich_warmup_steps", 20000))
        ft = self.cfg.fine_train
        self._base_lrs = {
            "density": float(getattr(ft, "lrate_density", None)),
            "k0":      float(getattr(ft, "lrate_k0", None)),
            "rgbnet":  float(getattr(ft, "lrate_rgbnet", None)),
            "sandwich":float(getattr(ft, "lrate_sandwich", None)),
        }

        # Start in WARMUP phase (only sandwich is trainable)
        self._make_optimizer(phase="warmup", step=0)

        if getattr(self.args, "debug_dump_once", False) and not getattr(self, "_already_dumped", False):
            out = os.path.join(self.cfg.basedir, self.cfg.expname, "debug_trainable_step0.txt")
            self._dump_trainability(step=0, path=out, after_backward=False)
            print(f"[DEBUG] wrote {out}; exiting by request.")
            sys.exit(0)

        if WANDB:
            wandb.watch(self.model, log="all", log_freq=self.args.i_log)


    def _build_model(self):
        xyz_min = torch.tensor(self.cfg.data.xyz_min)
        xyz_max = torch.tensor(self.cfg.data.xyz_max)
        ids     = torch.unique(self.data['frame_ids']).cpu().tolist()

        if self.cfg.codec.train_mode == 'ste':
            self.model = STE_DVGO_Sandwich_Image(ids, xyz_min, xyz_max, self.cfg).to(self.device)
        else:
            raise NotImplementedError(f"train_mode {self.cfg.codec.train_mode} not implemented")

        if self.cfg.ckptname:
            _ = self.model.load_checkpoints()

        # Cache a handle to sandwich params (if present)
        self._sandwich_params = []
        if hasattr(self.model, "codec") and hasattr(self.model.codec, "sandwich_parameters"):
            self._sandwich_params = list(self.model.codec.sandwich_parameters())

    def _set_triplane_requires_grad(self, enable: bool):
        """Toggle k0+density requires_grad."""
        for fid, dvgo in self.model.dvgos.items():
            # density
            if hasattr(dvgo, 'density'):
                for p in dvgo.density.parameters():
                    p.requires_grad_(enable)
            # k0
            if hasattr(dvgo, 'k0'):
                for p in dvgo.k0.parameters():
                    p.requires_grad_(enable)

    def _make_optimizer(self, phase: str, step: int):
        """
        Rebuild optimizer to reflect training phase.
        phase: "warmup" -> only sandwich trainable, TriPlane frozen
            "joint"  -> sandwich + TriPlane trainable
        """
        # Build a shallow copy of cfg.fine_train with phase-specific LRs
        ft = copy.deepcopy(self.cfg.fine_train)

        if phase == "warmup":
            # Freeze TriPlane by setting their LRs to 0 and requires_grad=False
            ft.lrate_density = 0.0
            ft.lrate_k0 = 0.0
            ft.lrate_rgbnet = 0.0
            self._set_triplane_requires_grad(False)
        elif phase == "joint":
            # Restore base LRs and enable grads on TriPlane
            ft.lrate_density = self._base_lrs["density"]
            ft.lrate_k0      = self._base_lrs["k0"]
            ft.lrate_rgbnet  = self._base_lrs["rgbnet"]
            self._set_triplane_requires_grad(True)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Always keep DCVC frozen (utils does that) and add sandwich params if present
        sandwich_params = self._sandwich_params
        codec_params = []  # keep DCVC core frozen

        # Recreate optimizer with current step so LR decay stays consistent
        self.optimizer = utils.create_optimizer_sandwich(
            self.model, ft, global_step=step, codec_params=codec_params, sandwich_params=sandwich_params
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

    def qp_to_lambda(self):
        # assert that only one of dcvc_qp and jpeg.quality is set
        dcvc_qp = self.cfg.codec.dcvc_qp
        quality = self.cfg.codec.quality
        assert (dcvc_qp is None) != (quality is None)
        lambda_min = self.cfg.fine_train.lambda_min
        lambda_max = self.cfg.fine_train.lambda_max


        if dcvc_qp is not None:
            lambda_val = math.log(lambda_min) + dcvc_qp / (64 - 1) * (
                    math.log(lambda_max) - math.log(lambda_min))
            lambda_val = math.pow(math.e, lambda_val)
        elif quality is not None:
            qp = quality
            lambda_val = math.log(lambda_min) + qp / (100 - 1) * (
                    math.log(lambda_max) - math.log(lambda_min))
            lambda_val = math.pow(math.e, lambda_val)
        else:
            raise NotImplementedError("Only DCVC and JPEG codecs are supported.")

        return lambda_val

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

        bpp_loss = self.lambda_bpp * avg_bpp

        return rec_loss, bpp_loss

    def _sandwich_identity_loss(
        self,
        frame_id: int,
    ) -> torch.Tensor:
        """
        Codec-free: trains ONLY the sandwich pre/post.
        Uses the currently-active pack/unpack functions of self.model.codec.
        Returns a scalar loss (float tensor on correct device).
        """
        # Preconditions
        if not getattr(self.cfg.codec, "use_sandwich", False):
            raise ValueError("sandwich_identity_loss: sandwich not enabled")
        w = float(getattr(self.cfg.codec, "weight_sandwich_id", 0.0))

        # Access the planes of THIS frame (the real nn.Parameters)
        planes = self.model._gather_planes_one_frame(frame_id)  # {'xy','xz','yz'} -> (1,C,H,W)
        codec  = self.model.codec

        loss = 0.0
        for ax in ('xy', 'xz', 'yz'):
            x = planes[ax]  # (1,C,H,W), raw-domain
            # 1) normalize to [0,1] (same as your forward path)
            x01, _, _ = normalize_planes(
                x, mode=codec.quant_mode, global_range=codec.global_range
            )  # [1,C,H,W]

            # 2) learned pack (pre) + pad
            y_pad, orig_size = codec.pack_fn(x01, mode=codec.packing_mode)

            # 3) learned unpack (post) + crop
            x01_rec = codec.unpack_fn(y_pad, x01.shape[1], orig_size, mode=codec.packing_mode)

            # 4) identity loss in normalized domain
            loss = loss + torch.mean((x01_rec - x01) ** 2)

        return w * loss / 3.0  # average over 3 planes


    def _map_param_to_group_and_lr(self):
        """Build a dict id(param)->(group_index, lr) for fast lookup."""
        id2glr = {}
        for gi, g in enumerate(self.optimizer.param_groups):
            lr = g.get('lr', None)
            # Make sure we handle lists/tuples
            plist = g['params']
            if not isinstance(plist, (list, tuple)):
                plist = list(plist)
            for p in plist:
                id2glr[id(p)] = (gi, lr)
        return id2glr

    def _summarize_param(self, name: str, p: torch.nn.Parameter, id2glr, sandwich_idset):
        gid, lr = id2glr.get(id(p), (None, None))
        which = []
        if "density" in name: which.append("density")
        if "k0."      in name: which.append("k0")
        if id(p) in sandwich_idset: which.append("SANDWICH")
        if not which: which.append("other")

        return {
            "name": name,
            "shape": tuple(p.shape),
            "dtype": str(p.dtype),
            "requires_grad": bool(p.requires_grad),
            "in_optim": gid is not None,
            "group": gid,
            "lr": (None if lr is None else float(lr)),
            "tag": "+".join(which),
            # Grad fields filled later (optional)
            "has_grad": (p.grad is not None),
            "grad_norm": (float(p.grad.norm().item()) if p.grad is not None else None),
        }

    def _aggregate_module_trainability(self):
        """
        Aggregate trainability per *module* using only its DIRECT parameters
        (i.e., parameters it owns; children are not counted here). This makes
        it easy to see which logical blocks (codec.pre_unet, dvgos.0.k0, etc.)
        are actually optimizable right now.
        Returns: dict mname -> stats dict
        """
        id2glr = self._map_param_to_group_and_lr()

        mod_stats = {}  # mname -> dict
        for mname, module in self.model.named_modules():
            # DIRECT params only
            direct = [(n, p) for n, p in module.named_parameters(recurse=False)]
            if not direct:
                continue

            st = {
                "n_params": 0,
                "n_reqgrad": 0,
                "n_inopt": 0,
                "any_opt_lr_pos": False,
                "any_in_opt": False,
                "any_reqgrad": False,
                "lrs": set(),            # distinct LRs seen for direct params
                "examples": [],          # up to a few param names
            }

            for pn, p in direct:
                numel = p.numel()
                st["n_params"] += numel
                if p.requires_grad:
                    st["n_reqgrad"] += numel
                    st["any_reqgrad"] = True
                gid, lr = id2glr.get(id(p), (None, None))
                if gid is not None:
                    st["n_inopt"] += numel
                    st["any_in_opt"] = True
                    if lr is not None:
                        st["lrs"].add(float(lr))
                        if float(lr) > 0.0:
                            st["any_opt_lr_pos"] = True

                if len(st["examples"]) < 3:
                    st["examples"].append(pn)

            mod_stats[mname] = st

        return mod_stats


    def _dump_trainability(self, step: int, path: str, after_backward: bool = False):
        """
        Writes a detailed snapshot of trainability and optimizer state.
        Now prints two groups:
        - OPTIMIZABLE MODULES (in optimizer, lr>0, requires_grad=True)
        - FROZEN/INACTIVE MODULES (have direct params but don't satisfy above)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 1) Collect per-module stats
        mod_stats = self._aggregate_module_trainability()

        # 2) Build lists
        optimizable = []
        frozen = []

        for mname, st in mod_stats.items():
            is_optimizable = (st["any_in_opt"] and st["any_opt_lr_pos"] and st["any_reqgrad"])
            if is_optimizable:
                optimizable.append((mname, st))
            else:
                # Explain why it's not optimizable
                flags = []
                if not st["any_in_opt"]:
                    flags.append("no-opt")
                else:
                    if not st["any_opt_lr_pos"]:
                        flags.append("lr=0")
                if not st["any_reqgrad"]:
                    flags.append("reqgrad=False")
                st["flags"] = ",".join(flags) if flags else "-"
                frozen.append((mname, st))

        # Helpful counts
        def _count_params(lst): return sum(s["n_params"] for _, s in lst)

        # 3) Write the report
        with open(path, "w") as f:
            f.write(f"==== DEBUG TRAINABILITY DUMP @ step {step} ====\n")

            # Optimizer groups summary
            f.write(f"optimizer groups: {len(self.optimizer.param_groups)}\n")
            for gi, g in enumerate(self.optimizer.param_groups):
                lr = g.get("lr", None)
                npar = len(g["params"]) if isinstance(g["params"], (list, tuple)) else "?"
                f.write(f"  - group[{gi}]: lr={lr}  num_params={npar}\n")
            f.write("\n")

            # Summary counts
            total_mods = len(mod_stats)
            f.write(f"modules_with_params: {total_mods}\n")
            f.write(f"optimizable_modules: {len(optimizable)} (params={_count_params(optimizable)})\n")
            f.write(f"frozen_modules     : {len(frozen)} (params={_count_params(frozen)})\n\n")

            # OPTIMIZABLE LIST
            f.write("==== OPTIMIZABLE MODULES (in optimizer + lr>0 + requires_grad) ====\n")
            for mname, st in sorted(optimizable, key=lambda x: x[0]):
                lrs_str = ",".join(f"{lr:.3e}" for lr in sorted(st["lrs"])) if st["lrs"] else "-"
                ex_str = ", ".join(st["examples"]) if st["examples"] else "-"
                f.write(f"{mname:<40s}  n_params={st['n_params']:>9d}  "
                        f"n_inopt={st['n_inopt']:>9d}  n_reqgrad={st['n_reqgrad']:>9d}  "
                        f"lrs={lrs_str:<20s}  ex=[{ex_str}]\n")
            f.write("\n")

            # FROZEN LIST
            f.write("==== FROZEN / INACTIVE MODULES ====\n")
            f.write("# flags: no-opt (not in optimizer), lr=0 (in optimizer but zero LR), reqgrad=False\n")
            for mname, st in sorted(frozen, key=lambda x: x[0]):
                lrs_str = ",".join(f"{lr:.3e}" for lr in sorted(st["lrs"])) if st["lrs"] else "-"
                ex_str = ", ".join(st["examples"]) if st["examples"] else "-"
                flags = st.get("flags", "-")
                f.write(f"{mname:<40s}  n_params={st['n_params']:>9d}  "
                        f"n_inopt={st['n_inopt']:>9d}  n_reqgrad={st['n_reqgrad']:>9d}  "
                        f"lrs={lrs_str:<20s}  flags=[{flags}]  ex=[{ex_str}]\n")
            f.write("\n")

            # (Optional) keep your per-parameter table for deep dives
            id2glr = self._map_param_to_group_and_lr()
            sandwich_set = set(id(p) for p in getattr(self, "_sandwich_params", []))
            f.write("---- per-parameter table (optional deep dive) ----\n")
            for name, p in self.model.named_parameters():
                gid, lr = id2glr.get(id(p), (None, None))
                which = []
                if "density" in name: which.append("density")
                if "k0." in name:     which.append("k0")
                if id(p) in sandwich_set: which.append("SANDWICH")
                if not which: which.append("other")
                has_grad = p.grad is not None if after_backward else False
                gnorm = (p.grad.norm().item() if p.grad is not None else None)
                f.write(
                    f"{name:<80s}  "
                    f"shape={tuple(p.shape)!s:<18s} "
                    f"dtype={str(p.dtype):<12s} "
                    f"req_grad={str(p.requires_grad):<5s} "
                    f"in_opt={str(gid is not None):<5s} "
                    f"group={str(gid):<3s} "
                    f"lr={(None if lr is None else float(lr))!s:<12s} "
                    f"tag={'+'.join(which):<18s} "
                    f"grad={str(has_grad):<5s} "
                    f"g_norm={('-' if gnorm is None else f'{gnorm:.3e}')}\n"
                )



    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(self):
        cfg_t = self.cfg.fine_train
        os.makedirs(os.path.join(self.cfg.basedir, self.cfg.expname), exist_ok=True)

        for step in trange(1, cfg_t.N_iters+1):
            # Phase switch (warmup -> joint) right at boundary
            if step == self.warmup_steps:
                print(f"[phase switch] Entering JOINT training at step {step}.")
                self._make_optimizer(phase="joint", step=step)

            if (step+500) % 1000 == 0:
                self.model.update_occupancy_cache()

            loss, plane_psnr_dict, avg_bpp, rec_loss, bpp_loss = self._train_step(step)

            if getattr(self.args, "debug_dump_after_backward", False) and not getattr(self, "_already_dumped", False):
                out = os.path.join(self.cfg.basedir, self.cfg.expname, f"debug_after_backward_s{step}.txt")
                self._dump_trainability(step=step, path=out, after_backward=True)
                print(f"[DEBUG] wrote {out}; exiting by request.")
                sys.exit(0)

            # Per-step LR decay (kept as you had it)
            decay = 0.1 ** (1 / (cfg_t.lrate_decay*1000))
            for g in self.optimizer.param_groups:
                g['lr'] *= decay

            if step % self.args.i_print == 0 or step == 1:
                dt = time.time() - self._tic
                psnr = np.mean(self._psnr_buffer); self._psnr_buffer.clear()
                tqdm.write(f'[step {step:6d}] loss {loss.item():.4e} psnr {psnr:5.2f} '
                        f'elapsed {dt/3600:02.0f}:{dt/60%60:02.0f}:{dt%60:02.0f}')

            if step % self.args.i_log == 0 and WANDB:
                wandb.log({
                    "train/psnr": float(psnr),
                    "train/loss": float(loss.item()),
                    "train/xy_plane_psnr": float(plane_psnr_dict['xy']),
                    "train/density_psnr": float(plane_psnr_dict['density']),
                    "train/rec_loss": float(rec_loss.item()),
                    "train/bpp_loss": float(bpp_loss.item()),
                    "train/total_bpp": float(avg_bpp),
                    "train/phase": "joint" if step >= self.warmup_steps else "warmup",
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