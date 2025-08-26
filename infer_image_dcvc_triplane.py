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
import math

from TeTriRF.lib import dvgo, dmpigo, dvgo_video, dcvc_dvgo_video, utils      # unchanged
from TeTriRF.lib.load_data import load_data
from torch_efficient_distloss import flatten_eff_distloss
from TeTriRF.lib.dcvc_wrapper import collect_trainable_iframe_params, collect_trainable_sandwich_params


"""
Usage:
    python train_image_dcvc_triplane.py --config TeTriRF/configs/N3D/flame_steak_image_dcvc.py --frame_ids 0 --training_mode 1 --resume
    python train_image_dcvc_triplane.py --config TeTriRF/configs/N3D/flame_steak_image_dcvc_sandwich.py --frame_ids 0  --resume
    python train_image_dcvc_triplane.py --config TeTriRF/configs/N3D/flame_steak_image_dcvc_qp12.py --frame_ids 0 
"""

# ------------------------------------------------------------------------------
# 2. Argument & config handling
# ------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--frame_ids', nargs='+', type=int, help='List of frame IDs')
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--training_mode", type=int, default=1)
    # misc I/O
    p.add_argument("--i_print", type=int, default=1000)
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


@dataclass
class Inferer:
    """Stateful wrapper around the whole TeTriRF fine-stage training loop."""
    cfg: Config
    args: argparse.Namespace
    device: torch.device = field(init=False)
    model: dvgo_video.DirectVoxGO_Video = field(init=False)

    # bookkeeping
    _psnr_buffer: List[float] = field(default_factory=list, init=False)
    _tic: float = field(default_factory=time.time, init=False)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()
        self.qp = self.cfg.fine_train.dcvc_qp

    def _build_model(self):
        xyz_min = torch.tensor(self.cfg.data.xyz_min)
        xyz_max = torch.tensor(self.cfg.data.xyz_max)
        ids      = [0]
       
        self.model = dcvc_dvgo_video.DCVC_DVGO_Video(
            ids, xyz_min, xyz_max, self.cfg, 
            dcvc_qp = self.cfg.fine_train.dcvc_qp, freeze_dcvc_enc=self.cfg.fine_model_and_render.freeze_dcvc_enc, 
            freeze_dcvc_dec=self.cfg.fine_model_and_render.freeze_dcvc_dec, convert_ycbcr=self.cfg.fine_model_and_render.convert_ycbcr, 
            infer_mode=True
        ).to(self.device)

        _ = self.model.load_checkpoints()

    def _infer_step(self):

        _, bits_dict = self.model.run_codec_once()
        total_bits = 0
        for k in bits_dict.keys():
            total_bits += bits_dict[k]
        return _, total_bits

    def infer(self):
        cfg_t = self.cfg.fine_train
        os.makedirs(os.path.join(self.cfg.basedir, self.cfg.expname), exist_ok=True)
        out_root = os.path.join(self.cfg.basedir, self.cfg.expname)

        plane_psnr_dict, total_bits = self._infer_step()
        print(f"Total bits: {total_bits}")
        print(plane_psnr_dict)

        bits_path = os.path.join(out_root, f"encoded_bits_{self.qp}.txt")
        with open(bits_path, "w") as f:
            f.write(str(int(total_bits)) + "\n")

        self.model.save_infer_checkpoints(self.qp)
        print('Compression finished.')

# ------------------------------------------------------------------------------
# 5. Entry point
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    cfg  = Config.fromfile(args.config)
    cfg.data.frame_ids = args.frame_ids


    seed_everything(args.seed)

    if not args.render_only:
        Inferer(cfg, args).infer()