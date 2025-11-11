"""
4-in-1 offline evaluation (no intermediate files):
  ckpt → pack → video codec roundtrip → unpack → render_test (PSNR/SSIM/LPIPS)

Writes ONLY:
  <outdir>/
    encoded_bits.txt
    render_test/{fid}_*.png and metric txts

Quantization modes (appearance planes):
  - absmax : fixed [-20, 20] (same as your old "global")
  - affine : robust per-channel affine via percentiles over selected frames

Example:


python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flatten --grid_packing_mode flatten \
  --qmode affine \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flatten_affine \
  --config configs/dynerf_flame_steak/av1_qp52_flatten_affine.py 

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flatten --grid_packing_mode flatten \
  --qmode affine \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flatten_affine_tv \
  --config configs/dynerf_flame_steak/av1_qp52_flatten_affine_tv.py 

  python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flat4 --grid_packing_mode flatten \
  --qmode affine \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flat4_affine \
  --config configs/dynerf_flame_steak/av1_qp52_flat4_affine.py 

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flat4 --grid_packing_mode flatten \
  --qmode affine \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flat4_affine_tv \
  --config configs/dynerf_flame_steak/av1_qp52_flat4_affine_tv.py 

  python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode mosaic --grid_packing_mode flatten \
  --qmode affine \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_mosaic_affine \
  --config configs/dynerf_flame_steak/av1_qp52_mosaic_affine.py 

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode mosaic --grid_packing_mode flatten \
  --qmode affine \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_mosaic_affine_tv \
  --config configs/dynerf_flame_steak/av1_qp52_mosaic_affine_tv.py

  

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flatten --grid_packing_mode flatten \
  --qmode absmax \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flatten_absmax \
  --config configs/dynerf_flame_steak/av1_qp52_flatten_absmax.py 

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flatten --grid_packing_mode flatten \
  --qmode absmax \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flatten_absmax_tv \
  --config configs/dynerf_flame_steak/av1_qp52_flatten_absmax_tv.py 

  python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flat4 --grid_packing_mode flatten \
  --qmode absmax \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flat4_absmax \
  --config configs/dynerf_flame_steak/av1_qp52_flat4_absmax.py 

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flat4 --grid_packing_mode flatten \
  --qmode absmax\
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_flat4_absmax_tv \
  --config configs/dynerf_flame_steak/av1_qp52_flat4_absmax_tv.py 

  python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode mosaic --grid_packing_mode flatten \
  --qmode absmax \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_mosaic_absmax \
  --config configs/dynerf_flame_steak/av1_qp52_mosaic_absmax.py 

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode mosaic --grid_packing_mode flatten \
  --qmode absmax \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_mosaic_absmax_tv \
  --config configs/dynerf_flame_steak/av1_qp52_mosaic_absmax_tv.py

  

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 10 \
  --plane_packing_mode flatten --grid_packing_mode flatten \
  --qmode absmax \
  --codec av1 --qp 62 --gop 10 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp62_gop10_tv \
  --config configs/dynerf_flame_steak/av1_qp62_gop10_tv.py 

python eval_4in1_video_rd.py \
  --startframe 0 --numframe 20 \
  --plane_packing_mode flatten --grid_packing_mode flatten \
  --qmode absmax \
  --codec av1 --qp 52 --gop 20 --fps 30 --pix-fmt yuv444p \
  --ckpt_dir logs/dynerf_flame_steak/av1_qp52_k16_resume \
  --config configs/dynerf_flame_steak/av1_qp52_k16.py 

"""
import os, io, sys, copy, json, argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F

# ===== TeTriRF rendering deps (unchanged behavior) =====
from mmengine.config import Config
from TeTriRF.lib import utils, dvgo, dmpigo
from TeTriRF.lib.load_data import load_data
from TeTriRF.lib.dvgo_video import RGB_Net, RGB_SH_Net

# ===== EXACT same utilities used in training/packing =====
from src.models.model_utils import (
    DCVC_ALIGN,
    pack_planes_to_rgb, unpack_rgb_to_planes,
    pack_density_to_rgb, unpack_density_from_rgb,
    dens_to01, dens_from01,
    hevc_video_roundtrip, av1_video_roundtrip, vp9_video_roundtrip,
)

# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset / rendering config
    p.add_argument('--config', required=True, help='mmengine config used by TeTriRF')
    p.add_argument('--ckpt_dir', required=True, help='Folder of frame ckpts: fine_last_{fid}.tar')
    p.add_argument('--frame_ids', nargs='*', type=int, default=None,
                  help='Explicit frame id list. If not set, uses startframe..start+num-1.')
    p.add_argument('--startframe', type=int, default=0)
    p.add_argument('--numframe', type=int, default=1)

    # Packing & quant
    p.add_argument('--plane_packing_mode', choices=['flatten', 'mosaic', 'flat4'], default='flatten')
    p.add_argument('--grid_packing_mode',  choices=['flatten', 'mosaic', 'flat4'], default='flatten')
    p.add_argument('--qmode', choices=['absmax', 'affine'], default='absmax',
                   help="Quantization for appearance planes")

    # Video codec knobs
    p.add_argument('--codec', choices=['hevc','av1','vp9'], required=True)
    p.add_argument('--qp', type=int, required=True)
    p.add_argument('--gop', type=int, default=20)
    p.add_argument('--fps', type=int, default=30) 
    p.add_argument('--pix-fmt', type=str, default='yuv444p')
    p.add_argument('--align', type=int, default=32)

    # Render toggles
    p.add_argument('--render_test', action='store_true', default=True)
    p.add_argument('--dump_images', action='store_true', default=False)
    p.add_argument('--eval_ssim', action='store_true', default=True)
    p.add_argument('--eval_lpips_vgg', action='store_true', default=True)
    p.add_argument('--eval_lpips_alex', action='store_true', default=False)

    p.add_argument('--render_stride', type=int, default=1,
               help='Render every S-th frame from the selected frame_ids.')
    p.add_argument('--render_max_frames', type=int, default=5,
               help='If >0, render at most this many frames (after stride).')

    # Misc
    p.add_argument('--seed', type=int, default=777)
    return p.parse_args()

# ---------------------------------------------------------------------------------
# Shared helpers (ported from your scripts; unchanged math/behavior)
# ---------------------------------------------------------------------------------
# absmax (same as your old "global")
ABSMAX_LOW, ABSMAX_HIGH = -20.0, 20.0

# affine (robust bounds)
AFFINE_LO_P = 0.5     # percentile low  (0.5%)
AFFINE_HI_P = 99.5    # percentile high (99.5%)
AFFINE_EPS  = 1e-4    # min dynamic range per channel
AFFINE_MAX_SAMPLES_PER_CH = 2_000_000  # cap per-channel sample count across all frames

def _mse2psnr_with_peak(mse: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    return 10.0 * torch.log10((peak * peak) / (mse + 1e-12))

def set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)

def load_everything_for_render(args, cfg):
    data_dict = load_data(cfg.data)
    kept = {'hwf','HW','Ks','near','far','near_clip','i_train','i_val','i_test',
            'irregular_shape','poses','render_poses','images','frame_ids','masks'}
    for k in list(data_dict.keys()):
        if k not in kept: data_dict.pop(k)
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, frame_id=0, masks=None):
    from TeTriRF.lib import utils as Tutils
    assert len(render_poses) == len(HW) == len(Ks)
    if render_factor!=0:
        HW = (np.copy(HW)/render_factor).astype(int)
        Ks = np.copy(Ks); Ks[:, :2, :3] /= render_factor

    rgbs, depths, bgmaps = [], [], []
    psnrs, ssims, lpips_a, lpips_v = [], [], [], []

    for i, c2w in enumerate(tqdm(render_poses)):
        H,W = HW[i]; K = Ks[i]; c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H,W,K,c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched','depth','alphainv_last']
        rays_o = rays_o.flatten(0,-2); rays_d = rays_d.flatten(0,-2); viewdirs = viewdirs.flatten(0,-2)
        render_chunks = [
            {k:v for k,v in model(ro,rd,vd, **render_kwargs).items() if k in keys}
            for ro,rd,vd in zip(rays_o.split(150480,0), rays_d.split(150480,0), viewdirs.split(150480,0))
        ]
        render_out = {k: torch.cat([ret[k] for ret in render_chunks]).reshape(H,W,-1) for k in keys}
        rgb, depth, bgmap = [render_out['rgb_marched'].cpu().numpy(),
                             render_out['depth'].cpu().numpy(),
                             render_out['alphainv_last'].cpu().numpy()]
        rgbs.append(rgb); depths.append(depth); bgmaps.append(bgmap)

        if gt_imgs is not None and render_factor==0:
            if masks is not None:
                m = masks[i][...,0]>0.5
                p = -10.0*np.log10(np.mean((rgb[m,:]-gt_imgs[i][m,:])**2))
            else:
                p = -10.0*np.log10(np.mean((rgb-gt_imgs[i])**2))
            psnrs.append(p)
            if eval_ssim:        ssims.append(Tutils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:  lpips_a.append(Tutils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:   lpips_v.append(Tutils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    res = {}
    os.makedirs(savedir, exist_ok=True)
    if psnrs:
        res['psnr'] = float(np.mean(psnrs))
        with open(os.path.join(savedir, f'{frame_id}_psnr.txt'), 'w') as f: f.write(f"{res['psnr']}\n")
        if eval_ssim:
            res['ssim'] = float(np.mean(ssims))
            with open(os.path.join(savedir, f'{frame_id}_ssim.txt'), 'w') as f: f.write(f"{res['ssim']}\n")
        if eval_lpips_vgg:
            res['lpips'] = float(np.mean(lpips_v))
            with open(os.path.join(savedir, f'{frame_id}_lpips.txt'), 'w') as f: f.write(f"{res['lpips']}\n")

        if dump_images:
            import imageio
            for i in trange(len(rgbs)):
                # save RGB
                rgb8 = (np.clip(rgbs[i], 0, 1) * 255.0 + 0.5).astype(np.uint8)
                imageio.imwrite(os.path.join(savedir, f'{frame_id}_{i}.png'), rgb8)

                # --- robust depth visualization: ensure 3 channels ---
                d = depths[i]
                # squeeze trailing singleton channel if present
                if d.ndim == 3 and d.shape[-1] == 1:
                    d = d[..., 0]
                # normalize to [0,1] and invert for visualization
                d_norm = 1.0 - d / (np.max(d) + 1e-8)
                d8 = (np.clip(d_norm, 0, 1) * 255.0 + 0.5).astype(np.uint8)

                # make 3 channels for imageio/pillow (HxW -> HxWx3, or HxWx1 -> HxWx3)
                if d8.ndim == 2:
                    d8 = np.repeat(d8[..., None], 3, axis=-1)
                elif d8.ndim == 3 and d8.shape[-1] == 1:
                    d8 = np.repeat(d8, 3, axis=-1)

                imageio.imwrite(os.path.join(savedir, f'{frame_id}_{i}_depth.png'), d8)

    return res

# ---------------------------------------------------------------------------------
# Quantize / dequantize for feature planes
# ---------------------------------------------------------------------------------
def quantise_absmax_01(feat_1CHW: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[float,float]]]:
    """Fixed [-20,20] to [0,1], record identical bounds for all channels."""
    C = feat_1CHW.shape[1]
    lo, hi = ABSMAX_LOW, ABSMAX_HIGH
    x01 = (feat_1CHW - lo) / (hi - lo)
    bounds = [(lo, hi)] * C
    return x01.clamp_(0,1), bounds

def quantise_affine_01(feat_1CHW: torch.Tensor,
                       plane_bounds: List[Tuple[float,float]]) -> Tuple[torch.Tensor, List[Tuple[float,float]]]:
    """Per-channel affine using provided percentile bounds for this plane."""
    C = feat_1CHW.shape[1]
    assert plane_bounds is not None and len(plane_bounds) == C
    x01 = torch.empty_like(feat_1CHW)
    out_bounds: List[Tuple[float,float]] = []
    for c in range(C):
        lo, hi = plane_bounds[c]
        lo = float(lo); hi = float(hi)
        if hi - lo < AFFINE_EPS:  # safety
            hi = lo + AFFINE_EPS
        out_bounds.append((lo, hi))
        x01[:, c] = (feat_1CHW[:, c].clamp(lo, hi) - lo) / (hi - lo)
    return x01.clamp_(0,1), out_bounds

def dequantise_from_01(x01_1CHW: torch.Tensor, bounds: List[Tuple[float,float]]) -> torch.Tensor:
    C = x01_1CHW.shape[1]
    assert len(bounds)==C
    out = torch.empty_like(x01_1CHW)
    for c in range(C):
        lo, hi = bounds[c]
        out[:,c] = x01_1CHW[:,c]*(hi-lo)+lo
    return out

def gather_affine_bounds_for_planes(
    ckpt_dir: str, frame_ids: List[int], plane_names: List[str]
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Robust per-plane, per-channel bounds via percentiles (0.5%, 99.5%).
    CPU-only staging to avoid default CUDA tensor issues.
    """
    bounds: Dict[str, List[Tuple[float, float]]] = {}

    # First pass: get channel counts per plane from the first frame (CPU)
    first = torch.load(
        os.path.join(ckpt_dir, f"fine_last_{frame_ids[0]}.tar"),
        map_location="cpu", weights_only=False
    )
    feats0 = {
        k.split(".")[-1]: v.clone().to("cpu")
        for k, v in first["model_state_dict"].items()
        if "k0" in k and "plane" in k and "residual" not in k
    }
    C = {pn: feats0[pn].shape[1] for pn in plane_names}

    # Per-plane per-channel sample buckets (force CPU!)
    samples: Dict[str, List[torch.Tensor]] = {
        pn: [torch.empty(0, dtype=torch.float32, device="cpu") for _ in range(C[pn])]
        for pn in plane_names
    }

    # Cap total samples per channel across all frames
    per_frame_cap = max(1, AFFINE_MAX_SAMPLES_PER_CH // max(1, len(frame_ids)))

    for fid in tqdm(frame_ids, desc="[affine] sampling"):
        ckpt = torch.load(
            os.path.join(ckpt_dir, f"fine_last_{fid}.tar"),
            map_location="cpu", weights_only=False
        )
        feats = {
            k.split(".")[-1]: v.clone().to("cpu")
            for k, v in ckpt["model_state_dict"].items()
            if "k0" in k and "plane" in k and "residual" not in k
        }
        for pn in plane_names:
            f = feats[pn]  # [1,C,H,W] on CPU
            _, Cn, H, W = f.shape
            total = H * W

            # stride to cap per-frame samples
            stride = int(np.ceil(total / per_frame_cap))
            if stride < 1:
                stride = 1
            idx = torch.arange(0, total, stride, device="cpu")  # CPU index

            for c in range(Cn):
                ch = f[0, c].reshape(-1)            # CPU
                samp = ch.index_select(0, idx)      # CPU
                # append on CPU
                if samples[pn][c].numel() == 0:
                    samples[pn][c] = samp.to(dtype=torch.float32, device="cpu")
                else:
                    samples[pn][c] = torch.cat(
                        (samples[pn][c], samp.to(dtype=torch.float32, device="cpu")),
                        dim=0
                    )

    # Compute percentiles per channel (CPU)
    for pn in plane_names:
        b: List[Tuple[float, float]] = []
        for c_samp in samples[pn]:
            if c_samp.numel() == 0:
                lo, hi = ABSMAX_LOW, ABSMAX_HIGH  # fallback
            else:
                # ensure float32 CPU
                c_samp = c_samp.to(dtype=torch.float32, device="cpu")
                lo = torch.quantile(c_samp, AFFINE_LO_P / 100.0).item()
                hi = torch.quantile(c_samp, AFFINE_HI_P / 100.0).item()
                if hi - lo < AFFINE_EPS:
                    hi = lo + AFFINE_EPS
            b.append((lo, hi))
        bounds[pn] = b

    return bounds


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Resolve frame ids
    if args.frame_ids and len(args.frame_ids)>0:
        frame_ids = sorted(args.frame_ids)
    else:
        frame_ids = list(range(args.startframe, args.startframe + args.numframe))

    fid2idx = {fid: i for i, fid in enumerate(frame_ids)}

    render_frame_ids = frame_ids[::max(1, args.render_stride)]
    if args.render_max_frames > 0:
        render_frame_ids = render_frame_ids[:args.render_max_frames]

    # Output root
    out_root = os.path.abspath(os.path.join(args.ckpt_dir, '4in1test'))
    os.makedirs(out_root, exist_ok=True)
    render_out = os.path.join(out_root, "render_test")
    os.makedirs(render_out, exist_ok=True)

    # ------- Load cfg & dataset (unchanged behavior) -------
    cfg = Config.fromfile(args.config)
    if not hasattr(cfg.fine_model_and_render, 'dynamic_rgbnet'):
        cfg.fine_model_and_render.dynamic_rgbnet = True
    if not hasattr(cfg.fine_model_and_render, 'RGB_model'):
        cfg.fine_model_and_render.RGB_model = 'MLP'

    # IMPORTANT: match your render.py default device behavior
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # <- fixes reset_occupancy_cache CPU/CUDA mismatch

    # Dataset (kept on CPU like your render.py)
    data_dict = load_everything_for_render(args, cfg)

    # ====== Pre-pass for affine percentiles (if needed) ======
    per_plane_affine_bounds: Dict[str, List[Tuple[float,float]]] = {}
    # determine plane names from first frame
    ckpt0 = torch.load(os.path.join(args.ckpt_dir, f"fine_last_{frame_ids[0]}.tar"), map_location="cpu", weights_only=False)
    planes0 = {k.split(".")[-1]: v.clone()
               for k,v in ckpt0["model_state_dict"].items()
               if "k0" in k and "plane" in k and "residual" not in k}
    plane_names = sorted(planes0.keys())

    if args.qmode == "affine":
        per_plane_affine_bounds = gather_affine_bounds_for_planes(args.ckpt_dir, frame_ids, plane_names)

    # ====== Pass 1: pack all frames (in-memory) ======
    print("[INFO] Packing & quantizing (in-memory)…")
    feat_canv_by_plane: Dict[str, torch.Tensor] = {}     # plane -> [T,3,Hp,Wp]
    feat_orig_hw: Dict[str, Tuple[int,int]] = {}         # plane -> (H2,W2)
    feat_bounds_per_frame: Dict[str, List[List[Tuple[float,float]]]] = {pn: [] for pn in plane_names}
    dens_canv: Optional[torch.Tensor] = None             # [T,3,Hp_d,Wp_d]
    dens_orig_hw: Optional[Tuple[int,int]] = None        # (H2d,W2d)

    for fi, fid in enumerate(tqdm(frame_ids)):
        ckpt = torch.load(os.path.join(args.ckpt_dir, f"fine_last_{fid}.tar"), map_location="cpu", weights_only=False)

        # --- feature planes ---
        planes = {k.split(".")[-1]: v.clone()
                  for k,v in ckpt["model_state_dict"].items()
                  if "k0" in k and "plane" in k and "residual" not in k}

        for pn in plane_names:
            feat = planes[pn]  # [1,C,H,W]
            if args.qmode == "absmax":
                feat01, bnd = quantise_absmax_01(feat)
            elif args.qmode == "affine":
                feat01, bnd = quantise_affine_01(feat, per_plane_affine_bounds[pn])
            else:
                raise ValueError(f"Unknown qmode {args.qmode}")
            feat_bounds_per_frame[pn].append(bnd)

            canv_pad, orig_hw = pack_planes_to_rgb(feat01, align=args.align, mode=args.plane_packing_mode) # [1,3,Hp,Wp]
            if fi == 0:
                feat_canv_by_plane[pn] = canv_pad
                feat_orig_hw[pn] = tuple(orig_hw)
            else:
                feat_canv_by_plane[pn] = torch.cat([feat_canv_by_plane[pn], canv_pad], dim=0)

        # --- density grid ---
        density = ckpt["model_state_dict"]["density.grid"].clone()  # [1,1,Dy,Dx,Dz]
        dens_canv_pad, dens_orig = pack_density_to_rgb(density, align=args.align, mode=args.grid_packing_mode)
        if fi == 0:
            dens_canv = dens_canv_pad
            dens_orig_hw = tuple(dens_orig)
        else:
            dens_canv = torch.cat([dens_canv, dens_canv_pad], dim=0)

    # ====== Pass 2: one video roundtrip per stream (memory only) ======
    print(f"[INFO] Video codec roundtrip ({args.codec.upper()} QP={args.qp})…")
    total_bits = 0.0
    stream_stats: Dict[str, Dict[str,float]] = {}
    T = len(frame_ids)

    def _roundtrip(canv_3: torch.Tensor, use_gray: bool):
        if use_gray:
            mono = canv_3[:, :1].contiguous()  # [T,1,Hp,Wp]
            if args.codec == "hevc":
                rec_mono, bits = hevc_video_roundtrip(mono, fps=args.fps, gop=args.gop, qp=args.qp,
                                                      preset='medium', pix_fmt=args.pix_fmt, grayscale=True)
            elif args.codec == "av1":
                rec_mono, bits = av1_video_roundtrip(mono, fps=args.fps, gop=args.gop, qp=args.qp,
                                                     cpu_used='6', pix_fmt=args.pix_fmt, grayscale=True)
            else:
                rec_mono, bits = vp9_video_roundtrip(mono, fps=args.fps, gop=args.gop, qp=args.qp,
                                                     cpu_used='4', pix_fmt=args.pix_fmt, grayscale=True)
            rec = rec_mono.repeat(1,3,1,1)
        else:
            if args.codec == "hevc":
                rec, bits = hevc_video_roundtrip(canv_3, fps=args.fps, gop=args.gop, qp=args.qp,
                                                 preset='medium', pix_fmt=args.pix_fmt)
            elif args.codec == "av1":
                rec, bits = av1_video_roundtrip(canv_3, fps=args.fps, gop=args.gop, qp=args.qp,
                                                cpu_used='6', pix_fmt=args.pix_fmt)
            else:
                rec, bits = vp9_video_roundtrip(canv_3, fps=args.fps, gop=args.gop, qp=args.qp,
                                                cpu_used='4', pix_fmt=args.pix_fmt)
        return rec, int(bits)

    # Feature planes
    for pn, canv in feat_canv_by_plane.items():
        Hp, Wp = canv.shape[-2:]
        use_gray = (args.plane_packing_mode == "flatten")
        rec, bits = _roundtrip(canv, use_gray=use_gray)
        mse = F.mse_loss(rec, canv)
        psnr = float(_mse2psnr_with_peak(mse, peak=1.0))
        stream_stats[pn] = dict(T=T, Hp=Hp, Wp=Wp, total_bits=float(bits), bpp=float(bits)/(T*Hp*Wp), psnr=psnr)
        total_bits += float(bits)
        feat_canv_by_plane[pn] = rec  # overwrite with recon

    # Density stream (always grayscale packing)
    Hp_d, Wp_d = dens_canv.shape[-2:]
    rec_d, bits_d = _roundtrip(dens_canv, use_gray=True)
    mse_d = F.mse_loss(rec_d, dens_canv)
    psnr_d = float(_mse2psnr_with_peak(mse_d, peak=1.0))
    stream_stats['density'] = dict(T=T, Hp=Hp_d, Wp=Wp_d, total_bits=float(bits_d),
                                   bpp=float(bits_d)/(T*Hp_d*Wp_d), psnr=psnr_d)
    total_bits += float(bits_d)

    # ====== Pass 3: unpack & de-quantize to build in-memory planes/density per frame ======
    print("[INFO] Unpacking reconstructed canvases (in-memory)…")
    planes_recon_by_frame: List[Dict[str, torch.Tensor]] = [dict() for _ in range(T)]
    density_recon_by_frame: List[torch.Tensor] = [None]*T

    # Feature planes
    C_feat = next(iter(planes0.values())).shape[1]  # e.g. 12
    for pn, rec_canv in feat_canv_by_plane.items():
        H2, W2 = feat_orig_hw[pn]
        for ti in range(T):
            crop = rec_canv[ti:ti+1, :, :H2, :W2]  # [1,3,H2,W2]
            x01 = unpack_rgb_to_planes(crop, C=C_feat, orig_size=(H2, W2), mode=args.plane_packing_mode)  # [1,C,H,W]
            bounds = feat_bounds_per_frame[pn][ti]  # [(lo,hi)*C]
            feat = dequantise_from_01(x01, bounds)  # [1,C,H,W]
            planes_recon_by_frame[ti][pn] = feat.contiguous()

    # Density
    for ti in range(T):
        H2d, W2d = dens_orig_hw
        crop = rec_d[ti:ti+1, :, :H2d, :W2d]  # [1,3,H2d,W2d]
        ckpt_meta = torch.load(os.path.join(args.ckpt_dir, f"fine_last_{frame_ids[ti]}.tar"),
                               map_location='cpu', weights_only=False)
        dens_shape = ckpt_meta["model_state_dict"]["density.grid"].shape  # [1,1,Dy,Dx,Dz]
        Dy, Dx, Dz = dens_shape[2:]
        d01_5 = unpack_density_from_rgb(crop, Dy, Dx, Dz, (H2d, W2d), mode=args.grid_packing_mode)
        d_raw = dens_from01(d01_5).contiguous()
        density_recon_by_frame[ti] = d_raw

    # ====== Pass 4: render test views directly from in-memory recon (no ckpt write) ======
    print("[INFO] Rendering test views…")
    S, E = frame_ids[0], frame_ids[-1] if len(frame_ids)>0 else frame_ids[0]
    rgbnet_file = os.path.join(args.ckpt_dir, f"rgbnet_{S}_{E}.tar")
    if not os.path.isfile(rgbnet_file):
        rgbnet_file = os.path.join(os.path.dirname(args.ckpt_dir), f"rgbnet_{S}_{E}.tar")
    assert os.path.isfile(rgbnet_file), f"Cannot find rgbnet file: {rgbnet_file}"
    rgb_ckpt = torch.load(rgbnet_file, map_location='cpu', weights_only=False)
    mkw = rgb_ckpt['model_kwargs']
    if cfg.fine_model_and_render.RGB_model=='MLP':
        rgbnet = RGB_Net(dim0=mkw['dim0'], rgbnet_width=mkw['rgbnet_width'], rgbnet_depth=mkw['rgbnet_depth'])
    else:
        rgbnet = RGB_SH_Net(dim0=mkw['dim0'], rgbnet_width=mkw['rgbnet_width'],
                            rgbnet_depth=mkw['rgbnet_depth'], deg=2)
    rgbnet.load_state_dict(rgb_ckpt['model_state_dict'])
    rgbnet = rgbnet.to(device).eval()

    stepsize = cfg.fine_model_and_render.stepsize
    render_kwargs = dict(
        near=data_dict['near'], far=data_dict['far'], bg=1 if cfg.data.white_bkgd else 0,
        stepsize=stepsize, inverse_y=cfg.data.inverse_y, flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
        render_depth=True, shared_rgbnet=rgbnet,
    )

    overall_metrics = []
    agg_psnr, agg_ssim, agg_lpips = [], [], []
    for fid in render_frame_ids:
        ti = fid2idx[fid]
        print
        ckpt_path = os.path.join(args.ckpt_dir, f"fine_last_{fid}.tar")
        orig = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        st = orig['model_state_dict'].copy()
        for pn, feat in planes_recon_by_frame[ti].items():
            st[f'k0.{pn}'] = feat
        st['density.grid'] = density_recon_by_frame[ti]

        model_class = dmpigo.DirectMPIGO if cfg.data.ndc else dvgo.DirectVoxGO
        model = model_class(**orig['model_kwargs']).to(device)
        model.load_state_dict(st, strict=False)
        model.reset_occupancy_cache()  # now safe: default tensor type set to CUDA if available
        model.eval()

        testsavedir = render_out
        i_test = data_dict['i_test']
        frame_ids_all = data_dict['frame_ids']
        id_mask = (frame_ids_all==fid)[i_test].numpy()
        t_test = np.array(i_test)[id_mask]

        masks = None
        if data_dict['masks'] is not None:
            masks = [data_dict['masks'][i] for i in t_test]

        res = render_viewpoints(
            model=model,
            render_poses=data_dict['poses'][t_test],
            HW=data_dict['HW'][t_test],
            Ks=data_dict['Ks'][t_test],
            ndc=cfg.data.ndc,
            render_kwargs=render_kwargs,
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in t_test],
            savedir=testsavedir, dump_images=args.dump_images,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            frame_id=fid, masks=None
        )
        overall_metrics.append((fid, res))
        if isinstance(res, dict):
            if 'psnr' in res and np.isfinite(res['psnr']):    agg_psnr.append(float(res['psnr']))
            if 'ssim' in res and np.isfinite(res.get('ssim', np.nan)):   agg_ssim.append(float(res['ssim']))
            if 'lpips' in res and np.isfinite(res.get('lpips', np.nan)): agg_lpips.append(float(res['lpips']))

    # ====== Final: write encoded_bits.txt with per-stream + total ======
    bits_txt = os.path.join(out_root, "encoded_bits.txt")
    with open(bits_txt, "w") as f:
        for k in ['xy_plane','xz_plane','yz_plane']:
            if k in stream_stats:
                m = stream_stats[k]
                f.write(f"{k}: total_bits={int(m['total_bits'])}  bpp={m['bpp']:.8f}  T={int(m['T'])}  Hp={int(m['Hp'])}  Wp={int(m['Wp'])}\n")
        m = stream_stats['density']
        f.write(f"density: total_bits={int(m['total_bits'])}  bpp={m['bpp']:.8f}  T={int(m['T'])}  Hp={int(m['Hp'])}  Wp={int(m['Wp'])}\n")
        total_pixels_padded = sum(int(stream_stats[s]['T']*stream_stats[s]['Hp']*stream_stats[s]['Wp'])
                                  for s in stream_stats)
        overall_bpp = total_bits / float(total_pixels_padded) if total_pixels_padded>0 else 0.0
        f.write(f"TOTAL: total_bits={int(total_bits)}  bpp={overall_bpp:.8f}\n")
    print(f"[OK] wrote {bits_txt}")

    for fid, res in overall_metrics:
        if res:
            print(f"[render_test] frame {fid}: " +
                  ("PSNR=%.3f " % res.get('psnr', float('nan'))) +
                  ("SSIM=%.4f " % res.get('ssim', float('nan')) if 'ssim' in res else "") +
                  ("LPIPS=%.4f" % res.get('lpips', float('nan')) if 'lpips' in res else ""))
            
    def _maybe_write_mean(fname: str, vals: List[float]):
        if len(vals) > 0:
            with open(os.path.join(out_root, fname), "w") as f:
                f.write(f"{np.mean(vals):.6f}\n")

    _maybe_write_mean("mean_PSNR.txt",  agg_psnr)
    _maybe_write_mean("mean_SSIM.txt",  agg_ssim)
    _maybe_write_mean("mean_LPIPS.txt", agg_lpips)

    print("[DONE] All outputs in:", out_root)
