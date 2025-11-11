#!/usr/bin/env python3
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
    tile_1xCHW, untile_to_1xCHW,
)

from src.models.codec_wrapper import DCVCVideoCodecWrapper

"""
Example:

python eval_ours_4in1_dcvc.py \
  --config configs/dynerf_flame_steak/dcvc_qp24_flat4_absmax.py \
  --ckpt_dir logs/dynerf_flame_steak/dcvc_qp24_flat4_absmax \
  --qp 24 \
  --qmode absmax \
  --plane_packing_mode flat4 \
  --grid_packing_mode flatten

python eval_ours_4in1_dcvc.py \
  --config configs/nhr_sport1/dcvc_qp60_notv.py  \
  --ckpt_dir logs/NHR/sport1_dcvc_qp60_notv \
  --qp 60 \
  --qmode absmax \
  --plane_packing_mode flat4 \
  --grid_packing_mode flatten
"""

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

    # Packing & quant (used only for sanity checks vs cfg.codec)
    p.add_argument('--plane_packing_mode', choices=['flatten', 'mosaic', 'flat4'], default='flatten')
    p.add_argument('--grid_packing_mode',  choices=['flatten', 'mosaic', 'flat4'], default='flatten')
    p.add_argument('--qmode', choices=['absmax', 'affine'], default='absmax',
                   help="Quantization for appearance planes (for sanity check vs cfg.codec)")
    p.add_argument('--align', type=int, default=32)

    # DCVC codec knobs (for sanity checks vs cfg.codec)
    p.add_argument('--qp', type=int, required=True, help='DCVC QP (should match cfg.codec.dcvc_qp)')
    p.add_argument('--reset_interval', type=int, default=20)
    p.add_argument('--intra_period', type=int, default=-1)
    p.add_argument('--use_amp', action='store_true', default=False,
                   help='Use AMP (float16) for DCVC nets (should match cfg.codec.use_amp)')

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
# Shared helpers
# ---------------------------------------------------------------------------------
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
                if d.ndim == 3 and d.shape[-1] == 1:
                    d = d[..., 0]
                d_norm = 1.0 - d / (np.max(d) + 1e-8)
                d8 = (np.clip(d_norm, 0, 1) * 255.0 + 0.5).astype(np.uint8)
                if d8.ndim == 2:
                    d8 = np.repeat(d8[..., None], 3, axis=-1)
                elif d8.ndim == 3 and d8.shape[-1] == 1:
                    d8 = np.repeat(d8, 3, axis=-1)
                imageio.imwrite(os.path.join(savedir, f'{frame_id}_{i}_depth.png'), d8)

    return res

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
    out_root = os.path.abspath(os.path.join(args.ckpt_dir, '4in1test_dcvc'))
    os.makedirs(out_root, exist_ok=True)
    render_out = os.path.join(out_root, "render_test")
    os.makedirs(render_out, exist_ok=True)

    # ------- Load cfg & dataset (unchanged behavior) -------
    cfg = Config.fromfile(args.config)
    if not hasattr(cfg.fine_model_and_render, 'dynamic_rgbnet'):
        cfg.fine_model_and_render.dynamic_rgbnet = True
    if not hasattr(cfg.fine_model_and_render, 'RGB_model'):
        cfg.fine_model_and_render.RGB_model = 'MLP'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    data_dict = load_everything_for_render(args, cfg)

    # ====== Instantiate DCVC wrapper (EXACTLY as in training) ======
    assert hasattr(cfg, 'codec'), "cfg.codec not found; DCVC config is required."
    # Optional: sanity checks that CLI and cfg.codec agree (no overrides!)
    # QP
    cfg_qp = int(getattr(cfg.codec, "dcvc_qp", args.qp))
    if cfg_qp != args.qp:
        # print(f"[WARN] CLI qp={args.qp} but cfg.codec.dcvc_qp={cfg_qp}; using cfg.codec.dcvc_qp in wrapper.")
        cfg.codec.dcvc_qp = args.qp
    dcvc_wrapper = DCVCVideoCodecWrapper(cfg.codec, device=device)

    
    # quant mode
    cfg_qmode_raw = str(getattr(cfg.codec, "quant_mode", "global")).lower()
    cfg_qmode_norm = "absmax" if cfg_qmode_raw in ["global", "absmax"] else cfg_qmode_raw
    if cfg_qmode_norm != args.qmode:
        print(f"[WARN] CLI qmode={args.qmode} but cfg.codec.quant_mode={cfg_qmode_raw} (interpreted as {cfg_qmode_norm}).")
    # packing mode / align
    cfg_packing_mode = str(getattr(cfg.codec, "packing_mode", "flatten"))
    if cfg_packing_mode != args.plane_packing_mode:
        print(f"[WARN] CLI plane_packing_mode={args.plane_packing_mode} but cfg.codec.packing_mode={cfg_packing_mode}.")
        cfg.codec.packing_mode = args.plane_packing_mode

    cfg_align = int(getattr(cfg.codec, "align", DCVC_ALIGN))
    if cfg_align != args.align:
        print(f"[WARN] CLI align={args.align} but cfg.codec.align={cfg_align} (or DCVC_ALIGN={DCVC_ALIGN}).")

    # ====== Gather raw planes & density sequences from ckpts ======
    print("[INFO] Gathering raw planes & density from checkpoints…")

    # inspect first frame to get plane names (e.g. xy_plane, xz_plane, yz_plane)
    ckpt0 = torch.load(os.path.join(args.ckpt_dir, f"fine_last_{frame_ids[0]}.tar"),
                       map_location="cpu", weights_only=False)
    planes0 = {
        k.split(".")[-1]: v.clone()
        for k, v in ckpt0["model_state_dict"].items()
        if "k0" in k and "plane" in k and "residual" not in k
    }
    plane_names = sorted(planes0.keys())  # typically ['xy_plane', 'xz_plane', 'yz_plane']

    planes_by_name: Dict[str, List[torch.Tensor]] = {pn: [] for pn in plane_names}
    dens_list: List[torch.Tensor] = []

    for fid in tqdm(frame_ids, desc="[gather ckpts]"):
        ckpt = torch.load(os.path.join(args.ckpt_dir, f"fine_last_{fid}.tar"),
                          map_location="cpu", weights_only=False)
        planes = {
            k.split(".")[-1]: v.clone()
            for k, v in ckpt["model_state_dict"].items()
            if "k0" in k and "plane" in k and "residual" not in k
        }
        for pn in plane_names:
            planes_by_name[pn].append(planes[pn])  # each is [1,C,H,W]
        dens_list.append(ckpt["model_state_dict"]["density.grid"].clone())  # [1,1,Dy,Dx,Dz]

    # stack in time-major order
    T = len(frame_ids)
    for pn in plane_names:
        planes_by_name[pn] = torch.cat(planes_by_name[pn], dim=0)  # [T,C,H,W]
    d_seq = torch.cat(dens_list, dim=0)  # [T,1,Dy,Dx,Dz]

    # ====== DCVC roundtrip via DCVCVideoCodecWrapper (EXACTLY training pipeline) ======
    print("[INFO] DCVC video codec roundtrip via DCVCVideoCodecWrapper…")
    planes_recon_by_frame: List[Dict[str, torch.Tensor]] = [dict() for _ in range(T)]
    density_recon_by_frame: List[torch.Tensor] = [None] * T

    stream_stats: Dict[str, Dict[str, float]] = {}
    total_bits = 0.0

    # helper to recover Hp,Wp from packing logic (only for bits, not reconstruction)
    def _get_plane_canvas_shape(example_plane: torch.Tensor) -> Tuple[int, int]:
        """
        example_plane: [1,C,H,W] on any device; values don't matter.
        Returns (Hp, Wp) using the same pack_planes_to_rgb + align/padding
        as DCVCVideoCodecWrapper.
        """
        align = int(getattr(cfg.codec, "align", DCVC_ALIGN))
        packing_mode = str(getattr(cfg.codec, "packing_mode", "flatten"))
        dummy01 = torch.zeros_like(example_plane, dtype=torch.float32)
        canv, _ = pack_planes_to_rgb(dummy01, align=align, mode=packing_mode)  # [1,3,Hp,Wp]
        Hp, Wp = canv.shape[-2:]
        return int(Hp), int(Wp)

    def _get_density_canvas_shape(example_dens: torch.Tensor) -> Tuple[int, int]:
        """
        example_dens: [1,1,Dy,Dx,Dz] on any device; values don't matter.
        Mirrors DCVCVideoCodecWrapper.forward_density tiling + padding
        to recover Hp_d, Wp_d.
        """
        align = int(getattr(cfg.codec, "align", DCVC_ALIGN))
        # (i) map to [0,1]
        d01 = dens_to01(example_dens.to(dtype=torch.float32))  # [1,1,Dy,Dx,Dz]
        _, _, Dy, Dx, Dz = d01.shape
        # (ii) treat as [1,C,H,W] with C=Dy, H=Dx, W=Dz
        chw = d01.view(1, Dy, Dx, Dz)              # [1,C,H,W]
        mono, (Hc, Wc) = tile_1xCHW(chw)           # [Hc,Wc]
        pad_h = (align - Hc % align) % align
        pad_w = (align - Wc % align) % align
        Hp = Hc + pad_h
        Wp = Wc + pad_w
        return int(Hp), int(Wp)

    # --- feature planes: one DCVC call per plane stream (xy/xz/yz) ---
    for pn in plane_names:
        xseg = planes_by_name[pn].to(device=device, dtype=torch.float32)  # [T,C,H,W]
        rec, bpp, psnr = dcvc_wrapper(xseg)  # wrapper does quant+pack+DCVC+unpack+dequant

        rec = rec.to('cpu', dtype=torch.float32)
        bpp_val = float(bpp.detach().cpu().item())
        psnr_val = float(psnr.detach().cpu().item())

        # distribute per-frame reconstructions
        for t, fid in enumerate(frame_ids):
            planes_recon_by_frame[t][pn] = rec[t:t+1].contiguous()  # [1,C,H,W]

        # approximate padded canvas size for bits using the same pack logic
        Hp, Wp = _get_plane_canvas_shape(planes_by_name[pn][0:1])
        bits = bpp_val * float(T * Hp * Wp)

        stream_stats[pn] = dict(
            T=T, Hp=Hp, Wp=Wp,
            bpp=bpp_val,
            total_bits=bits,
            psnr=psnr_val,
        )
        total_bits += bits

    # --- density stream: one DCVC call via forward_density ---
    d_seq_dev = d_seq.to(device=device, dtype=torch.float32)  # [T,1,Dy,Dx,Dz]
    d_rec, d_bpp, d_psnr = dcvc_wrapper.forward_density(d_seq_dev)
    d_rec = d_rec.to('cpu', dtype=torch.float32)
    d_bpp_val = float(d_bpp.detach().cpu().item())
    d_psnr_val = float(d_psnr.detach().cpu().item())

    for t, fid in enumerate(frame_ids):
        density_recon_by_frame[t] = d_rec[t:t+1].contiguous()  # [1,1,Dy,Dx,Dz]

    # density canvas shape via tiling logic
    Hp_d, Wp_d = _get_density_canvas_shape(d_seq[0:1])
    bits_d = d_bpp_val * float(T * Hp_d * Wp_d)
    stream_stats['density'] = dict(
        T=T, Hp=Hp_d, Wp=Wp_d,
        bpp=d_bpp_val,
        total_bits=bits_d,
        psnr=d_psnr_val,
    )
    total_bits += bits_d

    # ====== Pass 4: render test views directly from in-memory recon (no ckpt write) ======
    print("[INFO] Rendering test views from DCVC-reconstructed features…")
    # match original VP9/HEVC/AV1 script: rgbnet over [S,E] of this segment
    # if len(frame_ids) > 0:
    #     S, E = frame_ids[0], frame_ids[-1]
    # else:
    #     S = E = 0
    S, E = 0, 9
    rgbnet_file = os.path.join(args.ckpt_dir, f"rgbnet_{S}_{E}.tar")
    if not os.path.isfile(rgbnet_file):
        # fallback: parent directory
        rgbnet_file = os.path.join(os.path.dirname(args.ckpt_dir), f"rgbnet_{S}_{E}.tar")
    assert os.path.isfile(rgbnet_file), f"Cannot find rgbnet file: {rgbnet_file}"
    rgb_ckpt = torch.load(rgbnet_file, map_location='cpu', weights_only=False)
    mkw = rgb_ckpt['model_kwargs']
    if cfg.fine_model_and_render.RGB_model == 'MLP':
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
        ckpt_path = os.path.join(args.ckpt_dir, f"fine_last_{fid}.tar")
        orig = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        st = orig['model_state_dict'].copy()

        # overwrite planes & density with DCVC reconstructions
        for pn, feat in planes_recon_by_frame[ti].items():
            st[f'k0.{pn}'] = feat  # pn is e.g. 'xy_plane'
        st['density.grid'] = density_recon_by_frame[ti]

        model_class = dmpigo.DirectMPIGO if cfg.data.ndc else dvgo.DirectVoxGO
        model = model_class(**orig['model_kwargs']).to(device)
        model.load_state_dict(st, strict=False)
        model.reset_occupancy_cache()
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
                f.write(
                    f"{k}: total_bits={int(round(m['total_bits']))}  "
                    f"bpp={m['bpp']:.8f}  T={int(m['T'])}  Hp={int(m['Hp'])}  Wp={int(m['Wp'])}\n"
                )
        m = stream_stats['density']
        f.write(
            f"density: total_bits={int(round(m['total_bits']))}  "
            f"bpp={m['bpp']:.8f}  T={int(m['T'])}  Hp={int(m['Hp'])}  Wp={int(m['Wp'])}\n"
        )
        total_pixels_padded = sum(int(stream_stats[s]['T']*stream_stats[s]['Hp']*stream_stats[s]['Wp'])
                                  for s in stream_stats)
        overall_bpp = total_bits / float(total_pixels_padded) if total_pixels_padded>0 else 0.0
        f.write(f"TOTAL: total_bits={int(round(total_bits))}  bpp={overall_bpp:.8f}\n")
    print(f"[OK] wrote {bits_txt}")

    for fid, res in overall_metrics:
        if res:
            print(
                f"[render_test] frame {fid}: " +
                ("PSNR=%.3f " % res.get('psnr', float('nan'))) +
                ("SSIM=%.4f " % res.get('ssim', float('nan')) if 'ssim' in res else "") +
                ("LPIPS=%.4f" % res.get('lpips', float('nan')) if 'lpips' in res else "")
            )

    def _maybe_write_mean(fname: str, vals: List[float]):
        if len(vals) > 0:
            with open(os.path.join(out_root, fname), "w") as f:
                f.write(f"{np.mean(vals):.6f}\n")

    _maybe_write_mean("mean_PSNR.txt",  agg_psnr)
    _maybe_write_mean("mean_SSIM.txt",  agg_ssim)
    _maybe_write_mean("mean_LPIPS.txt", agg_lpips)

    print("[DONE] All outputs in:", out_root)
