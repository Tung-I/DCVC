import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
# import mmcv
from mmengine.config import Config
import imageio
import numpy as np
import ipdb
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from TeTriRF.lib import utils, dvgo,  dmpigo
from TeTriRF.lib.load_data import load_data
from TeTriRF.lib.dvgo_video import RGB_Net, RGB_SH_Net

from torch_efficient_distloss import flatten_eff_distloss
import pandas as pd
import time

"""
Usage:
    python render_blender.py --config  configs/nerf_chair/image.py  \
        --render_test --frame_ids 0 \
        --ckpt_dir logs/nerf_chair/image
"""

def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument('--frame_ids', nargs='+', type=int, help='a list of ID')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--residual_train", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", type=bool, default=True)
    parser.add_argument("--eval_ssim", action='store_true', default=True)
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true',  default=True)

    parser.add_argument("--codec", type=str, default='h265', help='h265 or mpg2')

    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--ckpt_dir", type=str, default=None, help='path to ckpt')
    return parser

def seed_everything():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
                'i_train', 'i_val', 'i_test', 'irregular_shape',
                'poses', 'render_poses', 'images', 'frame_ids','masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
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
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,frame_id = 0, masks = None):
    '''Run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs, depths, bgmaps = [], [], []
    psnrs, ssims, lpips_alex, lpips_vgg = [], [], [], []
    model_device = 'cuda'

    for i, c2w_np in enumerate(tqdm(render_poses)):
        H, W = map(int, HW[i])
        K    = Ks[i]
        c2w  = torch.as_tensor(c2w_np, dtype=torch.float32)

        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H, W, K, c2w,
            ndc,
            inverse_y=render_kwargs['inverse_y'],
            flip_x=render_kwargs['flip_x'],
            flip_y=render_kwargs['flip_y'],
        )
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        # chunked render
        chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(150480, 0),
                                  rays_d.split(150480, 0),
                                  viewdirs.split(150480, 0))
        ]
        render_out = {
            k: torch.cat([c[k] for c in chunks]).reshape(H, W, -1)
            for k in keys
        }

        
        rgb   = render_out['rgb_marched'].cpu().numpy()
        depth = render_out['depth'].cpu().numpy()
        bgmap = render_out['alphainv_last'].cpu().numpy()

        if render_video_flipy:
            rgb   = np.flip(rgb,   axis=0)
            depth = np.flip(depth, axis=0)
            bgmap = np.flip(bgmap, axis=0)
        if render_video_rot90 != 0:
            rgb   = np.rot90(rgb,   k=render_video_rot90, axes=(0, 1))
            depth = np.rot90(depth, k=render_video_rot90, axes=(0, 1))
            bgmap = np.rot90(bgmap, k=render_video_rot90, axes=(0, 1))

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

        if i==0:
            print('Testing', rgb.shape)

        # metrics
        if gt_imgs is not None and render_factor == 0:
            if masks is not None and masks[i] is not None:
                m = masks[i][..., 0] > 0.5
                mse = np.mean((rgb[m] - gt_imgs[i][m]) ** 2)
            else:
                mse = np.mean((rgb - gt_imgs[i]) ** 2)
            psnrs.append(-10.0 * np.log10(max(mse, 1e-12)))

            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=model_device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=model_device))

    # save PNGs
    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            imageio.imwrite(os.path.join(savedir, f'{frame_id}_{i}.png'), rgb8)

            # depth_vis = utils.to8b(1 - depths[i] / (np.max(depths[i]) + 1e-12))
            # if depth_vis.ndim == 2:
            #     depth_vis = np.repeat(depth_vis[..., None], 3, axis=-1)
            # imageio.imwrite(os.path.join(savedir, f'{frame_id}_{i}_depth.png'), depth_vis)
    res_psnr = {}

    # pack results
    res_psnr = {}
    if psnrs:
        res_psnr['psnr'] = float(np.mean(psnrs))
        print('Testing psnr', res_psnr['psnr'], '(avg)')
        if eval_ssim:
            res_psnr['ssim'] = float(np.mean(ssims))
            print('Testing ssim', res_psnr['ssim'], '(avg)')
        if eval_lpips_vgg:
            res_psnr['lpips'] = float(np.mean(lpips_vgg))
            print('Testing lpips (vgg)', res_psnr['lpips'], '(avg)')
        if eval_lpips_alex:
            print('Testing lpips (alex)', float(np.mean(lpips_alex)), '(avg)')


    return np.array(rgbs), np.array(depths), np.array(bgmaps), res_psnr

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.frame_ids = args.frame_ids
    
    print("################################")
    print("--- Frame_ID:", args.frame_ids)
    print("################################")

    if not hasattr(cfg.fine_model_and_render, 'dynamic_rgbnet'):
        cfg.fine_model_and_render.dynamic_rgbnet = True
    if not hasattr(cfg.fine_model_and_render, 'RGB_model'):
        cfg.fine_model_and_render.RGB_model = 'MLP'

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()


    data_dict = load_everything(args=args, cfg=cfg)

    def _get_subset_indices(base_idx: np.ndarray, frame_ids_tensor: torch.Tensor,
                            frame_id: int, stride: int | None, maxn: int | None) -> np.ndarray:
        """Filter indices that match a given frame_id, then subsample by stride/maxn."""
        mask = (frame_ids_tensor == frame_id)[base_idx]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        picked = np.array(base_idx)[mask]
        if stride is not None and stride > 1:
            picked = picked[::stride]
        if maxn is not None and len(picked) > maxn:
            picked = picked[:maxn]
        return picked
    
    # Resolve subsampling policy
    subset_stride = int(getattr(cfg.fine_train, "test_subsample_stride", 10))
    subset_max    = int(getattr(cfg.fine_train, "test_subsample_max", 20))



    S, E = args.frame_ids[0], args.frame_ids[-1]
    if E is None:
        E = S
    for frame_id in args.frame_ids:
        ckpt_path = os.path.join(args.ckpt_dir, f"fine_last_{frame_id}.tar")
        testsavedir = os.path.join(args.ckpt_dir, f'render_test')
        rgbnet_file = os.path.join(args.ckpt_dir, f'rgbnet_{S}_{E}.tar')
        # if rgbnet_file does not exist, find whether os.path.join(ckpt_path.parent, "rgbnet.tar") exists
        if not os.path.exists(rgbnet_file):
            rgbnet_file = os.path.join(pathlib.Path(args.ckpt_dir).parent, f'rgbnet_{S}_{E}.tar')
            # assert and also print a warning if assertion fails
            assert os.path.exists(rgbnet_file), f"Cannot find rgbnet file: {rgbnet_file}"  
        print('Loading from', ckpt_path)
        print('Loading RGBNet from', rgbnet_file)
        print('Saving to', testsavedir)
        # raise Exception

        # Load model
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        else:
            model_class = dvgo.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path, weights_only=False).to(device)
        model.reset_occupancy_cache()

        checkpoint =torch.load(rgbnet_file, weights_only=False)
        model_kwargs = checkpoint['model_kwargs']
        if cfg.fine_model_and_render.RGB_model=='MLP':
            dim0 = model_kwargs['dim0']
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            rgbnet = RGB_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth)
        elif cfg.fine_model_and_render.RGB_model =='SH':
            dim0 = model_kwargs['dim0']
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            rgbnet = RGB_SH_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth, deg=2)

        rgbnet.load_state_dict(checkpoint['model_state_dict'])
        print('load rgbnet:', rgbnet_file )

        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'shared_rgbnet': rgbnet,
            },
        }

        # ---------- render train ----------
        if args.render_train:
            save_dir = pathlib.Path(args.ckpt_dir) / "render_train"
            os.makedirs(save_dir, exist_ok=True)
            print('All results are dumped into', str(save_dir))

            i_train = data_dict['i_train']
            t_train = _get_subset_indices(i_train, data_dict['frame_ids'], frame_id,
                                          stride=1, maxn=10)  # usually render all train
            rgbs, depths, bgmaps, res_psnr = render_viewpoints(
                render_poses=data_dict['poses'][t_train],
                HW=data_dict['HW'][t_train],
                Ks=data_dict['Ks'][t_train],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in t_train],
                savedir=str(save_dir), dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                frame_id=frame_id,
                **render_viewpoints_kwargs
            )


        # ---------- render test ----------
        if args.render_test:
            save_dir = pathlib.Path(args.ckpt_dir) / "render_test"
            os.makedirs(save_dir, exist_ok=True)
            print('All results are dumped into', str(save_dir))

            i_test = data_dict['i_test']
            t_test = _get_subset_indices(i_test, data_dict['frame_ids'], frame_id,
                                         stride=subset_stride, maxn=subset_max)

            data_mask = None
            if data_dict.get('masks', None) is not None:
                data_mask = [data_dict['masks'][i] for i in t_test]

            rgbs, depths, bgmaps, res_psnr = render_viewpoints(
                render_poses=data_dict['poses'][t_test],
                HW=data_dict['HW'][t_test],
                Ks=data_dict['Ks'][t_test],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in t_test],
                savedir=str(save_dir), dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                frame_id=frame_id, masks=data_mask,
                **render_viewpoints_kwargs
            )


    print('Done')

