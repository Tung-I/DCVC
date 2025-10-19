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
    python render.py --config  configs/dynerf_sear_steak/image_l.py --frame_ids 0 --render_test
        --frame_ids 0 1 2 3 4 5 6 7 8 9 --render_test \
        --ckpt_dir logs/dynerf_flame_steak/flame_steak_video_ds3/compressed_hevc_crf28_g10_yuv444p
    python render.py --config  configs/dynerf_flame_steak/image.py \
        --render_test --frame_ids 0 \
        --ckpt_dir logs/dynerf_flame_steak/flame_steak_image/planeimg_00_00_flatten_global_jpeg_qp20
    python render.py --config  configs/blender_lego/image.py  \
        --render_test --frame_ids 0 \
        --ckpt_dir logs/nerf_synthetic/lego_image
    python render.py --config  configs/blender_lego/image.py  \
        --render_test --frame_ids 0 \
        --ckpt_dir logs/nerf_synthetic/lego_image/planeimg_00_00_flatten_flatten_global_jpeg_qp20
    python render.py --config  configs/blender_lego/image.py  \
        --render_test --frame_ids 0 \
        --ckpt_dir logs/nerf_synthetic/dcvc_qp24_flatten_flatten/dcvc_qp24
    python render.py --render_test --frame_ids 0 5 10 15 19 \
        --config configs/dynerf_flame_steak/av1_qp20.py  \
        --ckpt_dir logs/dynerf_flame_steak/av1_qp20/compressed_av1_qp20_g20_yuv444p
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

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    curf = frame_id % len(render_poses) 

    for i, c2w in enumerate(tqdm(render_poses)):
        # if i != curf and ndc:
        #     continue
        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        with torch.no_grad():
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(150480, 0), rays_d.split(150480, 0), viewdirs.split(150480, 0))
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
                for k in render_result_chunks[0].keys()
            }

        
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:

            if masks is not None:
                p = -10. * np.log10(np.mean(np.square(rgb[masks[i][...,0]>0.5,:] - gt_imgs[i][masks[i][...,0]>0.5,:])))
            else:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    res_psnr = {}
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        res_psnr = {'psnr':np.mean(psnrs)}
        with open(os.path.join(savedir, f'{frame_id}_psnr.txt'),'w') as f:
            f.write('%f' % np.mean(psnrs))
        if eval_ssim: 
            print('Testing ssim', np.mean(ssims), '(avg)')
            res_psnr['ssim'] = np.mean(ssims)
            with open(os.path.join(savedir, f'{frame_id}_ssim.txt'),'w') as f:
                f.write('%f' % np.mean(ssims))
        if eval_lpips_vgg: 
            print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
            res_psnr['lpips'] = np.mean(lpips_vgg)
            with open(os.path.join(savedir, f'{frame_id}_lpips.txt'),'w') as f:
                f.write('%f' % np.mean(lpips_vgg))

        if eval_lpips_alex: 
            print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, f'{frame_id}_{i}.png')
            imageio.imwrite(filename, rgb8)

            rgb8 = utils.to8b(1 - depths[i] / np.max(depths[i]))
            filename = os.path.join(savedir, f'{frame_id}_{i}_depth.png')
            if rgb8.shape[-1]<3:
                rgb8 = np.repeat(rgb8, 3, axis=-1)
            imageio.imwrite(filename, rgb8)


    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps, res_psnr

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

        # render trainset and eval
        if args.render_train:
            # replace the last folder in testsavedir (e.g., render_test) with "render_train"
            save_dir = pathlib.Path(testsavedir)
            save_dir = save_dir.with_name("render_train")
            testsavedir = str(save_dir)

            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            i_train = data_dict['i_train']
            frame_ids = data_dict['frame_ids']
            id_mask = (frame_ids==frame_id)[i_train]
            t_train = np.array(i_train)[id_mask]
            rgbs, depths, bgmaps,res_psnr = render_viewpoints(
                    render_poses=data_dict['poses'][t_train],
                    HW=data_dict['HW'][t_train],
                    Ks=data_dict['Ks'][t_train],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in t_train],
                    savedir=testsavedir, dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id = frame_id,
                    **render_viewpoints_kwargs)


        if args.render_test:
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)
            i_test = data_dict['i_test']
            frame_ids = data_dict['frame_ids']
            id_mask = (frame_ids==frame_id)[i_test]
            t_test = np.array(i_test)[id_mask.numpy()]

            data_mask = None
            if data_dict['masks'] is not None:
                data_mask = [data_dict['masks'][i] for i in t_test]

            rgbs, depths, bgmaps,res_psnr = render_viewpoints(
                    render_poses=data_dict['poses'][t_test],
                    HW=data_dict['HW'][t_test],
                    Ks=data_dict['Ks'][t_test],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in t_test],
                    savedir=testsavedir, dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id = frame_id, masks = None,
                    **render_viewpoints_kwargs)

        # render video
        if args.render_video:
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video')
            os.makedirs(testsavedir, exist_ok=True)
            print('All results are dumped into', testsavedir)

            i_test = data_dict['i_test']
            frame_ids = data_dict['frame_ids']
            id_mask = (frame_ids==frame_id)[i_test].numpy()
            t_test = np.array(i_test)[id_mask]
            t_render_poses = data_dict['render_poses']

            rgbs, depths, bgmaps, res_psnr = render_viewpoints(
                    render_poses=t_render_poses,
                    HW=data_dict['HW'][t_test][[0]].repeat(len(t_render_poses), 0),
                    Ks=data_dict['Ks'][t_test][[0]].repeat(len(t_render_poses), 0),
                    render_factor=args.render_video_factor,
                    render_video_flipy=args.render_video_flipy,
                    render_video_rot90=args.render_video_rot90,
                    savedir=testsavedir, dump_images=args.dump_images, frame_id = frame_id,
                    **render_viewpoints_kwargs)


    print('Done')

