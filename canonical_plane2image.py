import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : ((2**16-1)*np.clip(x,0,1)).astype(np.uint16)

def tile_maker(feat_plane, h = 2560, w= 2560):
    image = torch.zeros(h,w)
    h,w = list(feat_plane.size())[-2:]
    x,y = 0,0
    for i in range(feat_plane.size(1)):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "Tile_maker: not enough space"
        image[x:x+h,y:y+w] = feat_plane[0,i,:,:]
        y = y + w
    return image

def density_quantize(density, nbits):
    levels = 2**nbits - 1
    data = density.clone()
    data.clamp_(-5, 30)
    data = (data + 5) / 35.0
    data = torch.round(data * levels) / levels
    return data

def density_dequantize(density):
    """
    density: [0, 1] -> [-5, 30]
    """
    return density * 35.0 - 5

def make_density_image(density_grid, nbits, h=2560, w=4096):
    data = density_quantize(density_grid, nbits)
    res = tile_maker(data[0], h=h,w=w)

    return res



if __name__=='__main__':
    """
    Usage:
        python canonical_plane2image.py --logdir logs/out_triplane/flame_steak_old --numframe 20 --strategy tiling
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True, help='Path to Tri-plane checkpoints (.tar files)')
    parser.add_argument("--numframe", type=int, default=20, help='number of frames')
    parser.add_argument("--codec", type=str, default='h265', help='h265 or mpg2')
    parser.add_argument("--startframe", type=int, default=0, help='start frame id')
    parser.add_argument('--strategy', type=str, default='tiling', choices=['tiling', 'separate', 'grouped'],
                        help='tiling: original; separate: one channel per stream; grouped: RGB triplets + leftover')
    args = parser.parse_args()

    low_bound, high_bound = -20, 20
    nbits = 2**16-1

    if args.logdir[-1] =='/':
        args.logdir = args.logdir[:-1]
    name = args.logdir.split('/')[-1]
    save_dir = os.path.join(args.logdir, f'planeimg_{args.startframe:02d}_{args.startframe+args.numframe-1:02d}_{args.strategy}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plane images to {save_dir}")

    for frameid in tqdm(range(args.startframe, args.startframe + args.numframe)):
        tmp_file = os.path.join(args.logdir, f'fine_last_{frameid}.tar')
        assert os.path.isfile(tmp_file), "Checkpoint not found."
        tqdm.write(f"Loading Checkpoint {tmp_file}")
        ckpt = torch.load(tmp_file, map_location='cpu', weights_only=False)

        # load density and compute mask 
        density = ckpt['model_state_dict']['density.grid'].clone()
        volume_size = list(density.size())[-3:]
        voxel_size_ratio = ckpt['model_kwargs']['voxel_size_ratio']  

        # Prepare density grid mask
        masks = None
        if 'act_shift' in ckpt['model_state_dict']:
            alpha = 1- (torch.exp(density+ckpt['model_state_dict']['act_shift'])+1)**(-voxel_size_ratio)
            alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)
            mask = alpha<1e-4
            density[mask] = -5

            feature_alpha = F.interpolate(alpha, size=tuple(np.array(volume_size)*3), mode='trilinear', align_corners=True)
            mask_fg = feature_alpha>=1e-4

            # mask projection
            masks = {}
            masks['xy'] = mask_fg.sum(axis=4)
            masks['xz'] = mask_fg.sum(axis=3)
            masks['yz'] = mask_fg.sum(axis=2)

        # load tri-plane features
        planes = {}
        for key in ckpt['model_state_dict'].keys():
            if 'k0' in key and 'plane' in key and 'residual' not in key:
                data = ckpt['model_state_dict'][key].clone()
                planes[key.split('.')[-1]]= data

        # process each plane
        plane_data = []
        ratios = []
        plane_sizes ={}

        for p, feat_tensor in planes.items():  # feat_tensor shape [1, C, H, W] ([1, 10, 543, 543])
            if masks is not None:
                # apply mask zero-out
                mask_p = masks[p].unsqueeze(1).repeat(1, feat_tensor.size(1), 1, 1)
                feat_tensor = feat_tensor.masked_fill(mask_p == 0, 0)

            # normalize & quantize
            feat_norm = (feat_tensor - low_bound) / (high_bound - low_bound)
            feat_q = torch.round(feat_norm * nbits) / nbits
            feat_q.clamp_(0, 1)
            C, H, W = feat_q.shape[1:]
            plane_sizes[p] = (1, C, H, W)  # store plane size for later use

            # prepare save dirs
            if args.strategy == 'tiling':
                out_dir = os.path.join(save_dir, p)
                os.makedirs(out_dir, exist_ok=True)
                # img = tile_maker(feat_q, h=H * int(np.ceil(C / (W / H))), w=W * int(min(C, W // H)))
                img = tile_maker(feat_q)
                cv2.imwrite(os.path.join(out_dir, f'im{frameid+1:05d}.png'), to16b(img.cpu().numpy().astype(np.float32)))

            elif args.strategy == 'separate':
                out_base = os.path.join(save_dir, p)
                for k in range(C):
                    ch_dir = os.path.join(out_base, f'c{k}')
                    os.makedirs(ch_dir, exist_ok=True)
                    img = feat_q[0, k]
                    cv2.imwrite(os.path.join(ch_dir, f'im{frameid+1:05d}.png'), to16b(img.cpu().numpy()))

            elif args.strategy == 'grouped':
                # group into triplets, last group may have <3 channels
                groups = [list(range(i, min(i+3, C))) for i in range(0, C, 3)]
                for gi, idxs in enumerate(groups):
                    stream_dir = os.path.join(save_dir, p, f'stream{gi}')
                    os.makedirs(stream_dir, exist_ok=True)
                    data = feat_q[0, idxs]  # [len(idxs), H, W]
                    arr = data.cpu().numpy()
                    if arr.shape[0] == 3:
                        # convert RGB->BGR for OpenCV by reversing the channel order
                        bgr = np.stack([arr[2], arr[1], arr[0]], axis=-1)
                        cv2.imwrite(os.path.join(stream_dir, f'im{frameid+1:05d}.png'),
                                    to16b(bgr))
                    else:
                        # grayscale for leftover
                        cv2.imwrite(os.path.join(stream_dir, f'im{frameid+1:05d}.png'),
                                    to16b(arr[0]))

        # density saving
        density_img = make_density_image(density, 16)
        dens_dir = os.path.join(save_dir, 'density')
        os.makedirs(dens_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dens_dir, f'im{frameid+1:05d}.png'), to16b(density_img.cpu().numpy()))

        # save meta info
        torch.save({'plane_size':plane_sizes, 
                    'bounds': (low_bound,high_bound),
                    'nbits':nbits}, 
                    os.path.join(save_dir, f'planes_frame_meta.nf'))
    




