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

# def to16b(x):
#     """
#     Convert a numpy array to 16-bit unsigned integer format.
#     """
#     x = np.clip(x, 0, 1).astype(np.float32)  # Ensure values are in [0, 1]
#     x = 65535 * x  # Scale to [0, 65535]
#     return (65535 * x).astype(np.uint16)  # Scale to [0, 65535] and convert to uint16

def tile_maker(feat_plane, h = 2560, w= 2560):
    image = torch.zeros(h,w)
    h,w = list(feat_plane.size())[-2:]
    # print(feat_plane.size())
    # print(f"h: {h}, w: {w}")
    x,y = 0,0
    for i in range(feat_plane.size(1)):
        # print(f"x: {x}, y: {y}")
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "Tile_maker: not enough space"
        image[x:x+h,y:y+w] = feat_plane[0,i,:,:]
        y = y + w

    return image

def density_quantize(density, nbits):
    """
    Quantize density to [0, 1] range with nbits precision.
    """
    nbits = 2**nbits-1
    data = density.clone()
    
    data[data<-5] = -5
    data[data>30] = 30

    data = data +5
    data = data /(30+5)
    
    data = torch.round(data *nbits)/nbits
    return data

def density_dequantize(density):
    """
    density: [0, 1] -> [-5, 30]
    """
    data = density *(30+5)
    data = data-5
    return data

def make_density_image(density_grid, nbits, act_shift=0, h=3840, w=4096):
    data = density_quantize(density_grid, nbits)
    res = tile_maker(data[0], h=h,w=w)

    return res



if __name__=='__main__':
    """
    Usage:
        ython triplane2img.py --logdir logs/out_triplane/flame_steak_old --numframe 20
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True, help='Path to Tri-plane checkpoints (.tar files)')
    parser.add_argument("--numframe", type=int, default=20, help='number of frames')
    parser.add_argument("--codec", type=str, default='h265', help='h265 or mpg2')
    parser.add_argument("--startframe", type=int, default=0, help='start frame id')

    args = parser.parse_args()

    thresh = 0 
    bound_thres = 20
    low_bound = -bound_thres
    high_bound = bound_thres
    nbits = 2**16-1
    start_frame_id = args.startframe
    numframe = args.numframe

    if args.logdir[-1] =='/':
        args.logdir = args.logdir[:-1]
    name = args.logdir.split('/')[-1]
    save_dir = os.path.join(args.logdir, f'planeimg_{start_frame_id:02d}_{numframe-1:02d}')
    os.makedirs(save_dir, exist_ok=True)

    for frameid in tqdm(range(0, args.numframe)):
        tmp_file = os.path.join(args.logdir, f'fine_last_{frameid}.tar')
        assert os.path.isfile(tmp_file), "Checkpoint not found."
        tqdm.write(f"Loading Checkpoint {tmp_file}")
        ckpt = torch.load(tmp_file, map_location='cpu', weights_only=False)
        density = ckpt['model_state_dict']['density.grid'].clone()
        volume_size = list(density.size())[-3:]
        # print(f"Density shape: {density.shape}")  # torch.Size([1, 1, 181, 181, 280])
        # print(f"Volume size: {volume_size}")  # [181, 181, 280]
        voxel_size_ratio = ckpt['model_kwargs']['voxel_size_ratio']  # Voxel size ratio: 0.9142857142857143
        # print(f"Voxel size ratio: {voxel_size_ratio}")  # 3.0
        # print(ckpt['model_state_dict'].keys())
        """
        keys of ckpt: (['xyz_min', 'xyz_max', 'viewfreq', 'density.grid', 'density.xyz_min', 'density.xyz_max', 
        'act_shift.grid', 'act_shift.xyz_min', 'act_shift.xyz_max', '
        k0.xy_plane', 'k0.xz_plane', 'k0.yz_plane', 'k0.xyz_min', 'k0.xyz_max', 
        'mask_cache.mask', 'mask_cache.xyz2ijk_scale', 'mask_cache.xyz2ijk_shift'])
        """

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

        # Prepare planes
        planes = {}
        for key in ckpt['model_state_dict'].keys():
            if 'k0' in key and 'plane' in key and 'residual' not in key:
                data = ckpt['model_state_dict'][key].clone()
                planes[key.split('.')[-1]]= data

        plane_data = []
        ratios = []
    
        for p in ['xy','xz','yz']:
            plane_size = list(planes[f"{p}_plane"].size())[-1:-3:-1]
            # print(planes[f"{p}_plane"].size()) # [1, 10, 543, 543]
            # print(f"Plane {p} size: {plane_size}")  # [543, 543]
            
            if masks is not None:
                cur_mask = masks[p].repeat(1,planes[f"{p}_plane"].size(1),1,1)
                planes[f"{p}_plane"][cur_mask<1] = 0

                #sanity check: 95% of the values should be within the bound
                ra = planes[f"{p}_plane"].abs()
                ra = ra[cur_mask>=1].abs()
                assert (ra<bound_thres).sum()/ra.size(0) >0.95, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
                ratios.append(((ra<bound_thres).sum()/ra.size(0)).item())
            else:
                ra = planes[f"{p}_plane"].abs()
                ra = ra.reshape(-1)
                assert (ra<bound_thres).sum()/ra.size(0) >0.95, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
                ratios.append(((ra<bound_thres).sum()/ra.size(0)).item())

            # Quantize the plane 
            feat = (planes[f"{p}_plane"] - low_bound)/(high_bound-low_bound)
            feat = torch.round(feat *nbits)/nbits
            feat[feat<0]=0
            feat[feat>1.0] = 1.0
            plane_data.append(feat)
            gt_feat = (planes[f"{p}_plane"]- low_bound)/(high_bound-low_bound) # gt_plane: without quantization

        xy_save_dir = os.path.join(save_dir, 'xy')
        xz_save_dir = os.path.join(save_dir, 'xz')
        yz_save_dir = os.path.join(save_dir, 'yz')
        density_save_dir = os.path.join(save_dir, 'density')
        os.makedirs(xy_save_dir, exist_ok=True)
        os.makedirs(xz_save_dir, exist_ok=True)
        os.makedirs(yz_save_dir, exist_ok=True)
        os.makedirs(density_save_dir, exist_ok=True)
  
        imgs = {}
        plane_sizes ={}
        for ind, plane in zip(['xy','xz','yz'],plane_data):
            img = tile_maker(plane).half()
            imgs[f'{ind}_plane'] = img
            plane_sizes[f'{ind}_plane'] = plane.size()
            # Ensure in float32 to avoid overflow when converting to 16-bit
            cv2.imwrite(os.path.join(save_dir, ind, f'im{frameid+1:05d}.png'), to16b(img.cpu().numpy().astype(np.float32)))

        
        density_image = make_density_image(density, 16)
        cv2.imwrite(os.path.join(density_save_dir, f'im{frameid+1:05d}.png'),to16b(density_image.cpu().numpy()))
        torch.save({'plane_size':plane_sizes, 
                    'bounds': (low_bound,high_bound),
                    'nbits':nbits}, 
                    os.path.join(save_dir, f'planes_frame_meta.nf'))
    




