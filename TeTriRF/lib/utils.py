import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_adam import MaskedAdam


''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group)

#参考create_optimizer_or_freeze_model写一个创建optimizer的函数,用于DirectVoxGO_Video的训练优化,使得其rgbnet和dvgos的参数可以被训练, learning rate 由配置参数决定
def create_optimizer_or_freeze_model_dvgovideo(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if k=='decay':
            continue

        if k=='rgbnet':
            param = getattr(model, k)
            if param is None:
                print(f'create_optimizer_or_freeze_model: param {k} not exist')
                continue

            lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
            if lr > 0:
                print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
                if isinstance(param, nn.Module):
                    param = param.parameters()
                    param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})

            else:
                print(f'create_optimizer_or_freeze_model: param {k} freeze')
                param.requires_grad = False
        else:
            for frameid in model.dvgos.keys():

                if not hasattr(model.dvgos[frameid], k):
                    print(f'create_optimizer_or_freeze_model: param {k} not exist')
                    continue

                param = getattr(model.dvgos[frameid], k)

                if int(frameid) in model.fixed_frame:
                    print(f'create_optimizer_or_freeze_model: param {k} freeze [previous frames]')
                    param.requires_grad = False
                    continue


                lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
                if lr > 0:
                    print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
                    if isinstance(param, nn.Module):
                        param = param.parameters()
                        param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})

                else:
                    print(f'create_optimizer_or_freeze_model: param {k} freeze')
                    param.requires_grad = False
            
    return MaskedAdam(param_group)

''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def load_model(model_class, ckpt_path,residual=False, weights_only=None):
    if weights_only is not None:
        ckpt = torch.load(ckpt_path, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path)
    
    ckpt['model_kwargs']['residual_mode'] = residual
    if 'RGB_model' in ckpt['model_kwargs']:
        ckpt['model_kwargs']['rgb_model'] = ckpt['model_kwargs']['RGB_model']
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model

def load_residual_model(model_class, ckpt_path,residual = False):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])

    model.density.commit_residual()
    model.k0.commit_residual()

    return model


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


class Ray_Dataset(torch.utils.data.Dataset):
    def __init__(self, rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler):
        super().__init__()
        self.rgb_tr = rgb_tr
        self.rays_o_tr = rays_o_tr
        self.rays_d_tr = rays_d_tr
        self.viewdirs_tr = viewdirs_tr
        self.imsz = imsz
        self.frame_id_tr = frame_id_tr
        self.batch_index_sampler = batch_index_sampler
    
    def __len__(self):
        return 9999999

    def __getitem__(self, idx):
        camera_id, sel_i = self.batch_index_sampler()
        sel_i = torch.from_numpy(sel_i)
        while sel_i.size(0) == 0:
            print('while loop in Ray_dataset')
            camera_id, sel_i = self.batch_index_sampler()
        

        frameids = self.frame_id_tr[camera_id][sel_i]
    
        target = self.rgb_tr[camera_id][sel_i]   
        rays_o = self.rays_o_tr[camera_id][sel_i]
        rays_d = self.rays_d_tr[camera_id][sel_i]
        viewdirs = self.viewdirs_tr[camera_id][sel_i]
        return frameids, target, target, rays_o, rays_d, viewdirs