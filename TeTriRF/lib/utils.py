import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .masked_adam import MaskedAdam


''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def debug_print_param_status(model: torch.nn.Module, optimizer: torch.optim.Optimizer, *,
                             max_names_per_group: int = 30, prefix_levels: int = 4) -> None:
    """
    Prints which parameters are trainable, frozen, or orphaned (requires_grad=True but not in any opt group).
    Also shows each optimizer group's LR and a compact list of param names in it.
    """
    # 1) Build name ↔ param maps
    name_to_param = dict(model.named_parameters())
    id_to_name = {id(p): n for n, p in name_to_param.items()}

    # 2) Collect optimizer membership
    group_infos = []
    opt_param_ids = set()
    for gi, g in enumerate(optimizer.param_groups):
        plist = list(g['params'])
        opt_param_ids.update(id(p) for p in plist)
        group_infos.append({
            "idx": gi,
            "lr": g.get("lr", None),
            "skip_zero_grad": g.get("skip_zero_grad", False),
            "names": [id_to_name.get(id(p), f"<unnamed:{id(p)}>") for p in plist],
        })

    # 3) Partition params
    trainable = []
    frozen = []
    orphan = []  # requires_grad=True but not in any optimizer group
    for name, p in name_to_param.items():
        if p.requires_grad:
            if id(p) in opt_param_ids:
                trainable.append(name)
            else:
                orphan.append(name)
        else:
            frozen.append(name)

    # 4) Pretty helpers
    def shorten(n: str) -> str:
        # keep only first few dotted components for readability
        parts = n.split(".")
        if len(parts) <= prefix_levels: return n
        return ".".join(parts[:prefix_levels]) + ".../" + parts[-1]

    def sample(lst, k):
        return lst if len(lst) <= k else lst[:k] + [f"... (+{len(lst)-k} more)"]

    # 5) Print summary
    total = len(name_to_param)
    print("\n========== Parameter Freeze/Train Summary ==========")
    print(f"Total params: {total}")
    print(f"  - trainable in optimizer: {len(trainable)}")
    print(f"  - frozen (requires_grad=False): {len(frozen)}")
    print(f"  - ORPHAN (requires_grad=True but NOT in optimizer): {len(orphan)}")
    if orphan:
        print(">>> WARNING: The following params require grad but are not in any optimizer group:")
        for n in sample(orphan, 50):
            print("    •", n)

    for n in sample(trainable, 30):
        print("    •", n)

    # # 6) Show some frozen names (useful sanity check)
    # if frozen:
    #     print("\nFrozen params (sample):")
    #     for n in sample([shorten(n) for n in frozen], 30):
    #         print("    -", n)

    # # 7) Per-group breakdown
    # print("\nOptimizer groups:")
    # for info in group_infos:
    #     names = info["names"]
    #     print(f"  [group {info['idx']}] lr={info['lr']}  skip_zero_grad={info['skip_zero_grad']}  count={len(names)}")
    #     for n in sample([shorten(n) for n in names], max_names_per_group):
    #         print("     •", n)
    print("====================================================\n")

def create_optimizer_sandwich(model, cfg_train, global_step,
                                                   codec_params=[], sandwich_params=[]):
    """
    Build optimizer groups for TriPlane (density, k0) and rgbnet, and (optionally)
    sandwich modules. DCVC core is frozen, sandwich stays trainable.
    """

    def _as_list(x):
        if x is None: return []
        if isinstance(x, torch.nn.Parameter): return [x]
        if isinstance(x, torch.nn.Module):    return list(x.parameters())
        try: return list(x)
        except TypeError: return []

    # learning-rate schedule
    decay_steps  = getattr(cfg_train, 'lrate_decay', 20) * 1000
    decay_factor = (0.1 ** (global_step / decay_steps)) if decay_steps > 0 else 1.0
    def lr_of(name, default=0.0):
        base = getattr(cfg_train, f'lrate_{name}', default)
        return float(base) * decay_factor

    skip_zero = set(getattr(cfg_train, 'skip_zero_grad_fields', []))
    param_groups = []

    # --- 0) DCVC: freeze ONLY the core; leave sandwich alone ---
    if hasattr(model, 'codec'):
        codec = model.codec
        frozen = 0
        if hasattr(codec, "core_parameters"):
            for p in codec.core_parameters():
                p.requires_grad_(False); frozen += 1
        else:
            # Fallback: freeze everything except known sandwich prefixes
            S_PREFIX = ("pre_unet", "pre_mlp", "bound_pre",
                        "post_unet", "post_mlp", "bound_post")
            for n, p in codec.named_parameters():
                if n.startswith(S_PREFIX):
                    continue
                p.requires_grad_(False); frozen += 1
        # (optional) print or log `frozen` if you want

    # --- 1) RGBNet group ---
    lr_rgb = lr_of('rgbnet', 0.0)
    if hasattr(model, 'rgbnet') and isinstance(model.rgbnet, torch.nn.Module):
        rgb_params = _as_list(model.rgbnet)
        if lr_rgb > 0 and len(rgb_params):
            param_groups.append({
                'params': rgb_params, 'lr': lr_rgb,
                'skip_zero_grad': ('rgbnet' in skip_zero)
            })
            for p in rgb_params: p.requires_grad_(True)
        else:
            for p in rgb_params: p.requires_grad_(False)

    # --- 2) Per-frame TriPlane params (density, k0) ---
    lr_den = lr_of('density', 0.0)
    lr_k0  = lr_of('k0', 0.0)

    for fid, dvgo in model.dvgos.items():
        is_fixed = (int(fid) in getattr(model, 'fixed_frame', []))
        # density
        if hasattr(dvgo, 'density'):
            p_den = _as_list(dvgo.density)
            if not is_fixed and lr_den > 0 and len(p_den):
                param_groups.append({
                    'params': p_den, 'lr': lr_den,
                    'skip_zero_grad': ('density' in skip_zero)
                })
                for p in p_den: p.requires_grad_(True)
            else:
                for p in p_den: p.requires_grad_(False)
        # k0
        if hasattr(dvgo, 'k0'):
            p_k0 = _as_list(dvgo.k0)
            if not is_fixed and lr_k0 > 0 and len(p_k0):
                param_groups.append({
                    'params': p_k0, 'lr': lr_k0,
                    'skip_zero_grad': ('k0' in skip_zero)
                })
                for p in p_k0: p.requires_grad_(True)
            else:
                for p in p_k0: p.requires_grad_(False)

    # --- 3) Sandwich group (always considered; on/off via LR) ---
    lr_sw = lr_of('sandwich', 0.0)
    sw_list = _as_list(sandwich_params)
    if lr_sw > 0 and len(sw_list):
        param_groups.append({
            'params': sw_list, 'lr': lr_sw,
            'skip_zero_grad': ('sandwich' in skip_zero)
        })
        for p in sw_list: p.requires_grad_(True)
    else:
        # If LR==0, keep them frozen
        for p in sw_list: p.requires_grad_(False)

    # Safety: dtype check
    for gi, g in enumerate(param_groups):
        plist = g['params']
        if not isinstance(plist, (list, tuple)):
            plist = list(plist)
            g['params'] = plist
        for p in plist:
            if p.dtype != torch.float32:
                raise RuntimeError(
                    f"Optimizer group {gi} contains {p.dtype} param {tuple(p.shape)}; must be float32."
                )

    return MaskedAdam(param_groups)


def create_optimizer_or_freeze_model_dcvc_triplane(model, cfg_train, global_step,
                                                   codec_params=[], sandwich_params=[]):
    """
    Build optimizer groups for TriPlane (density, k0) and rgbnet, while freezing DCVC.
    Converts all parameter iterables to concrete lists to avoid generator exhaustion.
    """

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, torch.nn.Parameter):
            return [x]
        # nn.Module.parameters() returns a generator; coerce to list
        if isinstance(x, torch.nn.Module):
            return list(x.parameters())
        # Fallback: if it looks like an iterable of params, materialize it
        try:
            return list(x)
        except TypeError:
            return []

    # learning-rate schedule
    decay_steps  = getattr(cfg_train, 'lrate_decay', 20) * 1000
    decay_factor = (0.1 ** (global_step / decay_steps)) if decay_steps > 0 else 1.0
    def lr_of(name, default=0.0):
        base = getattr(cfg_train, f'lrate_{name}', default)
        return float(base) * decay_factor

    skip_zero = set(getattr(cfg_train, 'skip_zero_grad_fields', []))
    param_groups = []

    # --- 0) DCVC: hard-freeze everything and DO NOT add to optimizer ---
    if hasattr(model, 'codec'):
        for p in model.codec.parameters():
            p.requires_grad_(False)

    # --- 1) RGBNet group ---
    lr_rgb = lr_of('rgbnet', 0.0)
    if hasattr(model, 'rgbnet') and isinstance(model.rgbnet, torch.nn.Module):
        rgb_params = _as_list(model.rgbnet)
        if lr_rgb > 0 and len(rgb_params):
            param_groups.append({
                'params': rgb_params, 'lr': lr_rgb,
                'skip_zero_grad': ('rgbnet' in skip_zero)
            })
            # ensure trainable
            for p in rgb_params: p.requires_grad_(True)
        else:
            for p in rgb_params: p.requires_grad_(False)

    # --- 2) Per-frame TriPlane params (density, k0) ---
    lr_den = lr_of('density', 0.0)
    lr_k0  = lr_of('k0', 0.0)

    for fid, dvgo in model.dvgos.items():
        is_fixed = (int(fid) in getattr(model, 'fixed_frame', []))
        # density
        if hasattr(dvgo, 'density'):
            p_den = _as_list(dvgo.density)
            if not is_fixed and lr_den > 0 and len(p_den):
                param_groups.append({
                    'params': p_den, 'lr': lr_den,
                    'skip_zero_grad': ('density' in skip_zero)
                })
                for p in p_den: p.requires_grad_(True)
            else:
                for p in p_den: p.requires_grad_(False)
        # k0 (feature planes)
        if hasattr(dvgo, 'k0'):
            p_k0 = _as_list(dvgo.k0)
            if not is_fixed and lr_k0 > 0 and len(p_k0):
                param_groups.append({
                    'params': p_k0, 'lr': lr_k0,
                    'skip_zero_grad': ('k0' in skip_zero)
                })
                for p in p_k0: p.requires_grad_(True)
            else:
                for p in p_k0: p.requires_grad_(False)

    if sandwich_params:
        lr_sw = lr_of('sandwich', 0.0)
        if lr_sw > 0:
            sw_list = _as_list(sandwich_params)
            if len(sw_list):
                param_groups.append({
                    'params': sw_list, 'lr': lr_sw,
                    'skip_zero_grad': ('sandwich' in skip_zero)
                })
                for p in sw_list: p.requires_grad_(True)

    # Safety: check dtype while NOT consuming generators (everything is list now)
    for gi, g in enumerate(param_groups):
        plist = g['params']
        if not isinstance(plist, (list, tuple)):
            plist = list(plist)
            g['params'] = plist
        for p in plist:
            if p.dtype != torch.float32:
                raise RuntimeError(
                    f"Optimizer group {gi} contains {p.dtype} param {tuple(p.shape)}; must be float32."
                )

    return MaskedAdam(param_groups)


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


def mse2psnr_with_peak(mse: torch.Tensor, peak: float | torch.Tensor) -> torch.Tensor:
    # Broadcast peak to mse if needed; clamp mse for safety
    mse = mse.clamp_min(1e-12)
    if isinstance(peak, torch.Tensor):
        peak2 = (peak ** 2)
        return 10.0 * torch.log10(peak2 / mse)
    else:
        return 10.0 * torch.log10((peak * peak) / mse)

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