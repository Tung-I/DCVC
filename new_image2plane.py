import os, argparse, glob
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import cv2

def untile_image(image, h, w, ndim):
    feat = torch.zeros(1, ndim, h, w)
    x = y = 0
    for i in range(ndim):
        if y + w >= image.shape[1]:
            y = 0
            x += h
        feat[0, i] = torch.from_numpy(image[x:x+h, y:y+w])
        y += w
    return feat

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(max_val**2 / mse)

if __name__=='__main__':
    """
    Usage:
        python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 10 --strategy separate
    """
    p = argparse.ArgumentParser()
    p.add_argument('--logdir',      required=True, help='root of planeimg and checkpoints')
    p.add_argument('--model_template', default='fine_last_0.tar', help='template ckpt')
    p.add_argument('--numframe',    type=int, default=20)
    p.add_argument('--startframe',  type=int, default=0)
    p.add_argument('--qp',          type=int, default=20)
    p.add_argument('--strategy',    choices=['tiling','separate','grouped'], required=True)
    p.add_argument("--dcvc", action='store_true', help='use compressed DCVC data or not')
    args = p.parse_args()

    S, N = args.startframe, args.startframe+args.numframe-1
    root = args.logdir.rstrip('/')
    # where the decoded PNGs live
    dec = os.path.join(root, f'planeimg_{S:02d}_{N:02d}_{args.strategy}')
    if args.dcvc:
        out = os.path.join(root, f'dcvc_triplanes_{S:02d}_{N:02d}_qp{args.qp}_{args.strategy}')
    else:
        out = os.path.join(root, f'triplanes_{S:02d}_{N:02d}_qp{args.qp}_{args.strategy}')
    os.makedirs(out, exist_ok=True)

    # load template
    ckpt = torch.load(os.path.join(root, args.model_template),
                      map_location='cpu', weights_only=False)
    # for density shape fallback
    dens_shape = ckpt['model_state_dict']['density.grid'].shape  # [1,1,181,181,280]

    # load meta once
    meta = torch.load(os.path.join(dec, 'planes_frame_meta.nf'))
    bounds = meta['bounds']
    low, hi = bounds
    nbits = meta['nbits']

    for f in tqdm(range(args.numframe)):
        # re-init a fresh copy of template for each frame
        cur = {**ckpt}
        for key, size in meta['plane_size'].items():
            # e.g. key = 'xy_plane'
            C, H, W = size[1], size[2], size[3]
            feat = torch.zeros(1, C, H, W)

            if args.strategy == 'tiling':
                if args.dcvc:
                    folder = os.path.join(dec, f'dcvc_{key}_qp{args.qp}')
                    img = cv2.imread(os.path.join(folder, f'im{f+1:05d}.png'), -1)
                    feat = untile_image(img.astype(np.float32)/(2**16-1), H, W, C)
                else:
                    folder = os.path.join(dec, f'{key}_qp{args.qp}')
                    img = cv2.imread(os.path.join(folder, f'im{f+1:05d}_decoded.png'), -1)
                    feat = untile_image(img.astype(np.float32)/ (2**16-1), H, W, C)

            elif args.strategy == 'separate':
                base = os.path.join(dec, key)
                if args.dcvc:
                    for c in range(C):
                        sub = os.path.join(base, f'dcvc_c{c}_qp{args.qp}')
                        img = cv2.imread(os.path.join(sub, f'im{f+1:05d}.png'), -1)
                        feat[0,c] = torch.from_numpy(img.astype(np.float32)/(2**16-1))
                else:
                    for c in range(C):
                        sub = os.path.join(base, f'c{c}_qp{args.qp}')
                        img = cv2.imread(os.path.join(sub, f'im{f+1:05d}_decoded.png'), -1)
                        feat[0,c] = torch.from_numpy(img.astype(np.float32)/(2**16-1))

            elif args.strategy == 'grouped': 
                base = os.path.join(dec, key)
                groups = (C + 2) // 3
                for g in range(groups):
                    idxs = list(range(3*g, min(3*g+3, C)))
                    if args.dcvc:
                        sub = os.path.join(base, f'dcvc_stream{g}_qp{args.qp}')
                        arr = cv2.imread(os.path.join(sub, f'im{f+1:05d}.png'), -1)
                    else:
                        sub = os.path.join(base, f'stream{g}_qp{args.qp}')
                        arr = cv2.imread(os.path.join(sub, f'im{f+1:05d}_decoded.png'), -1)
                    if len(idxs) == 3:
                        # BGR → RGB
                        b, g_, r = cv2.split(arr)
                        for i, ch in enumerate([r, g_, b]):
                            feat[0, idxs[i]] = torch.from_numpy(ch.astype(np.float32)/(2**16-1))
                    else:
                        # leftover grayscale
                        feat[0, idxs[0]] = torch.from_numpy(arr.astype(np.float32)/(2**16-1))
            else:
                raise ValueError(f"Unknown strategy: {args.strategy}")
            # dequantize back to original range
            feat = feat * (hi - low) + low
            cur['model_state_dict'][f'k0.{key}'] = feat.clone()
            
        # rebuild density
        if args.dcvc:
            dens_folder = os.path.join(dec, f'dcvc_density_qp{args.qp}')
            img = cv2.imread(os.path.join(dens_folder, f'im{f+1:05d}.png'), -1)
        else:
            dens_folder = os.path.join(dec, f'density_qp{args.qp}')
            img = cv2.imread(os.path.join(dens_folder, f'im{f+1:05d}_decoded.png'), -1)
        d = untile_image(img.astype(np.float32)/(2**16-1),
                         dens_shape[2], dens_shape[4], dens_shape[3])
        # undo your density quantization:
        d = d * (30+5) - 5
        cur['model_state_dict']['density.grid'] = d.clone().unsqueeze(0)

        torch.save(cur, os.path.join(out, f'fine_last_{f}.tar'))
