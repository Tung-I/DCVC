import os, argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import cv2

"""
Usage:
    python img2triplane.py --logdir logs/out_triplane/flame_steak_old --numframe 20 --qp 30

"""

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_val = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_val

def untile_image(image, h, w, ndim):
    features = torch.zeros(1,ndim,h,w)
    x,y = 0,0
    for i in range(ndim):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "untile_image: too many feature maps"
        features[0,i,:,:] = image[x:x+h,y:y+w]
        y = y + w

    return features

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

def make_density_image(density_grid, nbits, h=3840,w=4096):
    data = density_grid +5
    data[data<0] = 0
    data = data / 30
    data[data>1.0] = 1.0
    data = torch.round(data *nbits)/nbits
    res = tile_maker(data, h=h,w=w)
    return res

if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True, help='Path to Tri-plane checkpoints (.tar files)')
    parser.add_argument('--model_template', type=str, default='fine_last_0.tar', help='model template')
    parser.add_argument("--numframe", type=int, default=10, help='number of frames')
    parser.add_argument("--codec", type=str, default='h265', help='h265 or mpg2')
    parser.add_argument("--qp", type=int, default=20, help='qp value for video codec')
    parser.add_argument("--startframe", type=int, default=0, help='start frame id')
    args = parser.parse_args()

    logdir = args.logdir
    qp = args.qp
    start_frame_id = args.startframe
    numframe = args.numframe
    model_template = args.model_template

    # directory of the decoded images
    dec_im_dir = os.path.join(logdir, f'planeimg_{start_frame_id:02d}_{start_frame_id+numframe-1:02d}')
    # xy_save_dir = os.path.join(rawim_dir, f'xy_qp{qp}')
    # xz_save_dir = os.path.join(rawim_dir, f'xz_qp{qp}')
    # yz_save_dir = os.path.join(rawim_dir, f'yz_qp{qp}')

    # directory of the saved planes
    outdir = os.path.join(args.logdir, f'planes_{start_frame_id:02d}_{start_frame_id+numframe-1:02d}_qp{qp}')
    os.makedirs(outdir, exist_ok=True)

    # load template checkpoint
    ckpt = torch.load(os.path.join(logdir, model_template), map_location='cpu', weights_only=False)

    # name = args.dir.split('/')[-2]
   
    for frameid in tqdm(range(0, numframe)):
        metadata = torch.load(os.path.join(dec_im_dir, f'planes_frame_meta.nf'))
        low_bound, high_bound = metadata['bounds']

        for key in metadata['plane_size'].keys():

            quant_img_path = os.path.join(dec_im_dir, f"{key.split('_')[0]}_qp{qp}", f"im{frameid+1:05d}_decoded.png")
            assert os.path.exists(quant_img_path), f"Quantized image {quant_img_path} does not exist"
            quant_img = cv2.imread(quant_img_path, -1)
            plane = untile_image(torch.tensor(quant_img.astype(np.float32))/int(2**16-1), 
                                metadata['plane_size'][key][2],
                                metadata['plane_size'][key][3],
                                metadata['plane_size'][key][1])
            plane = plane*(high_bound-low_bound) + low_bound
            assert 'k0.'+key in ckpt['model_state_dict'], ' Wrong plane name'
            ckpt['model_state_dict']['k0.'+key] = plane.clone().cuda()

        quant_img_path = os.path.join(dec_im_dir, f"density_qp{qp}", f"im{frameid+1:05d}_decoded.png")
        quant_img = cv2.imread(quant_img_path, -1)
        desity_plane = untile_image(torch.tensor(quant_img.astype(np.float32))/int(2**16-1), 
                                181, 280, 181)
        desity_plane = desity_plane*(30+5) - 5
        


        ckpt['model_state_dict']['density.grid'] = desity_plane.clone().cuda().unsqueeze(0)
        torch.save(ckpt, os.path.join(outdir, f'fine_last_{frameid}.tar'))


