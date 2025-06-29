import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


if __name__=='__main__':
    """
    Usage:
        ython triplane2img.py --logdir logs/out_triplane/flame_steak_old --numframe 20
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True, help='Path to Tri-plane checkpoints (.tar files)')
    parser.add_argument('--qp', type=int, default=20, help='Quantization parameter for video compression')
    parser.add_argument("--numframe", type=int, default=20, help='number of frames')
    parser.add_argument("--codec", type=str, default='h265', help='h265 or mpg2')
    parser.add_argument("--startframe", type=int, default=0, help='start frame id')

    args = parser.parse_args()

    logdir = args.logdir
    qp = args.qp
    start_frame_id = args.startframe
    numframe = args.numframe

    save_dir = os.path.join(args.logdir, f'planeimg_{start_frame_id:02d}_{numframe-1:02d}')
    xy_save_dir = os.path.join(save_dir, f'xy_qp{qp}')
    xz_save_dir = os.path.join(save_dir, f'xz_qp{qp}')
    yz_save_dir = os.path.join(save_dir, f'yz_qp{qp}')
    density_save_dir = os.path.join(save_dir, f'density_qp{qp}')
    os.makedirs(xy_save_dir, exist_ok=True)
    os.makedirs(xz_save_dir, exist_ok=True)
    os.makedirs(yz_save_dir, exist_ok=True)
    os.makedirs(density_save_dir, exist_ok=True)

    if not os.path.exists(save_dir):
        raise Exception(f"Save directory {save_dir} does not exist. Please run triplane2img.py first.")
    filename = f'{save_dir}/ffmpeg_qp{qp}.sh'

    with open(filename,'w') as f:
        # go to the original density directory
        f.write(f'cd density\n')

        # encode density images
        if args.codec =='h265':
            #Q: say the images are named im00001.png, im00002.png, ..., im00020.png, how to sepcify the input images?
            # A: 
            f.write(f"ffmpeg -y -framerate 30 -i im%05d.png -c:v libx265 -pix_fmt gray12le -color_range pc -crf {args.qp} density_planes.mp4\n")
            # move the encoded video to the new save directory
            f.write(f"mv density_planes.mp4 ../density_qp{qp}/\n")
            # go to the new save directory
            f.write(f'cd ../density_qp{qp}\n')
            # decompress the video to images
            f.write(f"ffmpeg -y -i density_planes.mp4  -pix_fmt gray16be  im%05d_decoded.png\n")

            for p in ['xy','xz','yz']:
                f.write(f'cd ../{p}\n')
                # encode the plane images
                f.write(f"ffmpeg -y -framerate 30 -i im%05d.png -c:v libx265 -pix_fmt gray12le -color_range pc   -crf {args.qp}  {p}_planes.mp4\n")
                # move the encoded video to the new save directory: {p}_save_dir
                f.write(f"mv {p}_planes.mp4 ../{p}_qp{qp}/\n")
                # go to the new save directory of that plane: {p}
                f.write(f'cd ../{p}_qp{qp}\n')
                # decompress the video to images
                f.write(f"ffmpeg -y -i {p}_planes.mp4  -pix_fmt gray16be im%05d_decoded.png\n")

    os.system(f"chmod +x {filename}")