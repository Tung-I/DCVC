import os, sys, argparse
import glob

if __name__=='__main__':
    """
    Usage:
        python new_gen_ffmpeg.py --logdir logs/out_triplane/flame_steak_old --numframe 20 --strategy grouped --qp 20
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True, help='Path to planeimg output directories')
    parser.add_argument('--qp', type=int, default=20, help='Quantization parameter for video compression')
    parser.add_argument('--codec', type=str, default='h265', help='h265 or mpg2')
    parser.add_argument('--startframe', type=int, default=0, help='start frame id')
    parser.add_argument('--numframe', type=int, default=20, help='number of frames')
    parser.add_argument('--strategy', type=str, default='tiling', choices=['tiling','separate','grouped'],
                        help='triplane-to-image strategy')
    parser.add_argument('--planes', nargs='+', default=['xy_plane','xz_plane','yz_plane','density'],
                        help='List of plane subfolders')
    args = parser.parse_args()
    pwd = os.getcwd()
    base = os.path.join(pwd, args.logdir, f'planeimg_{args.startframe:02d}_{args.startframe+args.numframe-1:02d}_{args.strategy}')
    fname = f'{base}/ffmpeg_qp{args.qp}_{args.strategy}.sh'

    with open(fname, 'w') as f:
        for p in args.planes:
            # determine directories depending on strategy
            plane_root = os.path.join(base, p)
            if args.strategy == 'tiling' or p == 'density':
                dirs = [plane_root]
            elif args.strategy == 'separate':
                dirs = [os.path.join(plane_root, 'c0'),
                        os.path.join(plane_root, 'c1'),
                        os.path.join(plane_root, 'c2'),
                        os.path.join(plane_root, 'c3'),
                        os.path.join(plane_root, 'c4'),
                        os.path.join(plane_root, 'c5'),
                        os.path.join(plane_root, 'c6'),
                        os.path.join(plane_root, 'c7'),
                        os.path.join(plane_root, 'c8'),
                        os.path.join(plane_root, 'c9')]
            else:  # grouped
                dirs = [os.path.join(plane_root, 'stream0'),
                        os.path.join(plane_root, 'stream1'),
                        os.path.join(plane_root, 'stream2'),
                        os.path.join(plane_root, 'stream3')]

            for dirpath in dirs:
                # switch into the directory containing PNGs
                f.write(f'cd {dirpath}\n')

                # choose pixel formats
                if p == 'density' or args.strategy == 'tiling' or args.strategy == 'separate':
                    pix_in = '-pix_fmt gray12le'
                    pix_out = '-pix_fmt gray16be'
                elif args.strategy == 'grouped':
                    if os.path.basename(dirpath) == 'stream3':
                        pix_in = '-pix_fmt gray12le'
                        pix_out = '-pix_fmt gray16be'
                    else:
                        pix_in = '-pix_fmt yuv444p10le'
                        pix_out = '-pix_fmt rgb48be'
                else:
                    raise ValueError(f"Unknown strategy: {args.strategy}")
                # encode into video.mp4
                f.write(f"ffmpeg -y -framerate 30 -i im%05d.png -c:v libx265 {pix_in} -crf {args.qp} video.mp4\n")
               
                # create output folder and move encoded video
                folder_name = f"{os.path.basename(dirpath)}_qp{args.qp}"
                f.write(f"mkdir -p ../{folder_name}\n")
                f.write(f"mv video.mp4 ../{folder_name}/\n")

                # decode inside the new folder
                f.write(f"cd ../{folder_name}\n")
                f.write(f"ffmpeg -y -i video.mp4 {pix_out} im%05d_decoded.png\n")
                f.write("\n")

    os.system(f"chmod +x {fname}")
    print(f"Generated FFmpeg script: {fname}")
