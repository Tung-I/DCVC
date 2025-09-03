import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_encoded_filesize(
    logdir: str,
    start_frame: int,
    num_frames: int,
    strategy: str,
    qp: int
) -> float:
    """
    Sum up all .mp4 sizes (in KB) under
      {logdir}/planeimg_{start:02d}_{end:02d}_{strategy}/*_qp{qp}
    """
    end_frame = start_frame + num_frames - 1
    if 'dcvc' in strategy:
        strat_name = strategy.replace('dcvc_', '')
        pwd = os.getcwd()
        logdir = os.path.join(pwd, 'logs/out_bin')
        total_bytes = 0
        if 'tiling' in strategy:
            base = os.path.join(logdir, f'dcvc_planeimg_{start_frame:02d}_{end_frame:02d}_{strat_name}')
            assert os.path.exists(base), f"Directory {base} does not exist."
            for dirpath, _, files in os.walk(base):
                # collect all file sizes end with qp{qp}.bin
                for fn in files:
                    if fn.lower().endswith(f'_q{qp}.bin'):
                        total_bytes += os.path.getsize(os.path.join(dirpath, fn))
        else:
            for p in ['xy_plane', 'xz_plane', 'yz_plane', 'density']:
                base = os.path.join(logdir, f'dcvc_planeimg_{start_frame:02d}_{end_frame:02d}_{strat_name}_{p}')
                assert os.path.exists(base), f"Directory {base} does not exist."
                for dirpath, _, files in os.walk(base):
                    # collect all file sizes end with qp{qp}.bin
                    for fn in files:
                        if fn.lower().endswith(f'_q{qp}.bin'):
                            total_bytes += os.path.getsize(os.path.join(dirpath, fn))
    else:
        base = os.path.join(
            logdir,
            f'planeimg_{start_frame:02d}_{end_frame:02d}_{strategy}'
        )
        total_bytes = 0
        for dirpath, _, files in os.walk(base):
            if os.path.basename(dirpath).endswith(f'_qp{qp}'):
                for fn in files:
                    if fn.lower().endswith('.mp4'):
                        total_bytes += os.path.getsize(os.path.join(dirpath, fn))
    return total_bytes / 1024.0  # in KB

def get_eval_result(
    logdir: str,
    start_frame: int,
    num_frames: int,
    strategy: str,
    qps: list[int]
):
    """
    For each QP, compute:
      - average PSNR, SSIM, LPIPS over frames
      - average encoded KB/frame
    """
    datasizes, psnrs, ssims, lpipss = [], [], [], []
    S, N = start_frame, start_frame + num_frames - 1

    for qp in qps:
        # 1) compute compressed size (KB) per frame
        total_kb = get_encoded_filesize(logdir, S, num_frames, strategy, qp)
        datasizes.append(total_kb / num_frames)

        # 2) load render_test metrics
        if 'dcvc' in strategy:

            strat_name = strategy.replace('dcvc_', '')
          
            eval_dir = os.path.join(
                logdir,
                f'dcvc_triplanes_{S:02d}_{N:02d}_qp{qp}_{strat_name}',
                'render_test'
            )
        else:
            eval_dir = os.path.join(
                logdir,
                f'triplanes_{S:02d}_{N:02d}_qp{qp}_{strategy}',
                'render_test'
            )
        tmp_psnr, tmp_ssim, tmp_lpips = [], [], []
        for i in range(S, N+1):
            tmp_psnr.append(np.loadtxt(os.path.join(eval_dir, f'{i}_psnr.txt')))
            tmp_ssim.append(np.loadtxt(os.path.join(eval_dir, f'{i}_ssim.txt')))
            tmp_lpips.append(np.loadtxt(os.path.join(eval_dir, f'{i}_lpips.txt')))
        psnrs.append(float(np.mean(tmp_psnr)))
        ssims.append(float(np.mean(tmp_ssim)))
        lpipss.append(float(np.mean(tmp_lpips)))

        print(f"[{strategy} @ QP={qp}] size={datasizes[-1]:.1f} KB/frame, PSNR={psnrs[-1]:.2f} dB")

    return datasizes, psnrs, ssims, lpipss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',     required=True, help='root folder containing planeimg_* and planes_* dirs')
    parser.add_argument('--startframe', type=int, default=0)
    parser.add_argument('--numframe',   type=int, default=20)
    parser.add_argument('--qps',        nargs='+', type=int, default=[10,18,23,28,33])
    parser.add_argument('--dcvc_qps',  nargs='+', type=int, default=[0,21,42,63])
    args = parser.parse_args()

    # strategies = {
    #     'tiling'  : 'Flattened (1 stream)',
    #     'separate': 'Separate (10 streams)',
    #     'grouped' : 'Grouped (3 rgb + 1 density stream)',
    #     'dcvc_tiling'  : 'DCVC + Flattened',
    #     'dcvc_separate': 'DCVC + Separate',
    #     'dcvc_grouped' : 'DCVC + Grouped'
    # }
    strategies = {
        'tiling_global'  : 'Flattened (1-channel, 1 stream)',
        'separate_global': 'Separate (1-channel, 12 streams)',
        'grouped_global' : 'Grouped (3-channel, 4 streams)',
        'correlation_global': 'C-Grouped (3-channel, 4 streams)',
        'flatfour_global': 'FlatFour (3-channel, 1 stream)',
        'tiling_per_channel'  : 'Flattened (per-ch quant.)',
        'separate_per_channel': 'Separate (per-ch quant.)',
        'grouped_per_channel' : 'Grouped (per-ch quant.)',
        'correlation_per_channel': 'C-Grouped (per-ch quant.)',
        'flatfour_per_channel': 'FlatFour (per-ch quant.)'    
    }

    # collect all RD data
    rows = []
    for strat, label in strategies.items():
        if 'dcvc' in strat:
            ds, ps, ss, lp = get_eval_result(
                args.logdir, args.startframe, args.numframe,
                strat, args.dcvc_qps
            )
            for qp, d, p in zip(args.dcvc_qps, ds, ps):
                rows.append({
                    'Bitrate (KB/frame)': d,
                    'PSNR (dB)'          : p,
                    'Strategy'           : label,
                    'QP'                 : qp
                })
        else:
            ds, ps, ss, lp = get_eval_result(
                args.logdir, args.startframe, args.numframe,
                strat, args.qps
            )
            for qp, d, p in zip(args.qps, ds, ps):
                rows.append({
                    'Bitrate (KB/frame)': d,
                    'PSNR (dB)'          : p,
                    'Strategy'           : label,
                    'QP'                 : qp
                })

    df = pd.DataFrame(rows)

    # plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(8,5))
    sns.lineplot(
        data=df,
        x="Bitrate (KB/frame)",
        y="PSNR (dB)",
        hue="Strategy",
        style="Strategy",
        markers=True,
        dashes=False,
        linewidth=1.0
    )
    plt.gca().invert_xaxis()   # smaller size = better compression → on the right
    plt.xlabel("Bitrate (KB/frame)")
    plt.ylabel("PSNR (dB)")
    plt.title("Offline Tri-plane Encoding Strategies")
    plt.legend(title="")
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/rd_curve.png', dpi=300)
    print("Saved RD curve to results/rd_curve.png")
