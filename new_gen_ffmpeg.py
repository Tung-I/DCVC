#!/usr/bin/env python
"""
Generate a shell script that batch‑encodes **and** decodes the PNG frames emitted by
`triplane_packer.py` using FFmpeg.

The script supports all four packing strategies (tiling / separate / grouped /
correlation) and both quantisation modes (global, per_channel).  It discovers
sub‑streams automatically, so you do not need to hard‑code channel counts.

Example
-------
```
python gen_ffmpeg.py \
    --logdir logs/out_triplane/flame_steak_old \
    --numframe 20 \
    --strategy correlation \
    --qmode per_channel \
    --qp 22
```
"""

import argparse
import os
import re
from glob import glob

# -----------------------------------------------------------------------------

PIXFMT_GRAY   = "gray16le"   # ⇐ 16‑bit, no chroma
PIXFMT_COLOR  = "yuv444p16le"  # RGB → planar 4:4:4 16‑bit YUV
PIXFMT_DECODE = {
    "gray16le": "gray16le",
    "yuv444p16le": "rgb48be",  # decode back to 16‑bit RGB for loss analysis
}


# -----------------------------------------------------------------------------

def list_stream_dirs(base_plane_dir: str, strategy: str):
    """Return a list of directories that each hold one PNG sequence."""
    if strategy == "tiling" or strategy == "flatfour" or os.path.basename(base_plane_dir) == "density":
        return [base_plane_dir]

    if strategy == "separate":
        return sorted(d for d in glob(os.path.join(base_plane_dir, "c*")) if os.path.isdir(d))

    # grouped or correlation – use streamXX folders
    return sorted(d for d in glob(os.path.join(base_plane_dir, "stream*")) if os.path.isdir(d))

# -----------------------------------------------------------------------------

def infer_pixfmt(dirname: str, strategy: str):
    """Determine whether the stream is single‑channel or 3‑channel."""
    if strategy in ("tiling", "separate"):
        return PIXFMT_GRAY
    if "density" in dirname:
        return PIXFMT_GRAY
    # grouped / correlation – stream3 may have been grayscale *only* in the old 10‑channel case.
    # In the new 12‑channel setup every stream is RGB.
    return PIXFMT_COLOR

# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--logdir", required=True, help="Path to planeimg output directories")
    ap.add_argument("--qp", type=int, default=20, help="Quantization parameter for video compression (CRF)")
    ap.add_argument("--codec", type=str, default="h265", choices=["h265"], help="currently only libx265 pipeline tested")
    ap.add_argument("--startframe", type=int, default=0)
    ap.add_argument("--numframe", type=int, default=20)
    ap.add_argument("--strategy", type=str, required=True, choices=["tiling", "separate", "grouped", "correlation", "flatfour"])
    ap.add_argument("--qmode", type=str, required=True, choices=["global", "per_channel"])
    ap.add_argument("--planes", nargs="+", default=["xy_plane", "xz_plane", "yz_plane", "density"], help="sub‑folders to include")
    args = ap.parse_args()

    pwd = os.getcwd()
    base_dir = os.path.join(pwd, args.logdir,
                            f"planeimg_{args.startframe:02d}_{args.startframe + args.numframe - 1:02d}_"
                            f"{args.strategy}_{args.qmode}")

    script_path = os.path.join(base_dir, f"ffmpeg_qp{args.qp}_{args.strategy}_{args.qmode}.sh")
    os.makedirs(base_dir, exist_ok=True)

    # If sript already exists, delete it to avoid confusion
    if os.path.exists(script_path):
        print(f"Warning: Script {script_path} already exists. It will be overwritten.")
        os.remove(script_path)

    with open(script_path, "w") as sh:
        sh.write("#!/usr/bin/env bash\nset -e\n\n")

        for plane in args.planes:
            plane_root = os.path.join(base_dir, plane)
            for stream_dir in list_stream_dirs(plane_root, args.strategy):
                if 'qp' in stream_dir:
                    continue
                pix_in  = infer_pixfmt(stream_dir, args.strategy)
                pix_out = PIXFMT_DECODE[pix_in]

                # move into folder with PNGs
                sh.write(f"cd {stream_dir}\n")

                # Encode → video.mp4
                encode_filter = f"-vf format={pix_in} -pix_fmt {pix_in}"
                sh.write(
                    f"ffmpeg -y -framerate 30 -i im%05d.png {encode_filter} -c:v libx265 -crf {args.qp} video.mp4\n"
                )

                # create target folder, move video
                encoded_dir = f"../{os.path.basename(stream_dir)}_qp{args.qp}".replace("//", "/")
                sh.write(f"mkdir -p {encoded_dir}\n")
                sh.write(f"mv video.mp4 {encoded_dir}/\n")

                # Decode back to PNG
                sh.write(f"cd {encoded_dir}\n")
                sh.write(
                    f"ffmpeg -y -i video.mp4 -pix_fmt {pix_out} im%05d_decoded.png\n\n"
                )

    os.chmod(script_path, 0o755)
    print("Generated FFmpeg script:", script_path)


if __name__ == "__main__":
    main()
