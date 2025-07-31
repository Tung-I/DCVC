#!/usr/bin/env python3
"""
Extract the first N frames from each MP4 in an ILFV scene into the directory layout
that LLFF's imgs2poses.py expects: one folder per time step named frame######,
containing an 'images/' subfolder with camXX.png files.

Usage:
  python extract_ilfv_frames.py \
    --input_dir /path/to/11_Alexa_Meade_Face_Paint_2 \
    --output_dir /path/to/llff/scene \
    --num_frames 20

Resulting structure:

scene/
 ├ frame000000/
 │   └ images/
 │       ├ cam01.png
 │       ├ cam02.png
 │       └ ...
 ├ frame000001/
 │   └ images/
 │       └ ...
 └ frame000019/
     └ images/
         └ cam46.png

Author: ChatGPT
"""
import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path

"""
Usage:
    python extract_frames.py --input_dir $WORK/datasets/ILFV/11_Alexa_Meade_Face_Paint_2 --num_frames 20
"""

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract N frames per MP4 into LLFF-style frame######/images/camXX.png layout"
    )
    p.add_argument(
        "--input_dir", required=True,
        help="Directory containing ILFV `.mp4` files (e.g. camera_0001.mp4, ...)."
    )
    # p.add_argument(
    #     "--output_dir", required=True,
    #     help="Root directory to write frame directories (e.g. llff/scene)."
    # )
    p.add_argument(
        "--num_frames", type=int, default=20,
        help="Number of frames to extract from each video."
    )
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input_dir)
    # out = Path(args.output_dir)
    # out.mkdir(parents=True, exist_ok=True)
    out = inp

    # List all .mp4 files and sort them
    videos = sorted([f for f in inp.iterdir() if f.suffix.lower() == '.mp4'])
    num_cams = len(videos)
    if num_cams == 0:
        raise RuntimeError(f"No .mp4 files found in {inp}")

    # Pre-open all video captures
    caps = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {v}")
        caps.append(cap)

    # For each time step
    for t in tqdm(range(args.num_frames), desc="Frames"):
        frame_dir = out / f"frame{t:06d}"
        img_dir   = frame_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Extract one frame per camera
        for cam_idx, cap in enumerate(caps, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(
                    f"Video {videos[cam_idx-1].name} shorter than {args.num_frames} frames"
                )

            # Save as PNG
            out_path = img_dir / f"image_{cam_idx-1:04d}.png"
            cv2.imwrite(str(out_path), frame)

    # Release captures
    for cap in caps:
        cap.release()

    print("Done: extracted frames into LLFF layout.")

if __name__ == '__main__':
    main()
