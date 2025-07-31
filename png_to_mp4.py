#!/usr/bin/env python3
import argparse
import os
import glob
import cv2

def make_video_from_pngs(folder_path: str, fps: float = 30.0) -> None:
    # Collect all .png files in the folder
    pattern = os.path.join(folder_path, '*.png')
    png_files = sorted(glob.glob(pattern))
    if not png_files:
        raise ValueError(f"No PNG files found in {folder_path!r}")

    # Read first image to get frame size
    first_frame = cv2.imread(png_files[0])
    if first_frame is None:
        raise ValueError(f"Could not read image {png_files[0]!r}")
    height, width, channels = first_frame.shape

    # Prepare output video path
    base_name = os.path.basename(os.path.normpath(folder_path))
    output_path = os.path.join(folder_path, f"{base_name}.mp4")

    # Define the codec and create VideoWriter
    # 'mp4v' works for most installations; you can try 'avc1' if needed.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for img_path in png_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: skipping unreadable image {img_path!r}")
            continue
        # If the image size differs, resize to match the first frame
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved video to {output_path!r}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Wrap all PNGs in a folder into an MP4 video."
    )
    p.add_argument(
        "--folder",
        help="Path to the folder containing .png images"
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output video (default: 30)"
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make_video_from_pngs(args.folder, fps=args.fps)
