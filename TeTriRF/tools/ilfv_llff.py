#!/usr/bin/env python3
"""
Convert an Immersive Light-Field Video (ILFV) scene into the per-frame LLFF
layout expected by TeTriRF.

Directory created:

llff/
 ├ 0000/
 │   ├ images/
 │   │   ├ image_0001.jpg
 │   │   ├ image_0002.jpg
 │   │   └ …
 │   └ poses_bounds.npy        # one row per camera (17 floats)
 ├ 0001/
 │   └ …
 └ bbox.json                   # loose scene bounds (xyz_min/max)

Author: ChatGPT (adapted from the original n3d_llf.py)
"""

import argparse, json, os, re, cv2, numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_path", type=str, required=True,
                   help="Path to e.g. 11_Alexa_Meade_Face_Paint_2 (unzipped).")
    p.add_argument("--llff_path",  type=str, required=True,
                   help="Output directory that will hold the LLFF folders.")
    p.add_argument("--num_frames", type=int, default=200,
                   help="How many global frames to extract.")
    p.add_argument("--undistort", action="store_true",
                   help="Rectify fisheye to pin-hole using OpenCV fisheye model.")
    return p.parse_args()

# -----------------------------------------------------------------------------#
_num_re = re.compile(r"(\d+)")

def cam_index(name: str) -> int:
    """Turn 'camera_0001' → 1 (for ordering)."""
    m = _num_re.search(name)
    return int(m.group(1)) if m else -1

# -----------------------------------------------------------------------------#
def load_camera_models(scene_dir: Path):
    """Read models.json and construct per-camera info dict."""
    with open(scene_dir / "models.json", "r") as f:
        models = json.load(f)

    cams = {}
    for m in models:
        idx = cam_index(m["name"])
        assert idx >= 0, f"Unexpected camera name: {m['name']}"

        # World→camera rotation (axis-angle)  →  camera→world (transpose)
        R_cw = cv2.Rodrigues(np.array(m["orientation"]))[0]  # 3×3
        C_w  = np.array(m["position"])                       # camera centre (world)

        c2w = np.hstack([R_cw.T, C_w.reshape(3, 1)])         # 3×4
        # Append [H,W,F] column to get 3×5 as LLFF expects
        hwf = np.array([[m["height"]], [m["width"]], [m["focal_length"]]])
        pose_3x5 = np.hstack([c2w, hwf])                     # 3×5

        cams[idx] = {
            "pose_3x5": pose_3x5,
            "video":    scene_dir / f"{m['name']}.mp4",
            "K": np.array([[m["focal_length"], 0, m["principal_point"][0]],
                           [0, m["focal_length"], m["principal_point"][1]],
                           [0, 0, 1]], dtype=np.float64),
            "dist": np.array(m["radial_distortion"], dtype=np.float64),
            "size": (int(m["width"]), int(m["height"])),
        }
    return cams

# -----------------------------------------------------------------------------#
def init_undistort_map(cam):
    """Pre-compute fisheye undistortion maps (optional)."""
    w, h = cam["size"]
    K    = cam["K"].copy()
    D    = cam["dist"][[0, 1, 2,]]  # k1,k2,k3 (OpenCV fisheye wants 4 but k3=0 ok)
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)
    return map1, map2

# -----------------------------------------------------------------------------#
def main():
    args  = parse_args()
    scene = Path(args.scene_path).expanduser()
    out   = Path(args.llff_path).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    cams = load_camera_models(scene)
    cam_ids = sorted(cams.keys())                # deterministic order

    # --------- open VideoCapture objects & (optionally) rectification maps ----
    for cid in cam_ids:
        cap = cv2.VideoCapture(str(cams[cid]["video"]))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {cams[cid]['video']}")
        cams[cid]["cap"] = cap
        if args.undistort:
            cams[cid]["maps"] = init_undistort_map(cams[cid])

    # Near / far bounds – start conservatively; can be tightened later
    near, far = 0.1, 5.0

    # --------- iterate over global frames ------------------------------------
    for f_idx in tqdm(range(args.num_frames), desc="Frames"):
        frame_dir   = out / f"{f_idx:04d}"
        img_dir     = frame_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        pose_rows = []

        for view_i, cid in enumerate(cam_ids):
            cap = cams[cid]["cap"]
            # Seek once then read
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video {cid} shorter than {args.num_frames} frames.")

            if args.undistort:
                map1, map2 = cams[cid]["maps"]
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)

            # BGR→RGB not necessary for training; keep JPG to save space
            img_name = img_dir / f"image_{view_i:04d}.jpg"
            cv2.imwrite(str(img_name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            pose_rows.append(cams[cid]["pose_3x5"].reshape(-1))

        # poses_bounds.npy  (N_cams, 17)
        pose_arr = np.stack(pose_rows, axis=0)            # (N,15)
        bounds   = np.tile(np.array([near, far]), (pose_arr.shape[0], 1))
        np.save(frame_dir / "poses_bounds.npy",
                np.hstack([pose_arr, bounds]).astype(np.float32))

    # -------------- loose scene bounding box (optional) -----------------------
    bbox = {
        "xyz_min": [-2.0, -2.0, -2.0],
        "xyz_max": [ 2.0,  2.0,  2.0]
    }
    with open(out / "bbox.json", "w") as f:
        json.dump(bbox, f, indent=2)

    # --------- tidy up --------------------------------------------------------
    for cid in cam_ids:
        cams[cid]["cap"].release()

    print("✓ Conversion finished – ready to train with dataset_type='llff'")

# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
