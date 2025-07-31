#!/usr/bin/env python3
"""
Convert ILFV sequence -> LLFF poses_bounds.npy + rectified pin-hole images
compatible with TeTriRF's load_llff.py (expects 17 cols per row).

Each row (float32) in poses_bounds.npy:
    R(3×3) | t(3) | H W f | near far      = 17 values
"""

import os, json, argparse, re
import numpy as np
import cv2

# OpenCV ➜ NeRF axis flip:  +x right, +y up,  -z forward
CV2_TO_NERF = np.diag([1, -1, -1, 1]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Camera helpers
# --------------------------------------------------------------------------- #

def axisangle_to_mat(rvec):
    """3-vector axis-angle -> 3×3 rotation matrix (float32)."""
    return cv2.Rodrigues(np.asarray(rvec, np.float64))[0].astype(np.float32)


def new_pinhole_K(cam):
    """
    Create a *centred* pin-hole intrinsics matrix with the original focal length.
    This guarantees cx = W/2, cy = H/2 as expected by LLFF loader.
    """
    H, W = int(cam["height"]), int(cam["width"])
    f    = float(cam["focal_length"])
    return np.array([[f, 0, W * 0.5],
                     [0, f, H * 0.5],
                     [0, 0, 1]], np.float32)


def build_undistort_maps(cam):
    """Return rectification maps (map1, map2) for this camera."""
    K0 = np.array([[cam["focal_length"], 0,  cam["principal_point"][0]],
                   [0,  cam["focal_length"],  cam["principal_point"][1]],
                   [0,  0,                   1]], np.float32)
    D  = np.array(cam["radial_distortion"][:3] + [0.0], np.float32)  # k1,k2,k3,k4
    H, W = int(cam["height"]), int(cam["width"])

    newK = new_pinhole_K(cam)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K0, D, np.eye(3, dtype=np.float32), newK, (W, H), cv2.CV_32FC1)
    return map1, map2, newK[0, 0]   # last value is centred focal length


def make_pose_row(cam, f_centre):
    """
    Return 17-value row: 12 extrinsics + H W f + (near far will be appended later)
    """
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = axisangle_to_mat(cam["orientation"]).T
    c2w[:3,  3] = np.asarray(cam["position"], np.float32)
    c2w = CV2_TO_NERF @ c2w

    H, W = cam["height"], cam["width"]
    intr = np.array([H, W, f_centre], np.float32)        # 3 values
    return np.hstack([c2w[:3, :4].reshape(-1), intr])    # (15,)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main(scene_root, models_json, factor, near_scale, far_scale):
    # 1) load per-camera calibration
    with open(models_json, "r") as f:
        cams = {int(re.search(r"(\d+)$", c["name"]).group(1)): c
                for c in json.load(f)}

    # 2) compute near / far bounds (scene-level, metres)
    centres = np.stack([c["position"] for c in cams.values()])
    dists   = np.linalg.norm(centres - centres.mean(0), axis=1)
    near, far = max(0.1, dists.min()*near_scale), dists.max()*far_scale

    # 3) pre-compute undistortion maps and centred focal length per camera
    for cam in cams.values():
        cam["map1"], cam["map2"], cam["f_centre"] = build_undistort_maps(cam)

    img_suffix = f"_{factor}" if factor not in (None, 1) else ""
    for fr in sorted(os.listdir(scene_root)):
        # if fr is not a directory, skip
        if not os.path.isdir(os.path.join(scene_root, fr)):
            continue
        frame_dir = os.path.join(scene_root, fr)
        imgdir    = os.path.join(frame_dir, f"images{img_suffix}")
        undistdir = os.path.join(frame_dir, f"undistorted_images{img_suffix}")
        os.makedirs(undistdir, exist_ok=True)
        if not os.path.isdir(imgdir):
            continue

        files = sorted(f for f in os.listdir(imgdir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not files:
            continue

        # map filename -> camera
        mapping = {}
        for fn in files:
            m = re.search(r"(\d+)", fn)
            if m and (idx := int(m.group(1))) in cams:
                mapping[fn] = cams[idx]
        if not mapping:
            print(f"[{fr}] no cam IDs matched, skipped.")
            continue

        # 3-A) rectify images *once* per script run
        for fn, cam in mapping.items():
            fpath = os.path.join(imgdir, fn)
            img   = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            undist = cv2.remap(img, cam["map1"], cam["map2"], cv2.INTER_LINEAR)
            out_fpath = os.path.join(undistdir, fn)
            cv2.imwrite(out_fpath, undist)

        # 3-B) build poses_bounds.npy
        rows = np.stack([make_pose_row(cam, cam["f_centre"])
                         for cam in mapping.values()], 0)
        bds  = np.repeat([[near, far]], len(rows), axis=0)
        out  = np.hstack([rows.astype(np.float32), bds])   # (N, 17)
        np.save(os.path.join(frame_dir, "poses_bounds.npy"), out)
        print(f"[{fr}] {len(rows)} views  near={near:.2f}  far={far:.2f}")

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", required=True,
                    help="ILFV sequence directory (contains frame sub-dirs)")
    parser.add_argument("--models", required=True,
                    help="models.json path")
    parser.add_argument("--factor", type=int, default=None,
                    help="TeTriRF cfg.data.factor (only for image folder name)")
    parser.add_argument("--near_scale", type=float, default=0.9)
    parser.add_argument("--far_scale",  type=float, default=1.1)
    args = parser.parse_args()
    main(args.scene_root, args.models, args.factor,
         args.near_scale, args.far_scale)
