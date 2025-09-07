# load_blender.py
import os
import json
import numpy as np
import imageio
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Utilities copied/adapted from your LLFF loader so shapes/behavior match
# ---------------------------------------------------------------------

def imread(path):
    return imageio.imread(path)

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    """poses: [N,3,5] -> returns average c2w in same (3x5) style."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses):
    """match LLFF logic; returns same shape as input."""
    poses_ = poses.copy()
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses_homo = np.concatenate([poses[:,:3,:4], bottom], -2)          # [N,4,4]
    poses_recentered = np.linalg.inv(c2w) @ poses_homo                  # [N,4,4]
    poses_[:,:3,:4] = poses_recentered[:,:3,:4]
    return poses_

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

# ---------------------------------------------------------------------
# Core loader (LLFF-compatible return signature)
# ---------------------------------------------------------------------

def load_blender_data(
    basedir,
    factor=1,
    width=None,
    height=None,
    recenter=True,
    bd_factor=.75,
    spherify=False,
    path_zflat=False,
    load_depths=False,
    frame_id=None,                # unused; kept for signature parity
    movie_render_kwargs=None,
    split="train",
):
    """
    Returns (images, depths, poses, bds, render_poses, i_test) with the SAME
    shapes/semantics as your LLFF loader so you can swap datasets seamlessly.
    """
    if movie_render_kwargs is None:
        movie_render_kwargs = {}

    # --- read metadata ---
    meta_path = os.path.join(basedir, f"transforms_{split}.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"transforms file not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Original Blender images are 800×800 by default
    orig_W = 800
    orig_H = 800
    camera_angle_x = float(meta["camera_angle_x"])  # in radians
    focal_orig = 0.5 * orig_W / np.tan(0.5 * camera_angle_x)

    # --- decide target resolution & scale focal accordingly ---
    if height is not None and width is not None:
        target_H, target_W = int(height), int(width)
        scale = target_W / float(orig_W)
    elif width is not None:
        target_W = int(width)
        scale = target_W / float(orig_W)
        target_H = int(round(orig_H * scale))
    elif height is not None:
        target_H = int(height)
        scale = target_H / float(orig_H)
        target_W = int(round(orig_W * scale))
    else:
        # fall back to factor (like LLFF)
        factor = 1 if factor is None else factor
        scale = 1.0 / float(factor)
        target_W = int(round(orig_W * scale))
        target_H = int(round(orig_H * scale))

    focal = focal_orig * scale

    # --- load images + poses ---
    images = []
    c2ws = []
    for fr in meta["frames"]:
        # image path
        img_path = os.path.join(basedir, fr["file_path"] + ".png")
        if not os.path.isfile(img_path):
            # some datasets store relative paths already ending with .png
            img_path = os.path.join(basedir, fr["file_path"])
        img = imread(img_path)  # RGBA or RGB

        # ensure float [0,1] and H×W×3 after alpha blending on white bg (like common NeRF code)
        img = (img.astype(np.float32) / 255.0)
        if img.shape[-1] == 4:
            rgb, alpha = img[..., :3], img[..., 3:4]
            img = rgb * alpha + (1.0 - alpha)  # composite over white
        # resize if needed
        if (img.shape[1] != target_W) or (img.shape[0] != target_H):
            # imageio's imresize deprecated; use torch for consistent results
            t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
            t = F.interpolate(t, size=(target_H, target_W), mode="area")
            img = t.squeeze(0).permute(1,2,0).cpu().numpy()

        images.append(img)

        # c2w (Blender format is OpenGL; we’ll convert to the convention your LLFF code expects)
        c2w = np.array(fr["transform_matrix"], dtype=np.float32)  # [4,4]

        # Match LLFF downstream convention by re-basing axes as in your LLFF loader.
        # LLFF later does: [y, -x, z] reordering; we’ll stick to the same pipeline:
        c2ws.append(c2w[:3, :4])

    images = np.stack(images, axis=0).astype(np.float32)  # [N,H,W,3]
    N = images.shape[0]
    H, W = target_H, target_W

    # --- build poses [N,3,5]: 3x4 c2w + [H,W,focal] ---
    poses = np.zeros((N, 3, 5), dtype=np.float32)
    poses[:, :, :4] = np.stack(c2ws, axis=0)
    poses[:, 0, 4] = H
    poses[:, 1, 4] = W
    poses[:, 2, 4] = focal

    # ---- match LLFF axis convention (critical for compatibility with your renderer) ----
    # same as your LLFF code:
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

    # --- bounds (near/far) per image; Blender synthetic commonly uses [2,6] ---
    bds = np.tile(np.array([[2.0, 6.0]], dtype=np.float32), (N, 1))

    # --- optional global rescale like LLFF (uses min depth) ---
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc
    depths = 0  # no GT depth for blender by default

    # --- recenter / spherify and compute render_poses like LLFF ---
    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        # Synthetic scenes usually don’t need this, but keep parity with LLFF API:
        # If you have an existing spherify_poses implementation, import and call it here.
        render_poses = [poses_avg(poses)]
    else:
        c2w_avg = poses_avg(poses)
        up = normalize(poses[:, :3, 1].sum(0))
        # LLFF-style spiral defaults
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal_for_spiral = mean_dz * movie_render_kwargs.get('scale_f', 1.0)

        shrink_factor = .8
        zdelta = movie_render_kwargs.get('zdelta', 0.1)
        zrate  = movie_render_kwargs.get('zrate', 0.5)
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0) * movie_render_kwargs.get('scale_r', 0.8)
        c2w_path = c2w_avg.copy()
        N_views = int(movie_render_kwargs.get('N_views', 97))
        N_rots  = float(movie_render_kwargs.get('N_rots', 1.0))
        if path_zflat:
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views = max(1, N_views//2)

        render_poses = render_path_spiral(c2w_path, up, rads, focal_for_spiral,
                                          zdelta, zrate=zrate, rots=N_rots, N=N_views)

    render_poses = torch.tensor(render_poses, device='cpu')

    # --- choose a holdout view (same heuristic as LLFF) ---
    c2w = poses_avg(poses)
    dists = np.sum(np.square(c2w[:3,3] - poses[:, :3, 3]), -1)
    i_test = int(np.argmin(dists))

    return images.astype(np.float32), depths, poses.astype(np.float32), bds.astype(np.float32), render_poses, i_test


# ---------------------------------------------------------------------
# Dataset wrapper with the SAME interface as your LLFF_Dataset
# ---------------------------------------------------------------------

class Blender_Dataset(torch.utils.data.Dataset):
    def __init__(self, basedir, width=None, height=None, factor=None,
                 spherify=False, frameids=None, test_views=None, split="train"):
        self.basedir   = basedir
        self.frameids  = frameids if frameids is not None else []
        self.test_views = test_views if test_views is not None else []
        self.factor    = factor
        self.width     = width
        self.height    = height
        self.spherify  = spherify
        self.split     = split  # "train" | "val" | "test"

    def __len__(self):
        # keep parity with LLFF_Dataset contract
        return max(1, len(self.frameids))

    def __getitem__(self, idx):
        # LLFF_Dataset returns (images, poses, render_poses, i_test)
        # We ignore frameid semantics here (Blender doesn’t have per-frame subfolders).
        movie_render_kwargs = {
            'scale_r': 0.8,
            'scale_f': 8.0,
            'zdelta' : 0.1,
            'zrate'  : 0.1,
            'N_rots' : 1.0,
            'N_views': 97,
        }
        images, depths, poses, bds, render_poses, i_test = load_blender_data(
            self.basedir,
            factor=self.factor,
            width=self.width,
            height=self.height,
            spherify=self.spherify,
            frame_id=None,
            movie_render_kwargs=movie_render_kwargs,
            split=self.split,
        )
        return images, poses, render_poses, i_test
