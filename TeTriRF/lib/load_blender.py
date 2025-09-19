import os, json
import numpy as np
import imageio
import torch
import torch.nn.functional as F

# -------------------------- DVGO-style helpers --------------------------
# (only used to make render_poses; training uses dataset transforms as-is)
trans_t = lambda t : torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=torch.float32, device='cpu')

rot_phi = lambda phi : torch.tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=torch.float32, device='cpu')

rot_theta = lambda th : torch.tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=torch.float32, device='cpu')

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    convert = torch.tensor(
        [[-1,0,0,0],
         [ 0,0,1,0],
         [ 0,1,0,0],
         [ 0,0,0,1]], dtype=torch.float32, device='cpu')
    return convert @ c2w  # stays on CPU
# -----------------------------------------------------------------------

def imread(path): 
    return imageio.imread(path)

def load_blender_data(
    basedir,
    factor=1,
    width=None,
    height=None,
    recenter=False,             # MUST stay False for Blender (DVGO leaves poses as-is)
    bd_factor=None,             # unused for Blender; keep arg for API parity
    spherify=False,             # unused; keep for API parity
    path_zflat=False,           # unused
    load_depths=False,          # Blender has no GT depth by default
    frame_id=None,              # unused; keep for parity with LLFF loader signature
    movie_render_kwargs=None,   # only used for render path params; we generate a standard spiral
    split="train",              # "train" | "val" | "test"
):
    # --- metadata ---
    meta_path = os.path.join(basedir, f"transforms_{split}.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # original Blender images are 800×800
    orig_W = orig_H = 800
    cam_angle_x = float(meta["camera_angle_x"])
    focal_orig  = 0.5 * orig_W / np.tan(0.5 * cam_angle_x)

    # --- target resolution (like DVGO: scale by factor; or explicit width/height) ---
    if height is not None and width is not None:
        H, W = int(height), int(width)
        scale = W / float(orig_W)
    elif width is not None:
        W = int(width)
        scale = W / float(orig_W)
        H = int(round(orig_H * scale))
    elif height is not None:
        H = int(height)
        scale = H / float(orig_H)
        W = int(round(orig_W * scale))
    else:
        factor = 1 if factor is None else factor
        scale  = 1.0 / float(factor)
        W = int(round(orig_W * scale))
        H = int(round(orig_H * scale))

    focal = float(focal_orig * scale)

    # --- load frames (images, c2w) ---
    images = []
    c2ws   = []
    for fr in meta["frames"]:
        # --- image ---
        fp = os.path.join(basedir, fr["file_path"] + ".png")
        if not os.path.isfile(fp):
            fp = os.path.join(basedir, fr["file_path"])
        img = imread(fp).astype(np.float32) / 255.0  # could be RGBA

        # blend alpha -> RGB (DVGO does this later in load_data; doing it here keeps your pipeline simple)
        if img.shape[-1] == 4:
            rgb, a = img[..., :3], img[..., 3:4]
            # white background (matches DVGO when args.white_bkgd=True)
            img = rgb * a + (1.0 - a)
            # if you prefer premultiplied RGB (no white bg), use:
            # img = rgb * a

        # optional resize (keeps channel count)
        if img.shape[1] != W or img.shape[0] != H:
            t = torch.from_numpy(img).permute(2,0,1)[None]  # [1,3,H,W]
            t = F.interpolate(t, size=(H, W), mode="area")
            img = t[0].permute(1,2,0).cpu().numpy()

        images.append(img)  # now guaranteed 3 channels

        # camera-to-world (OpenGL convention) — **use as-is**
        c2w_44 = np.array(fr["transform_matrix"], dtype=np.float32)  # [4,4]
        c2ws.append(c2w_44[:3, :4])  # we’ll store 3×4; intrinsics appended separately

    images = np.stack(images, axis=0)                 # [N,H,W,3 or 4], float32
    N = images.shape[0]

    # --- build poses in your code’s expected shape [N,3,5] (3×4 + [H,W,f]) ---
    poses = np.zeros((N, 3, 5), dtype=np.float32)
    poses[:, :, :4] = np.stack(c2ws, axis=0)         # [N,3,4]
    poses[:, 0, 4]  = H
    poses[:, 1, 4]  = W
    poses[:, 2, 4]  = focal

    # --- render_poses like DVGO (purely for visualization) ---
    render_poses = torch.stack(
        [pose_spherical(a, -30.0, 4.0) for a in np.linspace(-180, 180, 160, endpoint=False)],
        dim=0,
    ).to('cpu')

    # --- a simple “closest to average” holdout index (matches LLFF heuristic) ---
    c2w_centers = poses[:, :3, 3]
    center_avg  = c2w_centers.mean(0, keepdims=True)
    dists = np.sum((c2w_centers - center_avg)**2, axis=-1)
    i_test = int(np.argmin(dists))

    # No depths/bounds scaling/recentering for Blender (DVGO does none)
    depths = None
    bds    = None

    return images, depths, poses.astype(np.float32), bds, render_poses, i_test


class Blender_Dataset(torch.utils.data.Dataset):
    """
    Keeps your existing Dataset/Dataloader contract:
      __getitem__ returns: (images, poses, render_poses, i_test)
    so your load_data() doesn’t need structural changes.
    """
    def __init__(self, basedir, width=None, height=None, factor=None,
                 spherify=False, frameids=None, test_views=None, split="train"):
        self.basedir    = basedir
        self.frameids   = frameids if frameids is not None else []
        self.test_views = test_views if test_views is not None else []
        self.factor     = factor
        self.width      = width
        self.height     = height
        self.spherify   = spherify  # unused for Blender; kept for API parity
        self.split      = split     # "train" by default

    def __len__(self):
        # Keep parity with your LLFF_Dataset usage pattern
        return max(1, len(self.frameids))

    def __getitem__(self, idx):
        images, depths, poses, bds, render_poses, i_test = load_blender_data(
            self.basedir,
            factor=self.factor, width=self.width, height=self.height,
            recenter=False,                   # <-- critical: DVGO uses raw poses
            bd_factor=None,                   # no LLFF-style scaling for Blender
            spherify=False, path_zflat=False, load_depths=False,
            frame_id=None, movie_render_kwargs=None, split="train",
        )
        # For your pipeline we return the same tuple shape as LLFF_Dataset
        return images, poses, render_poses, i_test
