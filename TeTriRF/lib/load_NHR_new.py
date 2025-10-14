# load_NHR.py
import os
import json
import collections

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

# Keep dataloader workers light-weight
try:
    cv2.setNumThreads(0)
except Exception:
    pass


class Image_Transforms(object):
    """
    Outputs (matching original semantics):
      - img      : float32 (H,W,3) in [0,1], resized to `size`
      - mask     : float32 (H,W,1) in [0,1], resized to `size`
      - img_ori  : float32 (H',W',3) in [0,1], resized to `ori_size` (fixed)
      - K        : intrinsics updated after translate+crop+scale
      - Tc       : 4x4 pose (unchanged)
    """
    def __init__(self, size, interpolation=Image.BICUBIC, is_center=False, isNHR=True,
                 ori_size=(1080, 1920)):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.ori_size = ori_size
        self.interpolation = interpolation
        self.is_center = is_center
        self.isNHR = isNHR

    def __call__(self, img_np, K_in, Tc, mask_np=None, residual=None):
        # Defensive copies; we mutate K
        K = K_in.copy()

        # PIL images
        img_pil  = Image.fromarray(img_np.astype('uint8'), 'RGB')
        if mask_np is None:
            mask_np = np.ones_like(img_np, dtype=np.uint8) * 255
        # Mask may be RGB JPG; treat as generic image and take first channel later
        mask_pil = Image.fromarray(mask_np.astype('uint8'))

        width, height = img_pil.size

        # Centering & crop ratio (same as original)
        translation = [0, 0]
        ration = 1.0
        if self.is_center:
            translation = [width / 2 - K[0, 2], height / 2 - K[1, 2]]
            ration = 1.05
            if (self.size[1] / 2) / (self.size[0] * ration / height) - K[0, 2] != translation[0]:
                ration = 1.2
            if not self.isNHR:
                ration = 1.0
            translation[1] = (self.size[0] / 2) / (self.size[0] * ration / height) - K[1, 2]
            translation[0] = (self.size[1] / 2) / (self.size[0] * ration / height) - K[0, 2]
            translation = tuple(translation)

        # Translate then crop
        img_pil  = T.functional.affine(img_pil,  angle=0, translate=translation, scale=1, shear=0)
        mask_pil = T.functional.affine(mask_pil, angle=0, translate=translation, scale=1, shear=0)

        crop_h = int(height / ration)
        crop_w = int(height * self.size[1] / ration / self.size[0])
        img_pil_c  = T.functional.crop(img_pil,  0, 0, crop_h, crop_w)
        mask_pil_c = T.functional.crop(mask_pil, 0, 0, crop_h, crop_w)

        # “Original” view at a fixed size to avoid stacking issues
        img_ori_pil = T.functional.resize(img_pil_c, self.ori_size, self.interpolation)

        # Final target size
        img_pil_r  = T.functional.resize(img_pil_c,  self.size, self.interpolation)
        mask_pil_r = T.functional.resize(mask_pil_c, self.size, self.interpolation)

        # Convert with TorchVision (→ float32 in [0,1]) and back to HWC
        img_t   = T.functional.to_tensor(img_pil_r).permute(1, 2, 0).contiguous()       # (H,W,3) float32
        img_ori = T.functional.to_tensor(img_ori_pil).permute(1, 2, 0).contiguous()     # (H',W',3)
        mask_t  = T.functional.to_tensor(mask_pil_r).permute(1, 2, 0)[..., :1].contiguous()  # (H,W,1)

        # Update intrinsics to reflect translate+crop+scale
        K[0, 2] += translation[0]
        K[1, 2] += translation[1]
        s = self.size[0] * ration / height
        K = K * s
        K[2, 2] = 1.0

        ROI = None  # Unused downstream; keep API slot

        # Return numpy for K / Tc consistency with original code
        return img_t, K.astype(np.float32), Tc.astype(np.float32), mask_t, None, ROI, img_ori


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class NHR_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, frameids=None, test_views=None, tar_size=(720, 960), cam_num=-1, isNHR=True):
        super().__init__()
        self.cam_num = cam_num
        self.frameids = frameids or []
        self.path = path
        self.transforms = Image_Transforms(tar_size, isNHR=isNHR)
        self.test_views = test_views or []
        self.isNHR = isNHR

    def read_frame(self, frame_id, cam_num=-1):
        jpath = os.path.join(self.path, f'cams_{frame_id}.json')
        with open(jpath, 'r') as f:
            meta = json.load(f)

        frames = sorted(meta["frames"], key=lambda d: d['file'])

        if cam_num < 0:
            cameras = list(range(len(frames)))
        else:
            # random subset
            idx = torch.randperm(len(frames), device='cpu').tolist()
            cameras = idx[:cam_num] if cam_num > 0 else idx

        poses, images, intrinsic, masks, images_ori = [], [], [], [], []

        for i in cameras:
            f = frames[i]
            f_img  = os.path.join(self.path, f['file'])
            f_mask = os.path.join(self.path, f['mask'])
            if (not os.path.exists(f_img)) or (not os.path.exists(f_mask)):
                print(f"{f_img} or {f_mask} missing; skip.")
                continue

            pose = np.array(f['extrinsic'], dtype=np.float32)  # (4,4)
            K    = np.array(f['intrinsic'], dtype=np.float32)  # (3,3)

            img_np  = _read_rgb(f_img)
            mask_np = cv2.imread(f_mask, cv2.IMREAD_COLOR)  # treat as color; we’ll take one channel

            img, K_upd, Tc, mask, _, _, img_ori = self.transforms(img_np, K, pose, mask_np, None)

            images.append(img)         # (H,W,3) float32 [0,1]
            masks.append(mask)         # (H,W,1) float32 [0,1]
            images_ori.append(img_ori) # (H',W',3) float32 [0,1]
            poses.append(Tc)
            intrinsic.append(K_upd)

        if len(images) == 0:
            raise RuntimeError(f"No valid images for frame_id={frame_id} in {self.path}")

        # Stack tensors on CPU (keeps memory modest)
        images     = torch.stack(images, dim=0)            # (N,H,W,3) float32
        masks      = torch.stack(masks,  dim=0)            # (N,H,W,1) float32
        images_ori = torch.stack(images_ori, dim=0)        # (N,H',W',3) float32

        # Concatenate RGB + mask to match original downstream expectation
        images = torch.cat([images, masks], dim=-1)        # (N,H,W,4)

        poses     = np.stack(poses, axis=0).astype(np.float32)
        intrinsic = np.stack(intrinsic, axis=0).astype(np.float32)

        return images, poses, intrinsic, images_ori

    def __len__(self):
        return len(self.frameids)

    def __getitem__(self, idx):
        fid = self.frameids[idx]
        images_t, poses_t, intrinsic_t, images_ori_t = self.read_frame(fid, cam_num=self.cam_num)
        print('** Finish data loading.', fid)
        return images_t, poses_t, intrinsic_t, images_ori_t

    # The rest (read_frame_and_append / load_data) can remain the same as your current file,
    # since the important change is that images are now float32 in [0,1].
    # If you want me to rewrite those as well, say the word, but this fixes the PSNR issue.
