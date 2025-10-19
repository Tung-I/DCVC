import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import ipdb
import PIL
from PIL import Image
import collections
import math
import torchvision.transforms as T
import tqdm
from multiprocessing import Pool, Manager, Process


def imread_using_pillow(image_path, flags=cv2.IMREAD_UNCHANGED):
    """
    Read an image using Pillow and convert it to a numpy array.
    """
    image = Image.open(image_path)

    if flags == cv2.IMREAD_UNCHANGED:
        if image.mode == 'RGBA':
            image_array = np.array(image)
            image_array = image_array[:, :, [2, 1, 0, 3]]
            return image_array
        elif image.mode == 'RGB':
            return np.array(image)[:, :, ::-1]
        else:
            return np.array(image)
    
    if flags == cv2.IMREAD_GRAYSCALE:
        return np.array(image.convert('L'))

    if image.mode == 'RGB':
        return np.array(image)[:, :, ::-1]
    else:
        return np.array(image)

class Image_Transforms(object):
    """
    Affine transformation and resizing for images based on camera intrinsics.
    """
    def __init__(self, size, interpolation=Image.BICUBIC, is_center = False,  isNHR = True):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.is_center = is_center
        self.isNHR = isNHR
        
    def __call__(self, img, Ks , Ts ,  mask = None, residual =None):
        K = Ks
        Tc = Ts
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.astype('uint8'), 'RGB')
        img_np = np.asarray(img)
        width, height = img.size

        translation = [0,0]
        ration = 1.0
        if self.is_center:
            translation = [width /2-K[0,2],height/2-K[1,2]]
            translation = list(translation)
            ration = 1.05
            
            if (self.size[1]/2)/(self.size[0]*ration  / height) - K[0,2] != translation[0] :
                ration = 1.2

            if not self.isNHR:
                ration = 1.0
            translation[1] = (self.size[0]/2)/(self.size[0]*ration  / height) - K[1,2]
            translation[0] = (self.size[1]/2)/(self.size[0]*ration  / height) - K[0,2]
            translation = tuple(translation)
        
        img = T.functional.affine(img, angle = 0, translate = translation, scale= 1,shear=0)
        img = T.functional.crop(img, 0, 0,  int(height/ration),int(height*self.size[1]/ration/self.size[0]) )

        img_ori = img
        img = T.functional.resize(img, self.size, self.interpolation)
        img = T.functional.to_tensor(img)
        img = img.permute(1,2,0)

        img_ori = T.functional.resize(img_ori, [1080,1920], self.interpolation)
        img_ori = T.functional.to_tensor(img_ori)
        img_ori = img_ori.permute(1,2,0)
        
        ROI = np.ones_like(img_np)*255.0
        ROI = Image.fromarray(np.uint8(ROI))
        ROI = T.functional.affine(ROI, angle = 0, translate = translation, scale= 1,shear=0)
        ROI = T.functional.crop(ROI, 0,0, int(height/ration),int(height*self.size[1]/ration/self.size[0]) )
        ROI = T.functional.resize(ROI, self.size, self.interpolation)
        ROI = T.functional.to_tensor(ROI)
        ROI = ROI[0:1,:,:]
        
        if mask is not None:
            mask = T.functional.affine(mask, angle = 0, translate = translation, scale= 1,shear=0)
            mask = T.functional.crop(mask, 0, 0,  int(height/ration),int(height*self.size[1]/ration/self.size[0]) )
            mask = T.functional.resize(mask, self.size, self.interpolation)
            mask = T.functional.to_tensor(mask)
            mask = mask.permute(1,2,0)
            mask = mask[:,:,0:1]

        if residual is not None:
            residual = T.functional.affine(residual, angle = 0, translate = translation, scale= 1,shear=0)
            residual = T.functional.crop(residual, 0, 0,  int(height/ration),int(height*self.size[1]/ration/self.size[0]) )
            residual = T.functional.resize(residual, self.size, self.interpolation)
            residual = T.functional.to_tensor(residual)
            residual = residual.permute(1,2,0)
            residual = residual[:,:,0:1]

        K[0,2] = K[0,2] + translation[0]
        K[1,2] = K[1,2] + translation[1]
        s = self.size[0] * ration / height
        K = K*s
        K[2,2] = 1   
 
        return img, K, Tc, mask, residual, ROI, img_ori
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


def process_frame(f, basedir, transforms):
    f_path = os.path.join(basedir, f['file'])
    f_path_mask = os.path.join(basedir, f['mask'])
    view_id = int(f['file'].split('_')[1].split('.')[0])
    frame_id = int(f['file'].split('/')[1])

    if not os.path.exists(f_path) or not os.path.exists(f_path_mask):
        print(f"{f_path} or {f_path_mask} doesn't exist.")
        return None
    
    pose = np.array(f['extrinsic'], dtype=np.float32)  # [4, 4]
    K = np.array(f['intrinsic'], dtype=np.float32)

    mask = cv2.imread(f_path_mask)
    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img, K, Tc, mask, _, ROI = transforms(image, K, pose, mask, None)

    return {'Tc': Tc, 'img': img, 'K': K, 'mask': mask}


def wrapper(args):
    your_class_instance, id, res_images, res_images_ori, res_poses, res_intrinsic, frame_ids = args
    your_class_instance.read_frame_and_append(id, res_images, res_images_ori, res_poses, res_intrinsic, frame_ids)

class NHR_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, frameids=[], test_views = [], tar_size=(720,960), cam_num = -1, isNHR=True):
        super().__init__()
        self.cam_num = cam_num
        self.frameids = frameids
        self.path = path
        self.transforms = Image_Transforms(tar_size, isNHR = isNHR)
        self.test_views = test_views
        self.isNHR = isNHR

    def read_frame(self,frame_id, cam_num = -1):
        transform_path = os.path.join(self.path, 'cams_%d.json' % frame_id)
        with open(transform_path, 'r') as f:
            transform = json.load(f)

        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file'])

        if cam_num<0:
            cameras = [i for i in range(len(frames))]
        else:
            cameras = torch.randperm(len(frames), device='cpu')
            if cam_num>0:
                cameras = cameras[:cam_num]

        poses = []
        images = []
        intrinsic =[]
        masks =[]
        images_ori = []

        for id in cameras:
            f = frames[id]
            f_path = os.path.join(self.path, f['file'])
            f_path_mask = os.path.join(self.path, f['mask'])

            if not os.path.exists(f_path):
                print(f_path, "doesn't exist.")
                continue
            if not os.path.exists(f_path_mask):
                print(f_path_mask, "doesn't exist.")
                continue
            
            pose = (np.array(f['extrinsic'], dtype=np.float32)) # [4, 4]
            K = np.array(f['intrinsic'], dtype=np.float32)

            mask = cv2.imread(f_path_mask)
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img, K, Tc, mask, _, ROI, img_ori = self.transforms(image, K, pose,mask, None)

            images_ori.append(img_ori)
            poses.append(Tc)
            images.append(img)
            intrinsic.append(K)
            masks.append(mask)


        poses = np.stack(poses, axis=0).astype(np.float32)
        intrinsic = np.stack(intrinsic, axis=0).astype(np.float32)

        images = torch.stack(images)
        images_ori = torch.stack(images_ori)
        masks = torch.stack(masks)
        
        
        images = torch.cat([images, masks],dim = -1).float()
        return images,poses,intrinsic, images_ori

    def __len__(self):
        return len(self.frameids)

    def __getitem__(self, idx):

        frame_id = self.frameids[idx]

        #while frame_id==15:
        #    frame_id = random.randint(0,100)

        images_t, poses_t, intrinsic_t, images_ori_t = self.read_frame(frame_id,cam_num = self.cam_num)
        print('** Finish data loading.', frame_id)

        return images_t, poses_t, intrinsic_t, images_ori_t

    def read_frame_and_append(self, id, res_images, res_images_ori, res_poses, res_intrinsic, frame_ids):
        images_t, poses_t, intrinsic_t, images_ori_t = self.read_frame(id, cam_num=-1)
        Ni = images_t.size(0)  # CHANGED: per-frame #views
        res_images.append(images_t)
        res_images_ori.append(images_ori_t)
        res_poses.append(poses_t)
        res_intrinsic.append(intrinsic_t)
        # CHANGED: was torch.ones(self.P, ...)*id (wrong if Ni != self.P)
        frame_ids.append(torch.full((Ni,), id, device='cpu', dtype=torch.long))


    def load_data(self, current_id, previous_ids, scale=1.0):
        scale = int(scale)

        images, poses, intrinsic, images_ori = self.read_frame(current_id, cam_num=-1)
        N_current = images.size(0)        # CHANGED: don't reuse self.P for others
        self.P = N_current                # (keep if other code relies on it)

        previous_ids.sort()

        # We'll also keep per-frame counts to fix i_split & test_views remap.
        prev_counts = []

        frame_ids = []
        res_images, res_images_ori = [], []
        res_poses, res_intrinsic = [], []

        # ------ previous frames ------
        for fid in previous_ids:
            images_t, poses_t, intrinsic_t, images_ori_t = self.read_frame(fid, cam_num=-1)
            Ni = images_t.size(0)         # CHANGED: per-frame #views
            prev_counts.append(Ni)

            res_images.append(images_t)
            res_images_ori.append(images_ori_t)
            res_poses.append(poses_t)
            res_intrinsic.append(intrinsic_t)
            frame_ids.append(torch.full((Ni,), fid, device='cpu', dtype=torch.long))  # CHANGED

        # ------ current frame ------
        res_images.append(images)
        res_images_ori.append(images_ori)
        res_poses.append(poses)
        res_intrinsic.append(intrinsic)
        frame_ids.append(torch.full((N_current,), current_id, device='cpu', dtype=torch.long))  # CHANGED

        # Concatenate
        res_images      = torch.cat(res_images,      dim=0)
        res_images_ori  = torch.cat(res_images_ori,  dim=0)
        res_poses       = np.concatenate(res_poses,       axis=0)
        res_intrinsic   = np.concatenate(res_intrinsic,   axis=0)
        frame_ids       = torch.cat(frame_ids).long()

        # ---------- test_views remap (no constant-N assumption) ----------
        # self.test_views is a list of *view indices per frame*. Build absolute indices.
        abs_test = []
        offset = 0
        # previous frames
        for Ni in prev_counts:
            for v in self.test_views:
                if v < Ni:
                    abs_test.append(offset + v)
            offset += Ni
        # current frame
        for v in self.test_views:
            if v < N_current:
                abs_test.append(offset + v)
        self.test_views = abs_test

        # # ---------- i_split construction ----------
        # (train/val/test) + [replay, current] without assuming constant views
        if len(self.test_views) == 0:
            i_split = [np.arange(0, len(res_poses)) for _ in range(3)]
        else:
            train_idx = [i for i in np.arange(0, len(res_poses)) if i not in self.test_views]
            i_split = [train_idx, self.test_views, self.test_views]
        
        # Replay range = sum of previous frame counts
        replay_len = sum(prev_counts)
        i_split.append(np.arange(0, replay_len))                       # replay data
        i_split.append(np.arange(replay_len, replay_len + N_current))  # current data

        # honor scale on training split only
        i_split[0] = i_split[0][::scale]

        # The third return here was already 'res_poses' in your code; keep your signature.
        res_images_np = res_images.detach().cpu().numpy().astype(np.float32)
        return (res_images_np, res_poses, res_poses,
                [res_images.shape[1], res_images.shape[2], intrinsic[0, 0, 0]],
                res_intrinsic, i_split, frame_ids)   # NOTE: frame_ids now correct





