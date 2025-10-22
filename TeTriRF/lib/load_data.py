import numpy as np
import os
from .load_llff import LLFF_Dataset
from .load_NHR import NHR_Dataset
from .load_blender import Blender_Dataset, load_blender_data
from .load_immersive import readColmapSceneInfoImmersive
from torch.utils.data import DataLoader
import ipdb
import tqdm
import torch

def load_data(args, split="train"):

    K, depths = None, None
    near_clip = None
    frame_ids = None
    masks = None

    if args.dataset_type == 'llff':
        args.frame_ids.sort()
        if not hasattr(args, 'spherify'):
            args.spherify = False

        dataset = LLFF_Dataset(args.datadir, factor = args.factor, frameids = args.frame_ids, test_views = args.test_frames, spherify = args.spherify)
        
        def my_collate_fn(batch):
            assert len(batch) ==1
            item = batch[0]
            data1 = item[0] 
            data2 = item[1] 
            data3 = item[2] 
            data4 = item[3] 
            return data1, data2, data3, data4
        
        train_dataloader = DataLoader(dataset, batch_size=1,num_workers = 12, shuffle=False, collate_fn = my_collate_fn)

        frame_ids = []
        res_images = []
        res_poses = []
        res_render_poses = []
   
        for i, data in enumerate(train_dataloader):
            images_t, poses_t, render_poses_t, i_test_t = data
            P = images_t.shape[0]  # number of views for one frame index
            res_images.append(images_t)
            res_poses.append(poses_t)
            res_render_poses.append(render_poses_t)
            frame_ids.append(torch.ones(P, device='cpu')*args.frame_ids[i])  # every block of P consecutive views has the same fid

        res_images = np.concatenate(res_images,axis=0)
        res_poses = np.concatenate(res_poses,axis=0)
        res_render_poses = np.concatenate(res_render_poses,axis=0)
        frame_ids = torch.cat(frame_ids).long()

        images = res_images
        render_poses = res_render_poses
        poses = res_poses

        hwf = res_poses[0,:3,-1]
        poses = res_poses[:,:3,:4]
        print('Loaded llff', res_images.shape, res_render_poses.shape, hwf, args.datadir)
        
        test_frames = []
        for i in args.test_frames:  # args.test_frames is [0]
            for j in range(i,int(res_images.shape[0]),P):
                test_frames.append(j)
        #i_val = i_test
        i_train = np.array([i for i in np.arange(int(res_images.shape[0])) if i not in test_frames])
        i_test = test_frames
        i_val = i_test

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        # else:
        #     near = np.ndarray.min(bds) * .9
        #     far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    #############################################
    elif args.dataset_type == 'blender':
        args.frame_ids.sort()
        if not hasattr(args, 'spherify'):
            args.spherify = False

        def _load_one_split(split_name):
            images, depths, poses35, bds, render_poses, i_holdout = load_blender_data(
                args.datadir,
                factor=args.factor, width=args.width, height=args.height,
                recenter=False, bd_factor=None, spherify=False,
                path_zflat=False, load_depths=False,
                frame_id=None, movie_render_kwargs=None, split=split_name,
            )
            # poses35: [N,3,5]; keep [N,3,4] + intrinsics
            hwf = poses35[0, :3, -1]
            poses = poses35[:, :3, :4]
            H, W, focal = int(hwf[0]), int(hwf[1]), float(hwf[2])
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ], dtype=np.float32)
            Ks = np.repeat(K[None], len(poses), axis=0)
            HW = np.array([[H, W]] * len(poses), dtype=np.int32)
            return images, poses, HW, Ks

        # Load all three splits (some scenes may have empty val; handle gracefully)
        splits = []
        for name in ["train", "val", "test"]:
            meta_path = os.path.join(args.datadir, f"transforms_{name}.json")
            if os.path.isfile(meta_path):
                splits.append(name)

        X = []   # images
        P = []   # poses
        HWs = [] # [N,2]
        Kss = [] # [N,3,3]
        fids = []# frame ids (repeat the user-provided ids)
        idx_train = []
        idx_val   = []
        idx_test  = []

        # Near/Far for Blender
        near, far = (0.0, 1.0) if args.ndc else (2.0, 6.0)

        base_ofs = 0
        # blender only support a single frame
        assert len(args.frame_ids) == 1
        for fid in args.frame_ids:
            for split_name in splits:
                imgs, poses, HW, Ks = _load_one_split(split_name)

                n = imgs.shape[0]
                X.append(imgs)
                P.append(poses)
                HWs.append(HW)
                Kss.append(Ks)
                fids.append(torch.full((n,), int(fid), dtype=torch.long))

                if split_name == "train":
                    idx_train.extend(range(base_ofs, base_ofs + n))
                elif split_name == "val":
                    idx_val.extend(range(base_ofs, base_ofs + n))
                elif split_name == "test":
                    idx_test.extend(range(base_ofs, base_ofs + n))

                base_ofs += n

        images = np.concatenate(X, axis=0)
        poses  = np.concatenate(P, axis=0)
        HW     = np.concatenate(HWs, axis=0)
        Ks     = np.concatenate(Kss, axis=0)
        frame_ids = torch.cat(fids, dim=0).cpu()

        # masks/depths are not used for Blender synthetic by default
        depths = None
        masks  = None
        irregular_shape = False
        near_clip = None
        render_poses = np.zeros((1,3,4), dtype=np.float32)  # unused placeholder

        # Cast intrinsics to match the rest of the code
        H0, W0 = HW[0]
        focal0 = Ks[0,0,0]
        hwf = [int(H0), int(W0), float(focal0)]

        # Finalize splits
        i_train = np.array(idx_train, dtype=np.int64)
        i_val   = np.array(idx_val,   dtype=np.int64)
        i_test  = np.array(idx_test,  dtype=np.int64)

        # print('Loaded blender',
        #   images.shape, poses.shape, HW.shape, Ks.shape, args.datadir)
        # print(f'#train={len(i_train)}  #val={len(i_val)}  #test={len(i_test)}')
        # raise Exception("Stop here for debug")

    ############################################
    elif args.dataset_type == 'immersive':

        scene = readColmapSceneInfoImmersive(
            path      = args.datadir,
            images    = None,
            eval      = False,
            duration  = len(args.frame_ids),
            testonly  = False
        )

        train_cams = scene.train_cameras
        test_cams  = scene.test_cameras
        cam_infos = test_cams + train_cams
        """
        train_cam_infos =  cam_infos[duration:]
        test_cam_infos = cam_infos[:duration]
        """

        images = np.stack([np.array(cam.image) for cam in cam_infos], axis=0)  
        poses  = np.stack([
            np.concatenate([cam.R, cam.T.reshape(3,1)], axis=1)
            for cam in cam_infos
        ], axis=0)  # (N_train, 3, 4)

        # 5) compute H, W, focal for all cameras (we assume they’re identical)
        H, W = cam_infos[0].height, cam_infos[0].width
        # recover focal from horizontal FOV: focal = W/(2*tan(FovX/2))
        focal = W / (2.0 * np.tan(cam_infos[0].FovX / 2.0))  # fov2focal
        hwf   = [H, W, focal]
    
        # 6) build K or Ks
        K = np.array([
            [focal,    0.0,   0.5 * W],
            [0.0,    focal,   0.5 * H],
            [0.0,      0.0,     1.0 ]
        ], dtype=np.float32)
        Ks = np.repeat(K[None], len(poses), axis=0)  # (N_train, 3, 3)

        # 7) near/far from any camera
        if args.ndc:
            near = 0.0
            far  = 1.0
        else:
            near = cam_infos[0].near
            far  = cam_infos[0].far

        # if near != 0.0 or far != 1.0:
        #     raise Exception(f"Warning: near {near} and far {far} from camera info do not match NDC convention (0.0, 1.0).")
            
        # 8) build frame_ids
        frame_ids = []
        for i in range(0, len(cam_infos), len(args.frame_ids)):
            frame_ids.append(torch.tensor(args.frame_ids))
        frame_ids = torch.cat(frame_ids).long()
        assert len(frame_ids) == images.shape[0], f"Frame ids length {len(frame_ids)} does not match number of cameras {len(cam_infos)}."

        render_poses=None # To-do: generate spherical poses for visualization; for now it does not matter

        i_test = np.arange(len(test_cams))
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if i not in i_test])
        i_val = i_test

        # 11) misc
        HW = np.array([im.shape[:2] for im in images])
        irregular_shape = False  # all same size
        depths = None           # no depths in immersive pipeline
        masks  = None

        return dict(
            hwf=hwf, HW=HW, Ks=Ks,
            near=near, far=far, near_clip=near_clip,
            i_train=i_train, i_val=i_val, i_test=i_test,
            poses=poses, render_poses=render_poses,
            images=images, depths=depths,
            irregular_shape=irregular_shape,
            frame_ids=frame_ids, masks=masks,
        )

    # elif args.dataset_type == 'NHR':
    #     args.frame_ids.sort()
    #     frame_id = args.frame_ids[-1]
    #     previous_frame_ids = args.frame_ids[:-1]
    #     tar_size = (args.height, args.width)
    #     isNHR = (args.width <= 960)

    #     dataset = NHR_Dataset(
    #         args.datadir,
    #         frameids=args.frame_ids,
    #         test_views=args.test_frames,
    #         tar_size=tar_size,
    #         isNHR=isNHR
    #     )

    #     # >>> Use the dataset's own loader (already patched to be frame-safe) <<<
    #     images, poses, render_poses, hwf, K, i_split, frame_ids = dataset.load_data(
    #         current_id=frame_id, previous_ids=previous_frame_ids, scale=1.0
    #     )
    #     print('Loaded NHR', images.shape, render_poses.shape, hwf, args.datadir,
    #         ' frame:', frame_id, ' previous:', previous_frame_ids)

    #     i_train, i_val, i_test, i_replay, i_current = i_split
    #     print('@@@@@@ training:', len(i_train), 'test:', len(i_test))

    #     # Near/far from current-frame camera centers
    #     near, far = inward_nearfar_heuristic(poses[i_current, :3, 3])
    #     near = near * 1.1
    #     far  = far  * 1.6

    #     # masks & alpha handling (same as before)
    #     assert images.shape[-1] in (3, 4)
    #     masks = images[..., -1:]
    #     if images.shape[-1] == 4:
    #         if args.white_bkgd:
    #             images = images[..., :3] * masks + (1. - masks)
    #         else:
    #             images = images[..., :3]


    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        frame_ids = frame_ids, masks= masks,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far

