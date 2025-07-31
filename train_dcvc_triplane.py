import os, copy, time, random, argparse
from tqdm import tqdm, trange
from mmengine.config import Config
import imageio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_efficient_distloss import flatten_eff_distloss

from TeTriRF.lib import utils, dvgo, dmpigo, dvgo_video, dcvc_dvgo_video
from TeTriRF.lib.dcvc_dvgo_video import DCVC_DVGO_Video
from TeTriRF.lib.load_data import load_data


os.environ['CUDA_VISIBLE_DEVICES']='0'

def seed_everything():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--training_mode", type=int, default=0)
    parser.add_argument('--frame_ids', nargs='+', type=int, help='a list of ID')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False, 
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    # Optional downsampling for preview speed
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs, depths, bgmaps = [], [], []
    psnrs, ssims, lpips_alex, lpips_vgg = [], [], [], []

    for i, c2w in enumerate(tqdm(render_poses)):
        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)

        # Generate rays for this view
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']

        # Flatten and chunk rays to avoid OOM
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,  **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]

        # Reassemble full image
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

        # Evaluate metrics if GT available and not downsampled
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    res_psnr = None
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        res_psnr = np.mean(psnrs)
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    # Optional flips/rotations for video
    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)
    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    # Dump images to disk
    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.jpg'.format(i))
            imageio.imwrite(filename, rgb8)


            rgb8 = utils.to8b(1 - depths[i] / np.max(depths[i]))
            #rgb8 = utils.to8b(1 - depths[i] / 5)
            filename = os.path.join(savedir, '{:03d}_depth.jpg'.format(i))
            if rgb8.shape[-1]<3:
                rgb8 = np.repeat(rgb8, 3, axis=-1)
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps, res_psnr

def load_everything(args, cfg):
    data_dict = load_data(cfg.data)
    kept_keys = {'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
                'i_train', 'i_val', 'i_test', 'irregular_shape',
                'poses', 'render_poses', 'images', 'frame_ids','masks'}
    
    # # Debug
    # images, poses, render_poses = data_dict['images'], data_dict['poses'], data_dict['render_poses']
    # print("images.shape:", images.shape)    # should be (P, H, W, 3)
    # print("poses.shape: ", poses.shape)     # should be (P, 3, 4)
    # print("render_poses.shape:", render_poses.shape)  # should be (N, 3, 4)
    # print(poses[0])
    # print(render_poses[0])
    # raise Exception("Debugging")

    for k in list(data_dict.keys()):
        if k not in kept_keys:  # remove useless field
            data_dict.pop(k)

    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, frameid=0, coarse_ckpt_path=None):
    """
    Train the multi-frame video extension of DirectVoxGO over all specified frames.

    Args:
        args, cfg, cfg_model, cfg_train: as above, but cfg_model here is fine_model_and_render
        xyz_min, xyz_max: fine-stage bounding box
        data_dict: full scene data including masks and frame_ids
        stage: 'fine'
        frameid: unused here (multi-frame)
        coarse_ckpt_path: path to coarse mask cache

    Returns:
        None (saves final checkpoints via model.save_checkpoints())
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # shift bbox if needed
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, frame_ids,masks = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'frame_ids','masks'
        ]
    ]
   
    frame_ids = frame_ids.cpu()
    unique_frame_ids = torch.unique(frame_ids, sorted=True).cpu().numpy().tolist()

    # initialize video model and load any existing checkpoints
    # model = dvgo_video.DirectVoxGO_Video(frameids=unique_frame_ids,xyz_min=xyz_min,xyz_max=xyz_max,cfg=cfg)
    model = dcvc_dvgo_video.DCVC_DVGO_Video(frameids=unique_frame_ids,xyz_min=xyz_min,xyz_max=xyz_max,cfg=cfg)
    ret = model.load_checkpoints()
    if not cfg.fine_model_and_render.dynamic_rgbnet and args.training_mode>0:
        cfg.fine_train.lrate_rgbnet = 0
    # model.set_fixedframe(ret)
    model = model.cuda()
    
    #create optimizer for model
    optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(model, cfg_train, global_step=0)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # gather multi-frame rays per unfixed frame
    def gather_training_rays(tmasks=None):
        rgb_tr_s = []
        rays_o_tr_s = []
        rays_d_tr_s = []
        viewdirs_tr_s = []
        imsz_s = []
        frame_id_s = []

        for id in unique_frame_ids:
            if id in model.fixed_frame:
                continue
            id_mask = (frame_ids==id)[i_train]
            t_train = np.array(i_train)[id_mask]
            rgb_tr_ori = images[t_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

            pmasks = None
            if tmasks is not None:
                pmasks = torch.from_numpy(tmasks[t_train]).to('cpu' if cfg.data.load2gpu_on_the_fly else device)

            if cfg_train.ray_sampler == 'in_maskcache':
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz,frame_ids_tr = dvgo.get_training_rays_multi_frame(
                        rgb_tr_ori=rgb_tr_ori,
                        train_poses=poses[t_train],
                        HW=HW[t_train], Ks=Ks[t_train],
                        ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                        frame_ids = frame_ids[t_train],
                        model=model.dvgos[str(id)], masks = pmasks, render_kwargs=render_kwargs)
                
            elif cfg_train.ray_sampler == 'flatten':
                rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz,frame_ids_tr = dvgo.get_training_rays_multi_frame(
                        rgb_tr_ori=rgb_tr_ori,
                        train_poses=poses[t_train],
                        HW=HW[t_train], Ks=Ks[t_train],
                        ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                        frame_ids = frame_ids[t_train],
                        model=model.dvgos[str(id)], masks = pmasks, render_kwargs=render_kwargs,
                        flatten = True)
            else:
                raise NotImplementedError

            rgb_tr_s += rgb_tr
            rays_o_tr_s+=rays_o_tr
            rays_d_tr_s+=rays_d_tr
            viewdirs_tr_s+=viewdirs_tr
            imsz_s+=imsz
            frame_id_s += frame_ids_tr
        
        print(rgb_tr_s[0].size())
        index_generator = dvgo.batch_indices_generator_MF(rgb_tr_s, cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr_s, rays_o_tr_s, rays_d_tr_s, viewdirs_tr_s, imsz_s, frame_id_s, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler = gather_training_rays()

    def my_collate_fn(batch):
            # Separate the data and labels from each sample in the batch
            assert len(batch) ==1
            item = batch[0]
            data1 = item[0] 
            data2 = item[1] 
            data3 = item[2] 
            data4 = item[3] 
            data5 = item[4]
            data6 = item[5]
            # Return the collated data and labels as a single batch
            return data1, data2, data3, data4, data5, data6

    # wrap into DataLoader for single-batch iteration
    ray_dataset = utils.Ray_Dataset(rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler)
    ray_dataloader = DataLoader(ray_dataset, batch_size=1,num_workers = 1, shuffle=False, collate_fn = my_collate_fn)
    raydata_iter = iter(ray_dataloader)
    

    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    psnr_res = []
    time_res = []
    psnr_test = []

    for global_step in trange(1, 1+cfg_train.N_iters):

        # 1) Periodically update the occupancy cache for acceleration
        if (global_step + 500) % 1000 == 0:
            ret = model.update_occupancy_cache()

        # 2) At the mask-out iteration, rebuild ray batches without masked frames 
        if  global_step in [ cfg_train.maskout_iter] and not cfg.data.ndc:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler = gather_training_rays(tmasks = None)
            ray_dataset = utils.Ray_Dataset(rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler)
            ray_dataloader = DataLoader(ray_dataset, batch_size=1,num_workers = 6, shuffle=False, collate_fn = my_collate_fn)
            raydata_iter = iter(ray_dataloader)

        # 3) For NDC scenes, also rebuild when hitting mask-out iter
        if  global_step in [ cfg_train.maskout_iter] and  cfg.data.ndc:
            # cfg.data.factor = 3
            data_dict = load_everything(args=args, cfg=cfg)

            # re-extract splits & images from data_dict..
            HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, frame_ids, masks = [
            data_dict[k] for k in [
                    'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'frame_ids','masks'
                ]
            ]
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler = gather_training_rays(tmasks = None)
            ray_dataset = utils.Ray_Dataset(rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, frame_id_tr, batch_index_sampler)
            ray_dataloader = DataLoader(ray_dataset, batch_size=1, num_workers = 6, shuffle=False, collate_fn = my_collate_fn)
            raydata_iter = iter(ray_dataloader)

        # 4) Progressive-growing: reduce voxel count at defined steps
        if args.training_mode != -1:
            if global_step in cfg_train.pg_scale:
                # compute remaining scales
                n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
                cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
                model.scale_volume_grid(cur_voxels)
                optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(model, cfg_train, global_step=0)

                for frameid in model.dvgos.keys():  # adjust act_shift for unfixed frames
                    if int(frameid) in model.fixed_frame:
                        continue
                    model.dvgos[frameid].act_shift -= cfg_train.decay_after_scale
                torch.cuda.empty_cache()

            # 5) Second-level PG schedule
            if global_step in cfg_train.pg_scale2:
                print('**Second Level PG****')
                n_rest_scales = len(cfg_train.pg_scale2)-cfg_train.pg_scale2.index(global_step)-1
                cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
                model.scale_volume_grid(cur_voxels)
                optimizer = utils.create_optimizer_or_freeze_model_dvgovideo(model, cfg_train, global_step=0)

                for frameid in model.dvgos.keys():
                    if int(frameid) in model.fixed_frame:
                        continue
                    model.dvgos[frameid].act_shift -= cfg_train.decay_after_scale/2.0
                torch.cuda.empty_cache()

        # 6) Sample a minibatch of rays from one frame   
        camera_id, sel_i = batch_index_sampler()
        sel_i = torch.from_numpy(sel_i)
        while sel_i.size(0) == 0:
            print('while loop in Ray_dataset')
            camera_id, sel_i = batch_index_sampler()
        
        frameids = frame_id_tr[camera_id][sel_i]
        target = rgb_tr[camera_id][sel_i]   
        rays_o = rays_o_tr[camera_id][sel_i]
        rays_d = rays_d_tr[camera_id][sel_i]
        viewdirs = viewdirs_tr[camera_id][sel_i]
        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # # Debug
        # print("frameids:", frameids)
        # print(frameids[0])
        # print("target.shape:", target.shape)
        # print("target:", target[0])
        # print("rays_o.shape:", rays_o.shape)
        # print("rays_o:", rays_o[0])
        # print("rays_d.shape:", rays_d.shape)
        # print("rays_d:", rays_d[0])
        # print("viewdirs.shape:", viewdirs.shape)
        # print("viewdirs:", viewdirs[0])
        
        ##########################################
        if global_step % cfg_train.comp_every == 0:
            model.compress()

            # print(f"bpp_loss: {model.last_bpp.item():.6f} at step {global_step}")
            # raise Exception("Debugging bpp_loss")
        ##########################################

        render_result = model(
            rays_o, rays_d, viewdirs, frame_ids = frameids,
            global_step=global_step,  mode='feat',
            **render_kwargs)

        # 8) Compute losses and backpropagate
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        ##############

        loss += cfg_train.lambda_bpp * model.last_bpp
        ##############
        psnr = utils.mse2psnr(loss.detach())

        ################ Unknown loss terms for now
        # optional entropy loss on final alpha
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        # optional near-clip density penalty
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        ######################

        # distortion loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            # # Debug
            # print("n_max:", n_max)
            # print("s:", s)
            # print("w:", w)
            # print("ray_id:", ray_id)
            # raise Exception

            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion

        # L1 loss on k0 color grid in fine stage
        if  stage=='fine':
            l1loss = model.compute_k0_l1_loss(frameids)
            loss += cfg.fine_train.weight_l1_loss*l1loss

        loss.backward()

        # 9) Total-variation regularization on voxel grids
        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before, frameids)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before, frameids)
        
        # 10) Optimizer step & LR decay
        optimizer.step()
        psnr_lst.append(psnr.item())
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # 11) Periodic logging of train PSNR & time
        if global_step%1000==0:
            psnr_res.append(np.mean(psnr_lst))
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            time_res.append(eps_time_str)

        if global_step%args.i_print==0 or global_step==1:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []


        if (global_step%(cfg_train.N_iters-1)==0 and stage!='coarse') :
            for frameid in model.dvgos.keys():
                if int(frameid) in model.fixed_frame:
                    continue
              
                render_viewpoints_kwargs = {
                    'model': model.dvgos[frameid],
                    'ndc': cfg.data.ndc,
                    'render_kwargs': {
                        'near': data_dict['near'],
                        'far': data_dict['far'],
                        'bg': 1 if cfg.data.white_bkgd else 0,
                        'stepsize': cfg.fine_model_and_render.stepsize,
                        'inverse_y': cfg.data.inverse_y,
                        'flip_x': cfg.data.flip_x,
                        'flip_y': cfg.data.flip_y,
                        'render_depth': True,
                        'shared_rgbnet':model.rgbnet,
                    },
                }
                testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{frameid}')
                os.makedirs(testsavedir, exist_ok=True)

                #找出data_dict['i_test']中属于frameid的索引 TODO
                frame_index = torch.nonzero(data_dict['frame_ids'] ==int(frameid)).squeeze(1).cpu().numpy()

                i_test = np.intersect1d(data_dict['i_test'], frame_index).copy()


                rgbs, depths, bgmaps,res_psnr = render_viewpoints(
                        render_poses=data_dict['poses'][i_test],
                        HW=data_dict['HW'][i_test],
                        Ks=data_dict['Ks'][i_test],
                        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in i_test],
                        savedir=testsavedir, dump_images=True,
                        eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                        **render_viewpoints_kwargs)
                print('iter:',global_step,'test_psnr:',res_psnr)


    if global_step != -1:
        model.save_checkpoints()
    

    

def train(args, cfg, data_dict):

    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))


    coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    
    xyz_min_fine =  torch.tensor(cfg.data.xyz_min)
    xyz_max_fine = torch.tensor(cfg.data.xyz_max)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':
    """
    Usage:
        python train_dcvc_triplane.py --config TeTriRF/configs/N3D/flame_steak_dcvc.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --training_mode 1
    """

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.frame_ids = args.frame_ids

    print("################################")
    print("--- Frame_ID:", cfg.data.frame_ids)
    print("--- training_mode:", args.training_mode)
    print("################################")

    if args.training_mode>0 and cfg.data.ndc:
        cfg.fine_train.lrate_rgbnet /=5.0
        cfg.fine_train.weight_tv_density = 0
        cfg.fine_train.weight_tv_k0 = 0   

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    print('Done')

