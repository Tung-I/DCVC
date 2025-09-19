_base_ = '../default.py'
expname = 'dcvc_qp24_flatten_flatten'
ckptname = 'lego_image'
wandbprojectname = 'lego_image'
basedir = '/home/tungichen_umass_edu/DCVC/logs/nerf_synthetic'

data = dict(
	datadir='/work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/lego',
	dataset_type='blender',
 	ndc=False,
    inverse_y=False,         # usual for OpenGL-style cameras
    flip_x=False,
    flip_y=False,
	xyz_min = [-1.5,  -1.5, -1.5],
	xyz_max = [ 1.5,   1.5,  1.5],
	load2gpu_on_the_fly=True,
    test_frames = [0],
	factor = 1,
)
fine_model_and_render = dict(
	num_voxels=210**3,
	num_voxels_base=210**3,
	k0_type='PlaneGrid',
	rgbnet_dim=36,
    rgbnet_width=128,
    mpi_depth=192,
	stepsize=1,
	fast_color_thres = 1.0/256.0/80,
    viewbase_pe = 2,
    dynamic_rgbnet = True,
)

codec = dict(
    name = 'DCVCImageCodec',
    ckpt_path = '/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar',
    train_mode='ste',
    unet_pre_base = 32,             # UNet width
    unet_post_base = 32,
    use_sandwich = False,  
    convert_ycbcr=True,
    freeze_dcvc=True,
    dcvc_qp=24,
    quant_mode = "global",
    global_range = (-20.0, 20.0),
    plane_packing_mode = "flatten",
    grid_packing_mode = "flatten",
    mlp_layers = 2,
    in_channels = 12,  # Number of input channels for the DCVC codec
    use_amp=True,
    codec_refresh_k = 32,
    refresh_trigger_eps = 0.0,  # e.g., 0.05 to refresh early if planes drift >5% L2
)

_k = 1
fine_train = dict(
    ray_sampler='flatten',
	N_iters=30000,
	N_rand=5000,   
	tv_every=1e6,                   # count total variation loss every tv_every step
    tv_after=1e6,                   # count total variation loss from tv_from step
    tv_before=-1,                  # count total variation before the given number of iterations
    tv_dense_before=-1,            # count total variation densely before the given number of iterations
    weight_tv_density=0,        # weight of total variation loss of density voxel grid
	weight_tv_k0=0,
	weight_l1_loss=None,
	weight_distortion = 0.0,
	pg_scale=[],
    pg_scale2=[],
	maskout_iter = 1000000*_k,
    initialize_density = True,
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lambda_bpp = 0.01,          # weight of bpp loss
    lambda_min = 1,
    lambda_max = 768,
	save_every = 2000,          # save every save_every steps
    save_after = 10000,          # save after save_after steps
    use_bpp = False,
)

coarse_train = dict(
    N_iters=0,
)