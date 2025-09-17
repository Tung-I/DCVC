_base_ = '../default.py'
expname = 'dcvc_qp36_gradbpp_sandwich'
# ckptname = 'sear_steak_image'
ckptname = 'dcvc_qp36_gradbpp'
wandbprojectname = 'sear_steak_image'
basedir = '/home/tungichen_umass_edu/DCVC/logs/dynerf_sear_steak'

data = dict(
	datadir='/home/tungichen_umass_edu/DCVC/data/n3d/flame_steak/llff/',
	dataset_type='llff',
 	ndc=True,
	xyz_min = [-1.4,  -1.4, -0.6],
	xyz_max = [ 1.4,   1.4,  0.6],
	load2gpu_on_the_fly=True,
    test_frames = [0],
	factor = 3,
)
fine_model_and_render = dict(
	num_voxels=210**3,
	num_voxels_base=210**3,
	k0_type='PlaneGrid',
	rgbnet_dim=36,
    rgbnet_width=128,
    mpi_depth=280,
	stepsize=1,
	fast_color_thres = 1.0/256.0/80,
    viewbase_pe = 2,
    dynamic_rgbnet = True,
)

codec = dict(
    name = 'DCVCImageCodec',
    ckpt_path = '/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar',
    train_mode='ste',
    convert_ycbcr=True,
    freeze_dcvc=True,
    dcvc_qp = 36,
    quant_mode = "global",
    global_range = (-20.0, 20.0),
    packing_mode = None,
    use_amp=True,
    quality=None,
    codec_refresh_k = 32,
    refresh_trigger_eps = 0.0,  # e.g., 0.05 to refresh early if planes drift >5% L2
    gradbpp = True,  # enable entropy estimation for bpp loss
    bpp_refresh_k = 16,   # refresh bpp estimation every k steps
    use_sandwich = True,  # enable sandwich autoencoder
    use_linearpack_per_axis = True,  # use linear packing per axis
    lp_mid_dw=False,               # optional 3x3 DW in encoder/decoder
    unet_pre_base = 32,             # UNet width
    unet_post_base = 32,
    mlp_layers = 2,
    in_channels = 12,  # Number of input channels for the DCVC codec
    eps_pre = 1e-3,
    eps_post = 1e-3,
    weight_sandwich_id = 1e2,
)

_k = 1
fine_train = dict(
    ray_sampler='flatten',
	N_iters=30000,
	N_rand=5000,   
	tv_every=1,                   # count total variation loss every tv_every step
    tv_after=1e6,                   # count total variation loss from tv_from step
    tv_before=-1,                  # count total variation before the given number of iterations
    tv_dense_before=1e6,            # count total variation densely before the given number of iterations
    weight_tv_density=0,        # weight of total variation loss of density voxel grid
	weight_tv_k0=0,
	weight_l1_loss=0,
	weight_distortion = 0,
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
    sandwich_id_every  = 1,
    lrate_sandwich = 1e-4,
    sandwich_warmup_steps = 5000
)

coarse_train = dict(
    N_iters=0,
)