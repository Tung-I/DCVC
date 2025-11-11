_base_ = '../default.py'
expname = 'sport1_dcvc_qp48'
ckptname = 'sport1'
wandbprojectname = 'NHR'
basedir = '/home/tungichen_umass_edu/DCVC/logs/NHR'

data = dict(
	datadir='/work/pi_rsitaram_umass_edu/tungi/datasets/NHR/sport_1',   # scene data folder
	dataset_type='NHR',
	white_bkgd=True,
	xyz_min = [-0.1531104 , -0.99574334, -0.40296442],
	xyz_max = [0.20760746, 0.01441086, 0.49955044],
	test_frames =[5,41],
    height=480,
    width=640,
	inverse_y=True,
	load2gpu_on_the_fly=True,
)

fine_model_and_render = dict(
	num_voxels=120**3,
	num_voxels_base=120**3,
	k0_type='PlaneGrid',
	rgbnet_dim=36,
    rgbnet_width=128,
    mpi_depth=192,
	RGB_model = 'MLP',
	rgbnet_depth = 2,
	dynamic_rgbnet = True,
	viewbase_pe = 4,
	plane_scale = 3,
    stepsize=1,
)

codec = dict(
    name = 'DCVCVideoCodec',
    ckpt_path = None,
    train_mode='ste',
    unet_pre_base = 32,             # UNet width
    unet_post_base = 32,
    in_channels = 12,  
    convert_ycbcr=True,
    quant_mode = "absmax",
    global_range = (-20.0, 20.0),
    packing_mode = "flat4",
    use_amp=True,
    quality=None,
    codec_refresh_k = 32,
    refresh_trigger_eps = 0.05,  # e.g., 0.05 to refresh early if planes drift >5% L2
    gop = 10,
    fps = 30,
    pix_fmt = 'yuv444p',
    hevc_qp = None,
    vp9_qp = None,
    av1_qp = None,
    dcvc_qp = 48,
)


_k = 1
fine_train = dict(
    ray_sampler='flatten',
	N_iters=25000,
	N_rand=5000,   
	tv_every=1,                   # count total variation loss every tv_every step
    tv_after=10000,                   # count total variation loss from tv_from step
    tv_before=30000,                  # count total variation before the given number of iterations
    tv_dense_before=30000,            # count total variation densely before the given number of iterations
    weight_tv_density=5e-6,        # weight of total variation loss of density voxel grid
	weight_tv_k0=5e-5,
    weight_l1_loss=0,
	weight_distortion = 0,
	pg_scale=[],
    pg_scale2=[],
	maskout_iter = 1000000*_k,
    initialize_density = True,
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lambda_bpp = 0.0,          # weight of bpp loss
    lambda_min = 1,
    lambda_max = 768,
	save_every = 2000,          # save every save_every steps
    save_after = 10000,          # save after save_after steps
)

coarse_train = dict(
	N_iters = 0
)