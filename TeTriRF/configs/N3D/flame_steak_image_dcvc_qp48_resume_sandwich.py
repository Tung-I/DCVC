_base_ = '../default.py'
expname = 'flame_steak_image_dcvc_qp48_resume_sandwich'
ckptname = 'flame_steak_image'
wandbprojectname = 'image_dcvc_flame_steak'
basedir = '/home/tungichen_umass_edu/DCVC/logs/out_triplane'

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
    unet_pre_base = 32,             # UNet width
    unet_post_base = 32,
    sandwich = True,  # use sandwich codec
    convert_ycbcr=True,
    freeze_dcvc_enc=True,
    freeze_dcvc_dec=True,
)

_k = 1
fine_train = dict(
    ray_sampler='flatten',
	N_iters=60000,
	N_rand=5000,   
	tv_every=1000000,                   # count total variation loss every tv_every step
    tv_after=1000000,                   # count total variation loss from tv_from step
    tv_before=-1,                  # count total variation before the given number of iterations
    tv_dense_before=-1,            # count total variation densely before the given number of iterations
    weight_tv_density=1e-5,        # weight of total variation loss of density voxel grid
	weight_tv_k0=1e-4,
	weight_l1_loss=0.01,
	weight_distortion = 0.0015,
	pg_scale=[],
    pg_scale2=[],
	maskout_iter = 1000000*_k,
    initialize_density = True,
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lambda_bpp = 0.01,          # weight of bpp loss
	save_every = 2000,          # save every save_every steps
    save_after = 10000,          # save after save_after steps
    vis_every = 500,          # visualize every vis_every steps
    lambda_min = 1,
    lambda_max = 768,
    qp_pool = [0, 16, 32, 48, 63],
    dcvc_qp = 48,
    lrate_sandwich = 1e-2, 
)

coarse_train = dict(
    N_iters=0,
)