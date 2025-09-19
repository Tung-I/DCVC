_base_ = '../default.py'
expname = 'image_rf'
ckptname = None
wandbprojectname = 'chair_image'
basedir = '/home/tungichen_umass_edu/DCVC/logs/nerf_chair'

data = dict(
	datadir='/work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair',
	dataset_type='blender',
 	ndc=False,
    inverse_y=False,         # usual for OpenGL-style cameras
    flip_x=False,
    flip_y=False,
	xyz_min = [-1.5,  -1.5, -1.5],
	xyz_max = [ 1.5,   1.5,  1.5],
	load2gpu_on_the_fly=True,
    test_frames = [20],
	factor = 1,
)
fine_model_and_render = dict(
	num_voxels=160**3,
	num_voxels_base=160**3,
	k0_type='TensoRFGrid',
	rgbnet_dim=36,
    rgbnet_width=128,
    mpi_depth=192,
	stepsize=1,
	fast_color_thres = 1.0/256.0/80,
    viewbase_pe = 2,
    dynamic_rgbnet = True,
)

_k = 1
fine_train = dict(
    ray_sampler='flatten',
	N_iters=40000,
	N_rand=5000,   
	tv_every=1,                   # count total variation loss every tv_every step
    tv_after=100,                   # count total variation loss from tv_from step
    tv_before=25000,                  # count total variation before the given number of iterations
    tv_dense_before=25000,            # count total variation densely before the given number of iterations
    weight_tv_density=1e-5,        # weight of total variation loss of density voxel grid
	weight_tv_k0=1e-4,
	weight_l1_loss=0.01,
	weight_distortion = 0.0,
	pg_scale=[2000*_k,3500*_k, 5000*_k, 6000*_k],
    pg_scale2=[6500*_k, 9000*_k, 11000*_k],
	maskout_iter = 1000000*_k,
    initialize_density = True,
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    save_every = 2000,          # save every save_every steps
    save_after = 10000,          # save after save_after steps
    val_every = 2000,           # validate every val_every steps
)

coarse_train = dict(
    N_iters=0,
)