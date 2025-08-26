_base_ = '../default.py'
expname = 'flame_steak_image'
# ckptname = None
ckptname = 'flame_steak_image'
wandbprojectname = 'canerf_flame_steak_image'
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
)

_k = 1
fine_train = dict(
    ray_sampler='flatten',
	N_iters=32000,
	N_rand=5000,   
	tv_every=1,                   # count total variation loss every tv_every step
    tv_after=100,                   # count total variation loss from tv_from step
    tv_before=25000,                  # count total variation before the given number of iterations
    tv_dense_before=25000,            # count total variation densely before the given number of iterations
    weight_tv_density=1e-5,        # weight of total variation loss of density voxel grid
	weight_tv_k0=1e-4,
	weight_l1_loss=0.01,
	weight_distortion = 0.0015,
	pg_scale=[2000*_k,3500*_k, 5000*_k, 6000*_k],
    pg_scale2=[6500*_k, 9000*_k, 11000*_k],
	maskout_iter = 1000000*_k,
    initialize_density = True,
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    save_every = 2000,          # save every save_every steps
    save_after = 10000,          # save after save_after steps
)

coarse_train = dict(
    N_iters=0,
)