python eval_dcvc_ckpt_compress.py   \
    --src_ckpt logs/nerf_synthetic/dcvc_qp24_flatten_flatten_tv/fine_last_0.tar  \
    --plane_packing_mode flatten --grid_packing_mode flatten --qp 24 --save_json

python render.py --config  configs/blender_lego/image.py  \
    --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp24_flatten_flatten_tv/dcvc_qp24

python eval_dcvc_ckpt_compress.py   \
    --src_ckpt logs/nerf_synthetic/dcvc_qp36_flatten_flatten_tv/fine_last_0.tar  \
    --plane_packing_mode flatten --grid_packing_mode flatten --qp 36 --save_json

python render.py --config  configs/blender_lego/image.py  \
    --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp36_flatten_flatten_tv/dcvc_qp36

python eval_dcvc_ckpt_compress.py   \
    --src_ckpt logs/nerf_synthetic/dcvc_qp48_flatten_flatten_tv/fine_last_0.tar  \
    --plane_packing_mode flatten --grid_packing_mode flatten --qp 48 --save_json

python render.py --config  configs/blender_lego/image.py  \
    --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp48_flatten_flatten_tv/dcvc_qp48


python eval_dcvc_ckpt_compress.py   \
    --src_ckpt logs/nerf_synthetic/dcvc_qp24_flat4_flatten_tv/fine_last_0.tar  \
    --plane_packing_mode flat4 --grid_packing_mode flatten --qp 24 --save_json

python render.py --config  configs/blender_lego/image.py  \
    --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp24_flat4_flatten_tv/dcvc_qp24

python eval_dcvc_ckpt_compress.py   \
    --src_ckpt logs/nerf_synthetic/dcvc_qp36_flat4_flatten_tv/fine_last_0.tar  \
    --plane_packing_mode flat4 --grid_packing_mode flatten --qp 36 --save_json

python render.py --config  configs/blender_lego/image.py  \
    --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp36_flat4_flatten_tv/dcvc_qp36

python eval_dcvc_ckpt_compress.py   \
    --src_ckpt logs/nerf_synthetic/dcvc_qp48_flat4_flatten_tv/fine_last_0.tar  \
    --plane_packing_mode flat4 --grid_packing_mode flatten --qp 48 --save_json

python render.py --config  configs/blender_lego/image.py  \
    --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp48_flat4_flatten_tv/dcvc_qp48



# python eval_dcvc_ckpt_compress.py   \
#     --src_ckpt logs/nerf_synthetic/dcvc_qp24_flatten_flatten/fine_last_0.tar  \
#     --plane_packing_mode flatten --grid_packing_mode flatten --qp 24 --save_json

# python render.py --config  configs/blender_lego/image.py  \
#     --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp24_flatten_flatten/dcvc_qp24

# python eval_dcvc_ckpt_compress.py   \
#     --src_ckpt logs/nerf_synthetic/dcvc_qp36_flatten_flatten/fine_last_0.tar  \
#     --plane_packing_mode flatten --grid_packing_mode flatten --qp 36 --save_json

# python render.py --config  configs/blender_lego/image.py  \
#     --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp36_flatten_flatten/dcvc_qp36

# python eval_dcvc_ckpt_compress.py   \
#     --src_ckpt logs/nerf_synthetic/dcvc_qp48_flatten_flatten/fine_last_0.tar  \
#     --plane_packing_mode flatten --grid_packing_mode flatten --qp 48 --save_json

# python render.py --config  configs/blender_lego/image.py  \
#     --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp48_flatten_flatten/dcvc_qp48


# python eval_dcvc_ckpt_compress.py   \
#     --src_ckpt logs/nerf_synthetic/dcvc_qp24_flat4_flatten/fine_last_0.tar  \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 24 --save_json

# python render.py --config  configs/blender_lego/image.py  \
#     --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp24_flat4_flatten/dcvc_qp24

# python eval_dcvc_ckpt_compress.py   \
#     --src_ckpt logs/nerf_synthetic/dcvc_qp36_flat4_flatten/fine_last_0.tar  \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 36 --save_json

# python render.py --config  configs/blender_lego/image.py  \
#     --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp36_flat4_flatten/dcvc_qp36

# python eval_dcvc_ckpt_compress.py   \
#     --src_ckpt logs/nerf_synthetic/dcvc_qp48_flat4_flatten/fine_last_0.tar  \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 48 --save_json

# python render.py --config  configs/blender_lego/image.py  \
#     --render_test --frame_ids 0  --ckpt_dir logs/nerf_synthetic/dcvc_qp48_flat4_flatten/dcvc_qp48



# python eval_plane_packer.py     --logdir logs/nerf_synthetic/jpeg_qp20_flatten_flatten \
#             --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global

# python eval_plane_packer.py     --logdir logs/nerf_synthetic/jpeg_qp40_flatten_flatten \
#             --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global

# python eval_plane_packer.py     --logdir logs/nerf_synthetic/jpeg_qp60_flatten_flatten \
#             --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/jpeg_qp20_flatten_flatten \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flatten --grid_packing_mode flatten --qp 20

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/jpeg_qp40_flatten_flatten \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flatten --grid_packing_mode flatten --qp 40

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/jpeg_qp60_flatten_flatten \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flatten --grid_packing_mode flatten --qp 60

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/jpeg_qp20_flatten_flatten \
#     --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global \
#     --qp 20 --codec jpeg

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/jpeg_qp40_flatten_flatten \
#     --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global \
#     --qp 40 --codec jpeg

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/jpeg_qp60_flatten_flatten \
#     --numframe 1 --plane_packing_mode flatten --grid_packing_mode flatten --qmode global \
#     --qp 60 --codec jpeg

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/jpeg_qp20_flatten_flatten/planeimg_00_00_flatten_flatten_global_jpeg_qp20

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/jpeg_qp40_flatten_flatten/planeimg_00_00_flatten_flatten_global_jpeg_qp40

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/jpeg_qp60_flatten_flatten/planeimg_00_00_flatten_flatten_global_jpeg_qp60



# python eval_plane_packer.py     --logdir logs/nerf_synthetic/jpeg_qp20_flat4_flatten \
#             --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global

# python eval_plane_packer.py     --logdir logs/nerf_synthetic/jpeg_qp40_flat4_flatten \
#             --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global

# python eval_plane_packer.py     --logdir logs/nerf_synthetic/jpeg_qp60_flat4_flatten \
#             --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/jpeg_qp20_flat4_flatten \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 20

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/jpeg_qp40_flat4_flatten \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 40

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/jpeg_qp60_flat4_flatten \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 60

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/jpeg_qp20_flat4_flatten \
#     --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global \
#     --qp 20 --codec jpeg

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/jpeg_qp40_flat4_flatten \
#     --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global \
#     --qp 40 --codec jpeg

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/jpeg_qp60_flat4_flatten \
#     --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global \
#     --qp 60 --codec jpeg

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/jpeg_qp20_flat4_flatten/planeimg_00_00_flat4_flatten_global_jpeg_qp20

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/jpeg_qp40_flat4_flatten/planeimg_00_00_flat4_flatten_global_jpeg_qp40

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/jpeg_qp60_flat4_flatten/planeimg_00_00_flat4_flatten_global_jpeg_qp60






# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/lego_image \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 20

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/lego_image \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 40

# python eval_jpeg_triplane_image_compress.py \
#     --logdir logs/nerf_synthetic/lego_image \
#     --startframe 0 --numframe 1 --qmode global \
#     --plane_packing_mode flat4 --grid_packing_mode flatten --qp 60

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/lego_image \
#     --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global \
#     --qp 20 --codec jpeg

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/lego_image \
#     --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global \
#     --qp 40 --codec jpeg

# python eval_plane_unpacker.py  --root_dir logs/nerf_synthetic/lego_image \
#     --numframe 1 --plane_packing_mode flat4 --grid_packing_mode flatten --qmode global \
#     --qp 60 --codec jpeg

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/lego_image/planeimg_00_00_flat4_flatten_global_jpeg_qp20

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/lego_image/planeimg_00_00_flat4_flatten_global_jpeg_qp40

# python render.py --config  configs/blender_lego/image.py  \
#         --render_test --frame_ids 0 \
#         --ckpt_dir logs/nerf_synthetic/lego_image/planeimg_00_00_flat4_flatten_global_jpeg_qp60