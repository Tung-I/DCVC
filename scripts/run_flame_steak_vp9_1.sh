cd ..

python train_codec_nerf_video.py \
    --config configs/dynerf_flame_steak/vp9_qp28.py \
    --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 

python train_codec_nerf_video.py \
    --config configs/dynerf_flame_steak/vp9_qp32.py \
    --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

python train_codec_nerf_video.py \
    --config configs/dynerf_flame_steak/vp9_qp44.py \
    --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19