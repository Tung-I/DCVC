cd ..

# python train_codec_nerf_video.py \
#     --config configs/dynerf_flame_steak/av1_qp32.py \
#     --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 

# python train_codec_nerf_video.py \
#     --config configs/dynerf_flame_steak/av1_qp38.py \
#     --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 


# NHR
python train_codec_nerf_video_nhr.py \
--config configs/nhr_sport1/av1_qp50.py \
--frame_ids 0 1 2 3 4 5 6 7 8 9 --training_mode 1

python train_codec_nerf_video_nhr.py \
--config configs/nhr_sport1/av1_qp56.py \
--frame_ids 0 1 2 3 4 5 6 7 8 9 --training_mode 1
