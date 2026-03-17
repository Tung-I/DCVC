cd ..

python train_codec_nerf_video.py --config configs/dynerf_coffee_martini/av1_qp32.py --frame_ids 0 1 2 3 4 5 6 7 8 9
python train_codec_nerf_video.py --config configs/dynerf_coffee_martini/av1_qp44.py --frame_ids 0 1 2 3 4 5 6 7 8 9