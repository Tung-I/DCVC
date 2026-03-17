cd ..

python train_codec_nerf_video.py --config configs/dynerf_cut_roasted_beef/av1_qp50.py --frame_ids 0 1 2 3 4 5 6 7 8 9
python train_codec_nerf_video.py --config configs/dynerf_cut_roasted_beef/av1_qp56.py --frame_ids 0 1 2 3 4 5 6 7 8 9
python train_codec_nerf_video.py --config configs/dynerf_cut_roasted_beef/av1_qp60.py --frame_ids 0 1 2 3 4 5 6 7 8 9