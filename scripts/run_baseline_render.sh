cd /work/pi_rsitaram_umass_edu/tungi/DCVC

# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 10 --strategy tiling
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 20 --strategy tiling
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 24 --strategy tiling
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 28 --strategy tiling
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 30 --strategy tiling

# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 10 --strategy separate
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 20 --strategy separate
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 24 --strategy separate
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 28 --strategy separate
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 30 --strategy separate

# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 10 --strategy grouped
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 20 --strategy grouped
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 24 --strategy grouped
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 28 --strategy grouped
# python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 30 --strategy grouped


python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 0 --strategy tiling --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 21 --strategy tiling --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 42 --strategy tiling --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 63 --strategy tiling --dcvc

python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 0 --strategy separate --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 21 --strategy separate --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 42 --strategy separate --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 63 --strategy separate --dcvc

python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 0 --strategy grouped --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 21 --strategy grouped --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 42 --strategy grouped --dcvc
python render.py --config  TeTriRF/configs/N3D/flame_steak.py --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19  --render_test --startframe 0 --numframe 20 --qp 63 --strategy grouped --dcvc



