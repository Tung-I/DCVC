#!/usr/bin/env bash

scene_name="flame_steak"
logdir="/work/pi_rsitaram_umass_edu/tungi/DCVC/logs/out_triplane/${scene_name}"
numframe=20

# ——— parameter lists ———
# strategies=(tiling separate grouped correlation flatfour)
# qmodes=(per_channel global)
strategies=(separate grouped correlation)
qmodes=(per_channel)
qps=(10 18 23 28 33)

cd ..

# # pack
# for strategy in "${strategies[@]}"; do
#   for qmode in "${qmodes[@]}"; do
#     for qp in "${qps[@]}"; do
#       python triplane_packer.py \
#         --logdir "${logdir}" \
#         --numframe "${numframe}" \
#         --strategy "${strategy}" \
#         --qmode "${qmode}"
#     done
#   done
# done

# generate ffmpeg scripts
for strategy in "${strategies[@]}"; do
  for qmode in "${qmodes[@]}"; do
    for qp in "${qps[@]}"; do
      python new_gen_ffmpeg.py \
        --logdir "${logdir}" \
        --numframe "${numframe}" \
        --strategy "${strategy}" \
        --qmode "${qmode}" \
        --qp "${qp}"
    done
  done
done

# run the generated ffmpeg scripts
for strategy in "${strategies[@]}"; do
  for qmode in "${qmodes[@]}"; do
    for qp in "${qps[@]}"; do
      (
        cd "${logdir}/planeimg_00_19_${strategy}_${qmode}" \
          && ./ffmpeg_qp${qp}_${strategy}_${qmode}.sh
      )
    done
  done
done

# go back up
cd "${logdir}"
cd ../../..

# unpack
for strategy in "${strategies[@]}"; do
  for qmode in "${qmodes[@]}"; do
    for qp in "${qps[@]}"; do
      python triplane_unpacker.py \
        --logdir "${logdir}" \
        --numframe "${numframe}" \
        --qp "${qp}" \
        --strategy "${strategy}" \
        --qmode "${qmode}"
    done
  done
done

# render
for strategy in "${strategies[@]}"; do
  for qmode in "${qmodes[@]}"; do
    for qp in "${qps[@]}"; do
      python render.py \
        --config TeTriRF/configs/N3D/${scene_name}.py \
        --frame_ids {0..19} \
        --render_test \
        --startframe 0 \
        --numframe "${numframe}" \
        --qp "${qp}" \
        --strategy "${strategy}" \
        --qmode "${qmode}"
    done
  done
done
