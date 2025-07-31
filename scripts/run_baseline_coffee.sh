
#!/usr/bin/env bash

scene_name="coffee_martini"
logdir="/work/pi_rsitaram_umass_edu/tungi/DCVC/logs/out_triplane/${scene_name}"
numframe=20

cd ..
for strategy in tiling separate grouped correlation; do
    for qmode in per_channel global; do
        for qp in 10 18 23 28 33; do
            python triplane_packer.py \
            --logdir ${logdir} \
            --numframe ${numframe} \
            --strategy ${strategy} --qmode ${qmode}
        done
    done
done


for strategy in tiling separate grouped correlation; do
    for qmode in per_channel global; do
        for qp in 10 18 23 28 33; do
            python new_gen_ffmpeg.py \
            --logdir "$logdir" \
            --numframe "$numframe" \
            --strategy "$strategy" \
            --qmode "$qmode" \
            --qp "$qp"
        done
    done
done


for strategy in tiling separate grouped correlation; do
    for qmode in per_channel global; do
        for qp in 10 18 23 28 33; do
            cd ${logdir}/planeimg_00_19_${strategy}_${qmode}
            ./ffmpeg_qp${qp}_${strategy}_${qmode}.sh
        done
    done
done

cd ${logdir}
cd ../../..

for strategy in tiling separate grouped correlation; do
    for qmode in per_channel global; do

        for qp in 10 18 23 28 33; do
            python triplane_unpacker.py \
            --logdir ${logdir} \
            --numframe ${numframe} --qp ${qp} \
            --strategy ${strategy} --qmode ${qmode}
        done
    done
done

for strategy in tiling separate grouped correlation; do
    for qmode in per_channel global; do
        for qp in 10 18 23 28 33; do
            python render.py \
                --config  TeTriRF/configs/N3D/${scene_name}.py \
                --frame_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
                --render_test --startframe 0 --numframe 20 \
                --qp ${qp} --strategy ${strategy} --qmode ${qmode}
        done
    done
done




