#!/usr/bin/env bash

# rootdir = "/work/pi_rsitaram_umass_edu/tungi/DCVC"
logdir="/work/pi_rsitaram_umass_edu/tungi/DCVC/logs/out_triplane/coffee_martini"
# logdir="/work/pi_rsitaram_umass_edu/tungi/DCVC/logs/out_triplane/flame_steak"
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