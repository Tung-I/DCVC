cd /work/pi_rsitaram_umass_edu/tungi/DCVC

# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 10 --strategy tiling
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 20 --strategy tiling
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 24 --strategy tiling
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 28 --strategy tiling
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 30 --strategy tiling

# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 10 --strategy separate
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 20 --strategy separate
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 24 --strategy separate
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 28 --strategy separate
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 30 --strategy separate

# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 10 --strategy grouped
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 20 --strategy grouped
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 24 --strategy grouped
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 28 --strategy grouped
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 30 --strategy grouped

# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 0 --strategy tiling --dcvc
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 21 --strategy tiling --dcvc
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 42 --strategy tiling --dcvc
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 63 --strategy tiling --dcvc


# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 0 --strategy separate --dcvc
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 21 --strategy separate --dcvc
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 42 --strategy separate --dcvc
# python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 63 --strategy separate --dcvc


python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 0 --strategy grouped --dcvc
python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 21 --strategy grouped --dcvc
python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 42 --strategy grouped --dcvc
python new_image2plane.py --logdir logs/out_triplane/flame_steak --numframe 20 --qp 63 --strategy grouped --dcvc
