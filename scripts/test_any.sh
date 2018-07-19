set -x
set -e

python test_pixel_link_on_any_image.py \
            --checkpoint_path=$1 \
            --dataset_dir=$2 \
            --eval_image_width=640\
            --eval_image_height=360\
            --pixel_conf_threshold=0.5\
            --link_conf_threshold=0.5\
            --gpu_memory_fraction=-1
