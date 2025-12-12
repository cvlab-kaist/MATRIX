data_type=$3

CUDA_VISIBLE_DEVICES=$1 python detect_object.py \
    --data_dir $2/EVAL_DATA/$data_type