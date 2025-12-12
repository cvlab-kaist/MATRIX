data_directory=$5
data_type=$6
output_directory=$7

CUDA_VISIBLE_DEVICES=$1 python generate_video.py \
    --pid $2 \
    --n_pids $3 \
    --data_dir $data_directory/EVAL_DATA/$data_type \
    --cogvideox_version $4 \
    --output_dir $output_directory/$data_type