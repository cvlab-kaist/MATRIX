data_directory=$2
model_id=$3
data_type=$4

CUDA_VISIBLE_DEVICES=$1 python track_object.py \
    --data_dir $data_directory/EVAL_DATA/$data_type \
    --eval_dir $data_directory/EVAL_RESULTS/$model_id/$data_type \
    --video_dir $data_directory/OUTPUTS/$model_id/$data_type