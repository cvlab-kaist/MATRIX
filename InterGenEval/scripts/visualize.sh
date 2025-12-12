data_directory=$1
model_id=$2
data_type=$3

python visualize.py \
    --data_dir $data_directory/EVAL_DATA/$data_type \
    --eval_dir $data_directory/EVAL_RESULTS/$model_id/$data_type \
    --video_dir $data_directory/OUTPUTS/$model_id/$data_type \
