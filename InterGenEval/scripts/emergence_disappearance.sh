data_directory=$1
model_id=$2
data_type=$3

python emergence_disappearance.py \
    --metas_dir $data_directory/EVAL_RESULTS/$model_id/$data_type/eval_metas \
    --videos_dir $data_directory/EVAL_RESULTS/$model_id/$data_type/track_vis \
    --output $data_directory/SCORE_RESULTS/emergence_disappearance_results \
    --model gpt-5 \
    --stride 5 \
    --sample_type $data_type \
    --type $model_id