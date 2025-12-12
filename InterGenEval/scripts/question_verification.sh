data_directory=$1
model_id=$2
data_type=$3

python question_verification.py \
    --metas_dir $data_directory/EVAL_DATA/$data_type/questions \
    --videos_dir $data_directory/EVAL_RESULTS/$model_id/$data_type/track_vis \
    --output $data_directory/SCORE_RESULTS/question_verification_results \
    --model gpt-5 \
    --stride 5 \
    --type $model_id \
    --sample_type $data_type