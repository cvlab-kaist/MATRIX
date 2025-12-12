data_directory=$1
model_id=$2
data_type=$3

python question_generation.py \
    --metas_dir $data_directory/EVAL_RESULTS/$model_id/$data_type/eval_metas \
    --output $data_directory/EVAL_DATA/$data_type/questions \
    --model gpt-5