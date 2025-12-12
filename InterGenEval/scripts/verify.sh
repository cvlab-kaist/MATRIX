data_type=$2

python verify_object.py \
    --data_dir $1/EVAL_DATA/$data_type \
    --model "gpt-4o" \
    --pad_ratio 0.08