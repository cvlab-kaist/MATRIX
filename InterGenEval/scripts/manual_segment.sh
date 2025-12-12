data_type=$3

CUDA_VISIBLE_DEVICES=$1 python manual_segment.py \
    --sam2_model_id facebook/sam2.1-hiera-large \
    --image_dir $2/EVAL_DATA/$data_type/images \
    --output_dir $2/EVAL_DATA/$data_type \
    --port 7861