DEVICES=0
CHECKPOINT_PATH="/path/to/safetensors"
OUTPUT_DIR="/path/to/output directory"
EVAL_DIR="/path/to/eval directory"

CUDA_VISIBLE_DEVICES=$DEVICES python inference.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR \
    --num_frames 81 \
    --eval_data_root $EVAL_DIR


