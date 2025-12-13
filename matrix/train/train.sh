DEVICE=0
DATA_ROOT="/path/to/data"
META_CSV="/path/to/metadata.csv"
BASE_MODEL="Wan-AI/Wan2.1-I2V-14B-480P"
OUTPUT_DIR="/path/to/output"
CUDA_VISIBLE_DEVICES=$DEVICE python matrix/train/train.py \
    --dataset_base_path $DATA_ROOT \
    --dataset_metadata_path $META_CSV \
    --height 480 \
    --width 832 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "$BASE_MODEL:diffusion_pytorch_model*.safetensors,$BASE_MODEL:models_t5_umt5-xxl-enc-bf16.pth,$BASE_MODEL:Wan2.1_VAE.pth,$BASE_MODEL:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --learning_rate 1e-5 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path $OUTPUT_DIR \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 16 \
    --extra_inputs "input_image" \
    --trainable_models "dit,dit.patch_embedding,seg_head" \
    --v2t_layers "20,24" --sga --v2v_layers "6,8" --spa