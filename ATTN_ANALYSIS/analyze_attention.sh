DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE python analyze_attention.py --attn_whole \
    --model "cogvideox_i2v_5b" \
    --height 480 \
    --width 720  --text_len 226 --num_frames 49 --output_dir "/path/to/output dir/" \
    --data_dir "/path/to/data dir/" --masks_dir "/path/to/mask dir/" \
    --prompt_path "/path/to/prompt dir/"