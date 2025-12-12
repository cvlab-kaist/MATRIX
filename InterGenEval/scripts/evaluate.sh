python evaluate.py \
      --metas_dir /data/chan/data/eval_metas \
      --videos_dir /data/chan/data/cogvideox_eval \
      --output /home/cvlab19/project/chan/results/data \
      --model gpt-5 \
      --type CogvideoX

python evaluate.py \
      --metas_dir /data/chan/data_v2/contact/eval_metas \
      --videos_dir /data/chan/data_v2/contact/cogvideox_eval \
      --output /home/cvlab19/project/chan/results/contact \
      --model gpt-5 \
      --type CogvideoX

python evaluate.py \
      --metas_dir /data/chan/data_v2/force/eval_metas \
      --videos_dir /data/chan/data_v2/force/cogvideox_eval \
      --output /home/cvlab19/project/chan/results/force \
      --model gpt-5 \
      --type CogvideoX

python evaluate.py \
      --metas_dir /data/chan/data_v2/manipulation/eval_metas \
      --videos_dir /data/chan/data_v2/manipulation/cogvideox_eval \
      --output /home/cvlab19/project/chan/results/manipulation \
      --model gpt-5 \
      --type CogvideoX