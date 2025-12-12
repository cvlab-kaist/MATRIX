data_directory=$1

python calculate_combined_scores.py \
    --video-scores-file "$data_directory/SCORE_RESULTS/video_scores_summary.json" \
    --emergence-scores-file "$data_directory/SCORE_RESULTS/emergence_disappearance_scores_summary.json" \
    --output-file "$data_directory/SCORE_RESULTS/combined_scores_summary.json"