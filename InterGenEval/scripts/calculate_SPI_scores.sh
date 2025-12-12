data_directory=$1

python calculate_emergence_disappearance_scores.py \
    --base-dir "$data_directory/SCORE_RESULTS/emergence_disappearance_results" \
    --output-file "$data_directory/SCORE_RESULTS/emergence_disappearance_scores_summary.json"