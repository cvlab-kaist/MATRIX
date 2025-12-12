data_directory=$1

python calculate_model_averages.py \
    --base_dir $data_directory/SCORE_RESULTS \
    --output_file $data_directory/SCORE_RESULTS/model_averages_summary.json \
    --input_file $data_directory/SCORE_RESULTS/combined_scores_summary.json