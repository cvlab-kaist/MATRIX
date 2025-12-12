data_directory=$1
model_ids=$2  # Comma-separated list: e.g., "abl_final,cogvideox_2b,cogvideox_5b"
data_types=$3  # Comma-separated list: e.g., "easy,easy_v2,sampled"

# Convert comma-separated strings to arrays
IFS=',' read -ra MODEL_ARRAY <<< "$model_ids"
IFS=',' read -ra DATA_ARRAY <<< "$data_types"

# Process all combinations of model_id and data_type
for model_id in "${MODEL_ARRAY[@]}"; do
    for data_type in "${DATA_ARRAY[@]}"; do
        echo "Processing: model=$model_id, data_type=$data_type"
        
        json_file="$data_directory/SCORE_RESULTS/question_verification_results/$model_id/$data_type/question_verification.json"
        output_file="$data_directory/SCORE_RESULTS/question_verification_results/$model_id/$data_type/video_scores.json"
        
        # Check if input file exists
        if [ ! -f "$json_file" ]; then
            echo "Warning: Input file not found: $json_file, skipping..."
            continue
        fi
        
        # Calculate scores for this combination
        python calculate_question_verification_scores.py \
            --json-file "$json_file" \
            --output-file "$output_file"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed: $model_id/$data_type"
        else
            echo "✗ Error processing: $model_id/$data_type"
        fi
    done
done

# Reorganize all video scores after processing all combinations
echo "Reorganizing all video scores..."
python reorganize_video_scores.py \
    --base-dir "$data_directory/SCORE_RESULTS/question_verification_results" \
    --output-file "$data_directory/SCORE_RESULTS/video_scores_summary.json"

echo "All processing complete!"