#!/usr/bin/env python3
"""
Calculate combined scores: KISA_SPI, SGA_SPI, and IF_score for each video.
KISA_SPI = KISA_score * SPI_score
SGA_SPI = SGA_score * SPI_score  
IF_score = (KISA_SPI + SGA_SPI) / 2
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

def load_json_file(file_path):
    """Load JSON file and return data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def calculate_combined_scores(video_scores_file, emergence_scores_file):
    """Calculate combined scores from video_scores_summary.json and emergence_disappearance_scores_summary.json."""
    
    print("Loading video scores...")
    video_scores = load_json_file(video_scores_file)
    if not video_scores:
        print("Failed to load video scores!")
        return {}
    
    print("Loading emergence/disappearance scores...")
    emergence_scores = load_json_file(emergence_scores_file)
    if not emergence_scores:
        print("Failed to load emergence/disappearance scores!")
        return {}
    
    print("Calculating combined scores...")
    combined_results = {}
    
    # Process each model
    for model_name in video_scores:
        if model_name not in emergence_scores:
            print(f"Warning: Model {model_name} not found in emergence scores")
            continue
            
        combined_results[model_name] = {}
        
        # Process each sample type
        for sample_type in video_scores[model_name]:
            if sample_type not in emergence_scores[model_name]:
                print(f"Warning: Sample type {sample_type} not found in emergence scores for model {model_name}")
                continue
                
            combined_results[model_name][sample_type] = {}
            
            
            video_sample_data = video_scores[model_name][sample_type]
            emergence_sample_data = emergence_scores[model_name][sample_type]
            
            # Process each video
            for video_id in video_sample_data:
                if video_id not in emergence_sample_data:
                    print(f"Warning: Video {video_id} not found in emergence scores for {model_name}/{sample_type}")
                    continue
                
                # Get data
                video_data = video_sample_data[video_id]
                emergence_data = emergence_sample_data[video_id]
                
                kisa_score = video_data.get('KISA_score', 0.0)
                sga_score = video_data.get('SGA_score', 0.0)
                spi_score = emergence_data.get('SPI_score', 0.0)
                
                # Calculate combined scores
                kisa_spi = kisa_score * spi_score
                sga_spi = sga_score * spi_score
                if_score = (kisa_spi + sga_spi) / 2
            
                
                # Store results
                combined_results[model_name][sample_type][video_id] = {
                    'prompt': video_data.get('prompt', ''),
                    'KISA_score': kisa_score,
                    'SGA_score': sga_score,
                    'SPI_score': spi_score,
                    'KISA_SPI': kisa_spi,
                    'SGA_SPI': sga_spi,
                    'IF_score': if_score
                }
    
    return combined_results

def save_results(results, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("COMBINED SCORES SUMMARY")
    print("="*80)
    
    for model_name, model_data in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)
        
        for sample_type, sample_data in model_data.items():
            if not sample_data:
                print(f"  {sample_type}: No data")
                continue
                
            print(f"  {sample_type}: {len(sample_data)} videos")
            
            # Calculate average scores for this sample type
            kisa_spi_scores = [v['KISA_SPI'] for v in sample_data.values()]
            sga_spi_scores = [v['SGA_SPI'] for v in sample_data.values()]
            if_scores = [v['IF_score'] for v in sample_data.values()]
            
            if kisa_spi_scores:
                avg_kisa_spi = sum(kisa_spi_scores) / len(kisa_spi_scores)
                avg_sga_spi = sum(sga_spi_scores) / len(sga_spi_scores)
                avg_if = sum(if_scores) / len(if_scores)
                print(f"    Average KISA_SPI: {avg_kisa_spi:.4f}")
                print(f"    Average SGA_SPI: {avg_sga_spi:.4f}")
                print(f"    Average IF_score: {avg_if:.4f}")

def print_detailed_sample(results, model_name, sample_type, limit=5):
    """Print detailed sample of results for a specific model and sample type."""
    if model_name not in results or sample_type not in results[model_name]:
        print(f"No data found for {model_name}/{sample_type}")
        return
    
    sample_data = results[model_name][sample_type]
    print(f"\nDetailed sample for {model_name}/{sample_type}:")
    print("-" * 60)
    
    count = 0
    for video_id, video_data in sample_data.items():
        if count >= limit:
            break
        print(f"Video {video_id}:")
        print(f"  KISA_SPI: {video_data['KISA_SPI']:.4f}")
        print(f"  SGA_SPI: {video_data['SGA_SPI']:.4f}")
        print(f"  IF_score: {video_data['IF_score']:.4f}")
        print(f"  (KISA: {video_data['KISA_score']:.3f}, SGA: {video_data['SGA_score']:.3f}, SPI: {video_data['SPI_score']:.3f})")
        count += 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Calculate combined scores: KISA_SPI, SGA_SPI, and IF_score for each video.'
    )
    parser.add_argument(
        '--video-scores-file',
        type=str,
        default="/path/to/json",
        help='Input JSON file path for video scores '
    )
    parser.add_argument(
        '--emergence-scores-file',
        type=str,
        default="/path/to/json",
        help='Input JSON file path for emergence/disappearance scores'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default="/path/to/json",
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    print("Starting combined score calculation...")
    print(f"Video scores file: {args.video_scores_file}")
    print(f"Emergence scores file: {args.emergence_scores_file}")
    print(f"Output file: {args.output_file}")
    
    # Calculate combined scores
    results = calculate_combined_scores(args.video_scores_file, args.emergence_scores_file)
    
    if not results:
        print("No results to save!")
        return
    
    # Save results
    save_results(results, args.output_file)
    
    # Print summary
    print_summary(results)
    
    # Print detailed sample for first model/sample type
    first_model = list(results.keys())[0]
    first_sample = list(results[first_model].keys())[0]
    print_detailed_sample(results, first_model, first_sample)
    
    print(f"\nCombined score calculation complete! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()

