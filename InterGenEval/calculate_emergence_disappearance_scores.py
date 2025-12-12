#!/usr/bin/env python3
"""
Calculate emergence and disappearance scores for all models and sample types.

Emergence score = 1 - (frames with emergence="yes") / (total frames analyzed)
Disappearance score = 1 - (frames with disappearance="yes") / (total frames analyzed)
"""

import json
from math import exp
import os
import argparse
from pathlib import Path
from collections import defaultdict

def calculate_scores_for_file(file_path, sample_type):
    """Calculate emergence and disappearance scores for a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        excluded_count = 0
        
        for video_data in data:
            video_id = video_data['video_id']
            
            should_exclude = False
            
            frames_analyzed = len(video_data.get('hallucination_results', []))
            hallucination_results = video_data.get('hallucination_results', [])
            
            # Count frames with emergence="yes" and disappearance="yes"
            emergence_yes_count = 0
            disappearance_yes_count = 0
            
            for frame_result in hallucination_results:
                if frame_result.get('emergence') == 'yes':
                    emergence_yes_count += 1
                if frame_result.get('disappearance') == 'yes':
                    disappearance_yes_count += 1
            
            # Calculate scores
            lambda_value = 5
            emergence_score = 1 - lambda_value * (emergence_yes_count/frames_analyzed) if frames_analyzed > 0 else 0
            disappearance_score = 1 - lambda_value * (disappearance_yes_count/frames_analyzed) if frames_analyzed > 0 else 0
            SPI_score = (emergence_score + disappearance_score) / 2
            
            results[video_id] = {
                'emergence_score': emergence_score,
                'disappearance_score': disappearance_score,
                'frames_analyzed': frames_analyzed,
                'emergence_yes_count': emergence_yes_count,
                'disappearance_yes_count': disappearance_yes_count,
                'SPI_score': SPI_score
            }
        
        return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

def process_all_models(base_dir):
    """Process all models and sample types in the emergence_disappearance_results directory."""
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist!")
        return
    
    all_results = {}
    
    # Get all model directories
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        
        all_results[model_name] = {}
        
        # Get all sample type directories within each model
        sample_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        
        for sample_dir in sorted(sample_dirs):
            sample_type = sample_dir.name
            print(f"  Processing sample type: {sample_type}")
            
            # Look for the emergence_disappearance JSON file
            json_file = sample_dir / "_emergence_disappearance.json"
            
            if json_file.exists():
                print(f"    Found file: {json_file}")
                video_results = calculate_scores_for_file(json_file, sample_type)
                all_results[model_name][sample_type] = video_results
                print(f"    Processed {len(video_results)} videos")
            else:
                print(f"    No _emergence_disappearance.json found in {sample_dir}")
                all_results[model_name][sample_type] = {}
    
    return all_results

def save_results(results, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("SUMMARY")
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
            emergence_scores = [v['emergence_score'] for v in sample_data.values()]
            disappearance_scores = [v['disappearance_score'] for v in sample_data.values()]
            
            if emergence_scores:
                avg_emergence = sum(emergence_scores) / len(emergence_scores)
                avg_disappearance = sum(disappearance_scores) / len(disappearance_scores)
                print(f"    Average emergence score: {avg_emergence:.4f}")
                print(f"    Average disappearance score: {avg_disappearance:.4f}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Calculate emergence and disappearance scores for all models and sample types.'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default="/path/to/basedir",
        help='Base directory containing model directories'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default="/path/to/json",
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    print("Starting emergence and disappearance score calculation...")
    print(f"Base directory: {args.base_dir}")
    print(f"Output file: {args.output_file}")
    
    # Process all models and sample types
    results = process_all_models(args.base_dir)
    
    if not results:
        print("No results to save!")
        return
    
    # Save results
    save_results(results, args.output_file)
    
    # Print summary
    print_summary(results)
    
    print(f"\nCalculation complete! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
