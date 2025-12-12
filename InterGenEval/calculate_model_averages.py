#!/usr/bin/env python3
"""
Calculate average KISA_SPI, SGA_SPI, and IF_score for each model from combined_scores_summary.json
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

def calculate_model_averages(input_file):
    """Calculate average scores for each model."""
    
    # Load the combined scores file
    print("Loading combined scores...")
    combined_scores = load_json_file(input_file)
    if not combined_scores:
        print("Failed to load combined scores!")
        return {}
    
    print("Calculating model averages...")
    model_averages = {}
    
    # Process each model
    for model_name, model_data in combined_scores.items():
        print(f"Processing model: {model_name}")
        
        # Collect all scores for this model across all sample types
        all_kisa_spi = []
        all_sga_spi = []
        all_if_scores = []
        
        sample_type_stats = {}
        
        # Process each sample type
        for sample_type, sample_data in model_data.items():
            if not sample_data:
                continue
                
            # Collect scores for this sample type
            sample_kisa_spi = [v['KISA_SPI'] for v in sample_data.values()]
            sample_sga_spi = [v['SGA_SPI'] for v in sample_data.values()]
            sample_if_scores = [v['IF_score'] for v in sample_data.values()]
            
            # Calculate averages for this sample type
            if sample_kisa_spi:
                sample_type_stats[sample_type] = {
                    'video_count': len(sample_kisa_spi),
                    'avg_KISA_SPI': sum(sample_kisa_spi) / len(sample_kisa_spi),
                    'avg_SGA_SPI': sum(sample_sga_spi) / len(sample_sga_spi),
                    'avg_IF_score': sum(sample_if_scores) / len(sample_if_scores)
                }
                
                # Add to overall model scores
                all_kisa_spi.extend(sample_kisa_spi)
                all_sga_spi.extend(sample_sga_spi)
                all_if_scores.extend(sample_if_scores)
        
        # Calculate overall model averages
        if all_kisa_spi:
            model_averages[model_name] = {
                'total_videos': len(all_kisa_spi),
                'overall_avg_KISA_SPI': sum(all_kisa_spi) / len(all_kisa_spi),
                'overall_avg_SGA_SPI': sum(all_sga_spi) / len(all_sga_spi),
                'overall_avg_IF_score': sum(all_if_scores) / len(all_if_scores),
                'sample_type_breakdown': sample_type_stats
            }
        else:
            model_averages[model_name] = {
                'total_videos': 0,
                'overall_avg_KISA_SPI': 0.0,
                'overall_avg_SGA_SPI': 0.0,
                'overall_avg_IF_score': 0.0,
                'sample_type_breakdown': {}
            }
    
    return model_averages

def save_results(results, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*100)
    print("MODEL AVERAGES SUMMARY")
    print("="*100)
    
    # Sort models by overall IF_score for ranking
    sorted_models = sorted(results.items(), key=lambda x: x[1]['overall_avg_IF_score'], reverse=True)
    
    print(f"{'Model':<20} {'Videos':<8} {'KISA_SPI':<12} {'SGA_SPI':<12} {'IF_score':<12} {'Rank':<6}")
    print("-" * 100)
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        print(f"{model_name:<20} {data['total_videos']:<8} {data['overall_avg_KISA_SPI']:<12.4f} {data['overall_avg_SGA_SPI']:<12.4f} {data['overall_avg_IF_score']:<12.4f} {rank:<6}")
    
    print("\n" + "="*100)
    print("DETAILED BREAKDOWN BY SAMPLE TYPE")
    print("="*100)
    
    for model_name, data in sorted_models:
        print(f"\n{model_name}:")
        print("-" * 60)
        print(f"Total videos: {data['total_videos']}")
        print(f"Overall KISA_SPI: {data['overall_avg_KISA_SPI']:.4f}")
        print(f"Overall SGA_SPI: {data['overall_avg_SGA_SPI']:.4f}")
        print(f"Overall IF_score: {data['overall_avg_IF_score']:.4f}")
        
        if data['sample_type_breakdown']:
            print("\nSample type breakdown:")
            for sample_type, stats in data['sample_type_breakdown'].items():
                print(f"  {sample_type}: {stats['video_count']} videos")
                print(f"    KISA_SPI: {stats['avg_KISA_SPI']:.4f}")
                print(f"    SGA_SPI: {stats['avg_SGA_SPI']:.4f}")
                print(f"    IF_score: {stats['avg_IF_score']:.4f}")
                
    print(f"{'Model':<20} {'Videos':<8} {'KISA_SPI':<12} {'SGA_SPI':<12} {'IF_score':<12} {'Rank':<6}")
    print("-" * 100)
    
    print("\n" + "="*100)
    print("MODEL AVERAGES SUMMARY")
    print("="*100)
    print(f"{'Model':<20} {'Videos':<8} {'KISA_SPI':<12} {'SGA_SPI':<12} {'IF_score':<12} {'Rank':<6}")
    print("-" * 100)
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        print(f"{model_name:<20} {data['total_videos']:<8} {data['overall_avg_KISA_SPI']:<12.4f} {data['overall_avg_SGA_SPI']:<12.4f} {data['overall_avg_IF_score']:<12.4f} {rank:<6}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Calculate average KISA_SPI, SGA_SPI, and IF_score for each model")
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing combined_scores_summary.json'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save the model averages summary JSON file'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='Path to combined_scores_summary.json file (default: {base_dir}/combined_scores_summary.json)'
    )
    
    args = parser.parse_args()
    
    # Determine input file path
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = os.path.join(args.base_dir, 'combined_scores_summary.json')
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print("Starting model average calculation...")
    print(f"Input file: {input_file}")
    print(f"Output file: {args.output_file}")
    
    # Calculate model averages
    results = calculate_model_averages(input_file)
    
    if not results:
        print("No results to save!")
        return
    
    # Save results
    save_results(results, args.output_file)
    
    # Print summary
    print_summary(results)
    
    print(f"\nModel average calculation complete! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
