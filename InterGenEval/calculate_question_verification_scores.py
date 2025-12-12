#!/usr/bin/env python3
import json
import os
import argparse

def calculate_video_scores(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    video_scores = {}
    
    for video_data in data:
        video_id = video_data['video_id']
        prompt = video_data['prompt']
        items = video_data.get('items', [])
        
        all_verification_results = []
        for item in items:
            item_verification_results = item.get('question_verification_results', [])
            all_verification_results.extend(item_verification_results)
        
        yes_count = sum(1 for result in all_verification_results if result['answer'] == 'yes')
        total_questions = len(all_verification_results)
        
        score = yes_count / total_questions if total_questions > 0 else 0
        score = round(score, 3)
        
        # Q1-Q6: Interaction score
        interaction_results = [result for result in all_verification_results if result['question_id'] <= 6]
        interaction_yes = sum(1 for result in interaction_results if result['answer'] == 'yes')
        interaction_score = round(interaction_yes / len(interaction_results), 3) if interaction_results else 0
        
        # Q7-Q10: Grounding score
        grounding_results = [result for result in all_verification_results if 7 <= result['question_id'] <= 10]
        grounding_yes = sum(1 for result in grounding_results if result['answer'] == 'yes')
        grounding_score = round(grounding_yes / len(grounding_results), 3) if grounding_results else 0
        
        video_scores[video_id] = {
            'prompt': prompt,
            'yes_count': yes_count,
            'total_questions': total_questions,
            'score': score,
            'score_percentage': score * 100,
            'interaction_score': interaction_score,
            'interaction_score_percentage': round(interaction_score * 100, 3),
            'grounding_score': grounding_score,
            'grounding_score_percentage': round(grounding_score * 100, 3)
        }
    
    return video_scores

def print_scores(video_scores):
    print("=" * 120)
    print("Video Question Verification Scores")
    print("=" * 120)
    print(f"{'Video ID':<8} {'Overall':<12} {'Interaction':<12} {'Grounding':<12} {'Prompt'}")
    print(f"{'':8} {'Score':<12} {'Score':<12} {'Score':<12} {'':<50}")
    print("-" * 120)
    
    for video_id, data in video_scores.items():
        print(f"{video_id:<8} {data['score']:.3f} ({data['score_percentage']:.1f}%) {data['interaction_score']:.3f} ({data['interaction_score_percentage']:.1f}%) {data['grounding_score']:.3f} ({data['grounding_score_percentage']:.1f}%) {data['prompt'][:30]}...")
    
    print("-" * 120)

def print_overall_statistics(video_scores):
    total_videos = len(video_scores)
    total_yes = sum(data['yes_count'] for data in video_scores.values())
    total_questions = sum(data['total_questions'] for data in video_scores.values())
    overall_score = total_yes / total_questions if total_questions > 0 else 0
    
    avg_interaction = sum(data['interaction_score'] for data in video_scores.values()) / total_videos
    avg_grounding = sum(data['grounding_score'] for data in video_scores.values()) / total_videos
    
    print(f"Overall Statistics:")
    print(f"  Total Videos: {total_videos}")
    print(f"  Total Questions: {total_questions}")
    print(f"  Total Yes Answers: {total_yes}")
    print(f"  Overall Average Score: {overall_score:.3f} ({overall_score * 100:.3f}%)")
    print(f"  Average Interaction Score: {avg_interaction:.3f} ({avg_interaction * 100:.3f}%)")
    print(f"  Average Grounding Score: {avg_grounding:.3f} ({avg_grounding * 100:.3f}%)")
    print("=" * 80)

def save_scores_to_json(video_scores, output_file_path):
    total_videos = len(video_scores)
    total_yes = sum(data['yes_count'] for data in video_scores.values())
    total_questions = sum(data['total_questions'] for data in video_scores.values())
    overall_score = total_yes / total_questions if total_questions > 0 else 0
    overall_score = round(overall_score, 3)
    
    avg_interaction = sum(data['interaction_score'] for data in video_scores.values()) / total_videos
    avg_grounding = sum(data['grounding_score'] for data in video_scores.values()) / total_videos
    
    output_data = {
        "video_scores": video_scores,
        "overall_statistics": {
            "total_videos": total_videos,
            "total_questions": total_questions,
            "total_yes_answers": total_yes,
            "overall_average_score": overall_score,
            "overall_average_percentage": round(overall_score * 100, 3),
            "average_interaction_score": round(avg_interaction, 3),
            "average_interaction_score_percentage": round(avg_interaction * 100, 3),
            "average_grounding_score": round(avg_grounding, 3),
            "average_grounding_score_percentage": round(avg_grounding * 100, 3)
        }
    }
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Score results saved to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate question verification scores from JSON file.'
    )
    parser.add_argument(
        '--json-file',
        type=str,
        default="/path/to/json",
        help='Input JSON file path containing question verification results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default="/path/to/json",
        help='Output JSON file path for calculated scores'
    )
    
    args = parser.parse_args()
    
    json_file_path = args.json_file
    output_file_path = args.output_file
    
    print(f"Input JSON file: {json_file_path}")
    print(f"Output file: {output_file_path}")
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        return
    
    try:
        print("Calculating scores...")
        video_scores = calculate_video_scores(json_file_path)
        print_overall_statistics(video_scores)
        print_scores(video_scores)
        save_scores_to_json(video_scores, output_file_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
