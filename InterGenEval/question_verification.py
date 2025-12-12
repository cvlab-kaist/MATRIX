"""
question_verification.py

Usage:
  python question_verification.py \
      --metas_dir /media/data2/chan/EVAL_DATA/easy/interaction_order \
      --videos_dir /media/data2/chan/OUTPUTS/cogvideox_2b/easy \
      --output results \
      --model gpt-4o \
      --type CogvideoX
"""

import os
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re
import cv2
import numpy as np
import tempfile
import shutil

from openai import OpenAI  # pip install openai
from tqdm import tqdm  # pip install tqdm

QUESTION_VERIFICATION_SYSTEM_INSTRUCTION = """
Task — answer yes/no questions about video frames

Inputs

An ordered list of frames (indexed from 0).

A list of yes/no questions. Each question specifies entities with their colored bboxes (e.g., “a man (yellow bbox) touches a cup (blue bbox)”).

Rules

Judge by visible evidence only. Do not infer beyond what is clearly seen in the frames.

Color disambiguation. Because text alone may not uniquely identify an instance in a frame, use the specified colored bbox as the reference to pinpoint the intended entity, and base your judgment on that entity’s visible evidence.

Per-Question Procedure

Select the decisive frame.
Scan frames and choose the single frame that gives the clearest evidence for “yes” or “no”.

If multiple frames are equally decisive, pick the earliest index.

If no frame provides clear evidence, answer “no” and set frame_index to null.

Answer (yes/no).
Based solely on what is visible in the decisive frame (and color-tagged entities), answer “yes” or “no”.

Visual plausibility check (on the decisive frame).
If the decisive frame shows visually implausible anatomy/geometry, override the answer with “no (visually implausible)”.
Plausibility red flags include (not exhaustive):

Human anatomy anomalies: duplicated/missing hands/feet/arms, impossible joint bends, detached limbs.

Object/body fusion/splitting artifacts within a bbox, or severe distortions that break physical continuity.

Self-intersection or impossible penetration (e.g., hand passes through a solid object) that invalidates the observation.

Purpose: reject interactions that “occur” but are visually nonsensical.

Output (JSON)
Return an array of objects:

[
  { "question_id": 1, "answer": "yes", "frame_index": 12 },
  { "question_id": 2, "answer": "no (visually implausible)", "frame_index": 7 },
  { "question_id": 3, "answer": "no", "frame_index": null }
]


answer ∈ {"yes", "no", "no (visually implausible)"}

frame_index is the decisive frame used to judge the answer (or null if none was decisive).
"""


def natural_key(path_str: str) -> int:
    stem = Path(path_str).stem
    nums = re.findall(r'\d+', stem)
    return int(nums[-1]) if nums else 0


def load_json_metadata(json_path: Path) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_frames_from_video(video_path: Path, video_id: str, stride: int) -> List[str]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info - Total frames: {total_frames}, FPS: {fps:.2f}")
    
    frame_indices = [0]  
    
    if total_frames > 2:  
        middle_frames = list(range(stride, total_frames - 1, stride))
        frame_indices.extend(middle_frames)
    
    if total_frames > 1:
        last_frame = total_frames - 1
        if last_frame not in frame_indices:
            frame_indices.append(last_frame)
    
    frame_indices.sort()
    
    print(f"Selected frame indices: {frame_indices}")
    
    temp_dir = Path(tempfile.mkdtemp(prefix=f"frames_{video_id}_"))
    frame_files = []
    
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Cannot read frame {frame_idx}")
                continue
            
            temp_frame_path = temp_dir / f"{frame_idx:05d}.png"
            success = cv2.imwrite(str(temp_frame_path), frame)
            
            if success:
                frame_files.append(str(temp_frame_path))
            else:
                print(f"Warning: Failed to save frame {frame_idx}")
    
    finally:
        cap.release()
    
    if not frame_files:
        raise RuntimeError(f"No frames could be extracted from video: {video_path}")
    
    print(f"Extracted {len(frame_files)} frames from video {video_id}")
    return frame_files


def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    
    suffix = Path(image_path).suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png" 
    
    return f"data:{mime};base64,{b64}"


def analyze_questions_with_gpt(
    frame_paths: List[str],
    frame_indices: List[int],
    questions: List[str],
    metadata: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    seed: int = 42
) -> Dict[str, Any]:
    frame_urls = [encode_image_to_data_url(path) for path in frame_paths]
    
    gpt_input = {
        "prompt": metadata.get("prompt", ""),
        "questions": questions
    }
    
    metadata_text = f"Video Prompt: {metadata.get('prompt', '')}\n\nQuestions to answer:\n{json.dumps(questions, indent=2, ensure_ascii=False)}"
    
    user_content = [
        {"type": "text", "text": metadata_text},
        {"type": "text", "text": f"Available frames (indexed from 0):"},
    ]
    
    for i, (frame_url, frame_idx) in enumerate(zip(frame_urls, frame_indices)):
        user_content.extend([
            {"type": "text", "text": f"Frame {frame_idx}:"},
            {"type": "image_url", "image_url": {"url": frame_url}}
        ])
    
    messages = [
        {"role": "system", "content": QUESTION_VERIFICATION_SYSTEM_INSTRUCTION},
        {
            "role": "user", 
            "content": user_content
        }
    ]
    
    print(f"Analyzing {len(questions)} questions against {len(frame_paths)} frames...")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        seed=seed
    )
    
    raw_response = response.choices[0].message.content.strip()
    
    try:
        if "```json" in raw_response:
            json_start = raw_response.find("```json") + 7
            json_end = raw_response.find("```", json_start)
            json_str = raw_response[json_start:json_end].strip()
        elif raw_response.startswith("[") and raw_response.endswith("]"):
            json_str = raw_response
        else:
            return {"raw_response": raw_response, "parsing_error": "No valid JSON found"}
        
        json_str = json_str.strip()
        
        parsed_result = json.loads(json_str)
        return parsed_result
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse GPT response as JSON: {e}")
        print(f"Raw response: {raw_response}")
        print(f"JSON string: {json_str if 'json_str' in locals() else 'N/A'}")
        
        try:
            import re
            pattern = r'"question_id":\s*(\d+),\s*"answer":\s*"(yes|no)"'
            matches = re.findall(pattern, json_str)
            
            if matches:
                fallback_result = []
                for question_id, answer in matches:
                    fallback_result.append({
                        "question_id": int(question_id),
                        "answer": answer
                    })
                print(f"Fallback parsing successful: {fallback_result}")
                return fallback_result
        except Exception as fallback_e:
            print(f"Fallback parsing also failed: {fallback_e}")
        
        return {"raw_response": raw_response, "parsing_error": str(e)}


def process_single_video(
    video_id: str,
    metas_dir: Path,
    videos_dir: Path,
    client: OpenAI,
    model_name: str,
    stride: int
) -> Dict[str, Any]:
    json_path = metas_dir / f"{video_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    metadata = load_json_metadata(json_path)
    print(f"Loaded metadata for video {video_id}")
    
    json_video_id = metadata.get("video_id")
    if json_video_id is None:
        raise ValueError(f"video_id not found in JSON metadata for file: {json_path}")
    
    if str(json_video_id) != str(video_id):
        raise ValueError(f"Video ID mismatch! JSON file: {json_video_id}, Expected: {video_id}. Analysis stopped.")
    
    print(f"✓ Video ID verified: {video_id}")
    
    items = metadata.get("items", [])
    if not items or len(items) == 0:
        raise ValueError(f"No items found in metadata for video {video_id}")
    
    print(f"Found {len(items)} items for video {video_id}:")
    for i, item in enumerate(items):
        key_interaction = item.get("key_interaction", "")
        questions = item.get("questions", [])
        print(f"  Item {i}: {key_interaction} ({len(questions)} questions)")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_path = None
    
    for ext in video_extensions:
        candidate_path = videos_dir / f"{video_id}{ext}"
        if candidate_path.exists():
            video_path = candidate_path
            break
        candidate_path = videos_dir / f"{video_id}{ext.upper()}"
        if candidate_path.exists():
            video_path = candidate_path
            break
    
    if video_path is None:
        raise FileNotFoundError(f"Video file not found for video_id: {video_id}")
    
    frame_paths = load_frames_from_video(video_path, video_id, stride)
    print(f"Extracted {len(frame_paths)} frames for video {video_id}")
    
    if len(frame_paths) < 1:
        print(f"Warning: Video {video_id} has no frames, skipping questions analysis")
        return {
            "video_id": video_id,
            "prompt": metadata.get("prompt", ""),
            "frames_analyzed": len(frame_paths),
            "model_used": model_name,
            "items": [],
            "error": "No frames available for analysis"
        }
    
    frame_indices = []
    for frame_path in frame_paths:
        frame_filename = Path(frame_path).stem
        frame_idx = int(frame_filename)
        frame_indices.append(frame_idx)
    
    item_results = []
    
    for item_idx, item in enumerate(items):
        key_interaction = item.get("key_interaction", "")
        questions = item.get("questions", [])
        
        if len(questions) == 0:
            print(f"  ⚠️ No questions found for item {item_idx}: {key_interaction}")
            item_results.append({
                "item_index": item_idx,
                "key_interaction": key_interaction,
                "questions": questions,
                "question_verification_results": [],
                "error": "No questions found"
            })
            continue
        
        print(f"  🔍 Analyzing item {item_idx}: {key_interaction} ({len(questions)} questions)")
        
        try:
            question_verification_results = analyze_questions_with_gpt(
                frame_paths=frame_paths,
                frame_indices=frame_indices,
                questions=questions,
                metadata=metadata,
                client=client,
                model_name=model_name
            )
            print(f"  ✓ Completed analysis for item {item_idx}: {key_interaction}")
            
            item_results.append({
                "item_index": item_idx,
                "key_interaction": key_interaction,
                "questions": questions,
                "question_verification_results": question_verification_results
            })
            
        except Exception as e:
            print(f"  ✗ Error analyzing item {item_idx} ({key_interaction}): {e}")
            item_results.append({
                "item_index": item_idx,
                "key_interaction": key_interaction,
                "questions": questions,
                "question_verification_results": [],
                "error": str(e)
            })
    
    print(f"✓ Completed all items analysis for video {video_id}")
    
    if frame_paths:
        temp_dir = Path(frame_paths[0]).parent
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary files in: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")
    
    result = {
        "video_id": video_id,
        "prompt": metadata.get("prompt", ""),
        "frames_analyzed": len(frame_paths),
        "frame_indices": frame_indices,
        "model_used": model_name,
        "items": item_results  
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify questions against video frames using GPT")
    parser.add_argument("--metas_dir", type=str, required=True, help="Path to the metas directory containing JSON files")
    parser.add_argument("--videos_dir", type=str, required=True, help="Path to the videos directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save the question verification results (JSON)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--stride", type=int, default=5, help="Stride for frame sampling")
    parser.add_argument("--type", type=str, default="CogvideoX", help="Type of video generation model")
    parser.add_argument("--sample_type", type=str, default="all", help="Type of sample to use")
    args = parser.parse_args()

    metas_dir = Path(args.metas_dir)
    videos_dir = Path(args.videos_dir)
    
    if not metas_dir.exists():
        raise FileNotFoundError(f"Metas directory not found: {metas_dir}")
    
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    client = OpenAI()  
    json_files = sorted([f for f in metas_dir.iterdir() if f.suffix == '.json'], key=natural_key)
    print(f"Found {len(json_files)} video metadata files to process")
    
    if not json_files:
        raise RuntimeError(f"No valid JSON files found in {metas_dir}")
    
    results = []
    
    for json_file in tqdm(json_files, desc="Processing videos", unit="video"):
        video_id = json_file.stem
        try:
            result = process_single_video(
                video_id=video_id,
                metas_dir=metas_dir,
                videos_dir=videos_dir,
                client=client,
                model_name=args.model,
                stride=args.stride
            )
            results.append(result)
            items_count = len(result.get('items', []))
            total_questions = sum(len(item.get('questions', [])) for item in result.get('items', []))
            print(f"✓ Successfully processed video {video_id} with {len(result.get('frame_indices', []))} frames analyzed ({items_count} items, {total_questions} total questions)")                
        except Exception as e:
            print(f"✗ Error processing video {video_id}: {e}")
            continue
    
    if not results:
        raise RuntimeError("No videos were successfully processed")
    
    video_model = args.type 
    sample_type = args.sample_type
    output_path = Path(f"{args.output}/{video_model}/{sample_type}/question_verification.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Question verification complete! Results saved to: {output_path}")
    print(f"📊 Successfully processed {len(results)} out of {len(json_files)} videos")
    print(f"📋 Each video entry contains question verification results for all questions")


if __name__ == "__main__":
    main()