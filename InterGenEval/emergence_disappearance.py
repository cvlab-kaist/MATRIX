"""
hallucination_detector.py

Usage:
  python hallucination_detector.py \
      --metas_dir metas \
      --videos_dir generated_videos \
      --output results/hallucination_analysis.json \
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

# System instruction for hallucination detection
EMERGE_DISAPPEAR_INSTRUCTION = """
# Role

You are a hallucination detection expert.

Your task is to evaluate a sequence of video frames relative to a fixed anchor frame (frame 0) and determine whether:

- Any known entity class emerged without a bounding box, or
- Any previously tracked instance disappeared in later frames.

---

## Inputs

You are given:

- `frame_0`: the anchor/reference frame
- `frames_k`: a list of frames where k ∈ [1, N]
- Each frame contains bounding boxes, and every bbox is defined by a `(class, color)` identity
- A color-to-class mapping JSON:

```json
{
  "entities": ["person", "cup", "paper"],
  "colors": ["green", "red", "blue"]
}

```

- The arrays `entities` and `colors` are index-aligned, e.g., `"red"` → `"cup"`

Use this mapping to identify and track instances consistently across all frames.

---

##  Detection Rules

### 1. Emergence

Mark `emergence = "yes"` if any frame *k* contains:

- An unboxed object of a class listed in `entities`, and
- That class had no visual instance (boxed or unboxed) in frame 0

This includes cases where:

- The object appears fully unboxed in the background
- The object appears embedded inside another bbox (e.g., a ball inside a person)

 Track all frame indices where emergence occurred.

---

### 2. Disappearance

Let the set of `(class, color)` pairs from `frame_0` define the complete instance roster.

For each frame *k*, there must be a bbox with the same (class, color) for every such instance.

If any original instance is missing in frame *k*, mark `disappearance = "yes"` and include:

- The frame index *k*
- A description of which instances were lost (by `(class, color)` pair or class count)

---

##  Output Format

Produce a single JSON object that summarizes emergence and disappearance across all frames:

```json
{
  "emergence": "yes" | "no",
  "emergence_frames": [<frame_idx_1>, <frame_idx_2>, ...],
  "emergence_reason": "brief explanation or empty string if no",

  "disappearance": "yes" | "no",
  "disappearance_frames": [<frame_idx_1>, <frame_idx_2>, ...],
  "disappearance_reason": "list missing instances as (class,color) and/or class-level count deltas"
}

```
##  Evaluation Notes

- You must compare all frames after frame 0 against the instance roster from frame 0.
- Ignore any objects not listed in the `entities` array.
- Emergence is class-based: a second instance of a class (without a bbox) can be emergent if not present in frame 0.
- If no emergence or disappearance occurs in any frame, all values should default to `"no"`, `[]`, and `""`.
"""


def natural_key(path_str: str) -> int:
    stem = Path(path_str).stem
    nums = re.findall(r'\d+', stem)
    return int(nums[-1]) if nums else 0


def load_json_metadata(json_path: Path) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'detailed_caption' in data:
        del data['detailed_caption']
    
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
    
    if total_frames == 0:
        frame_indices = []
    elif total_frames == 1:
        frame_indices = [0]
    else:
        middle_indices = list(range(stride, total_frames - 1, stride))
        frame_indices = [0] + middle_indices + [total_frames - 1]
        frame_indices = sorted(set(frame_indices))
    
    temp_dir = Path(tempfile.mkdtemp(prefix=f"frames_{video_id}_"))
    frame_files = []
    
    try:
        for i, frame_idx in enumerate(frame_indices):
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
    return frame_files, frame_indices


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


def analyze_all_frames_with_gpt(
    frame_0_path: str,
    frame_paths: List[str],
    frame_indices: List[int],
    metadata: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    seed: int = 42
) -> Dict[str, Any]:
    frame_0_url = encode_image_to_data_url(frame_0_path)
    
    gpt_input = {
        "entities": metadata.get("entity", []),
        "colors": metadata.get("colors", []),
        "frame_0": "Reference frame with bounding boxes",
        "frames_k": f"Frames {frame_indices[1:]} to be compared against frame 0"
    }
    
    metadata_text = f"Input Data:\n{json.dumps(gpt_input, indent=2, ensure_ascii=False)}"
    
    user_content = [
        {"type": "text", "text": metadata_text},
        {"type": "text", "text": f"Frame 0 (Reference):"},
        {"type": "image_url", "image_url": {"url": frame_0_url}}
    ]
    print(frame_indices)
    for i, (frame_path, frame_idx) in enumerate(zip(frame_paths[1:], frame_indices[1:]), 1):
        frame_url = encode_image_to_data_url(frame_path)
        user_content.extend([
            {"type": "text", "text": f"Frame {frame_idx}:"},
            {"type": "image_url", "image_url": {"url": frame_url}}
        ])
    
    messages = [
        {"role": "system", "content": EMERGE_DISAPPEAR_INSTRUCTION},
        {
            "role": "user", 
            "content": user_content
        }
    ]
    
    print(f"Analyzing frame 0 vs frames {frame_indices[1:]}...")
    
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
        elif raw_response.startswith("{") and raw_response.endswith("}"):
            json_str = raw_response
        else:
            return {"raw_response": raw_response, "parsing_error": "No valid JSON found"}
        
        parsed_result = json.loads(json_str)
        return parsed_result
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse GPT response as JSON: {e}")
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
    
    frame_paths, frame_indices = load_frames_from_video(video_path, video_id, stride)
    print(f"Extracted {len(frame_paths)} frames for video {video_id}")
    
    if len(frame_paths) < 2:
        print(f"Warning: Video {video_id} has less than 2 frames, skipping hallucination detection")
        return {
            "video_id": metadata.get("video_id"),
            "prompt": metadata.get("prompt"),
            "entity": metadata.get("entity", []),
            "colors": metadata.get("colors", []),
            "frames_analyzed": len(frame_paths),
            "model_used": model_name,
            "hallucination_results": [],
            "error": "Insufficient frames for comparison"
        }
    
    frame_0_path = frame_paths[0]
    
    try:
        analysis_result = analyze_all_frames_with_gpt(
            frame_0_path=frame_0_path,
            frame_paths=frame_paths,
            frame_indices=frame_indices,
            metadata=metadata,
            client=client,
            model_name=model_name
        )
        
        print(f"✓ Completed analysis for frame 0 vs all frames {frame_indices[1:]}")
        
        hallucination_results = []
        
        emergence_frames = analysis_result.get("emergence_frames", [])
        disappearance_frames = analysis_result.get("disappearance_frames", [])
        
        for i, frame_idx in enumerate(frame_indices[1:], 1):
            frame_result = {
                "frame_idx": frame_idx,
                "emergence": "yes" if frame_idx in emergence_frames else "no",
                "disappearance": "yes" if frame_idx in disappearance_frames else "no",
                "emergence_reason": analysis_result.get("emergence_reason", "") if frame_idx in emergence_frames else "",
                "disappearance_reason": analysis_result.get("disappearance_reason", "") if frame_idx in disappearance_frames else ""
            }
            hallucination_results.append(frame_result)
        
    except Exception as e:
        print(f"✗ Error analyzing all frames: {e}")
        hallucination_results = []
        for i, frame_idx in enumerate(frame_indices[1:], 1):
            hallucination_results.append({
                "frame_idx": frame_idx,
                "error": str(e),
                "emergence": "error", 
                "disappearance": "error"
            })
    
    # calculate hallucination score
    emergence_count = 0
    disappearance_count = 0
    
    for frame_result in hallucination_results:
        if frame_result["emergence"] == "yes":
            emergence_count += 1
        if frame_result["disappearance"] == "yes":
            disappearance_count += 1
    
    total_frames = len(frame_paths) - 1
    emergence_score = emergence_count / total_frames if total_frames > 0 else 0
    disappearance_score = disappearance_count / total_frames if total_frames > 0 else 0
    
    hallucination_score = (emergence_score + disappearance_score) / 2
    
    if frame_paths:
        temp_dir = Path(frame_paths[0]).parent
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary files in: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")
    
    result = {
        "video_id": metadata.get("video_id"),
        "prompt": metadata.get("prompt"),
        "entity": metadata.get("entity", []),
        "colors": metadata.get("colors", []),
        "frames_analyzed": len(frame_paths),
        "model_used": model_name,
        "emergence_score": round(1 - emergence_score, 3),
        "disappearance_score":round(1 - disappearance_score),
        "hallucination_score": round(1 - hallucination_score),
        "hallucination_results": hallucination_results,  
        "overall_analysis": analysis_result if 'analysis_result' in locals() else {}  # 전체 분석 결과
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Detect hallucinations in video frames using GPT")
    parser.add_argument("--metas_dir", type=str, required=True, help="Path to the metas directory containing JSON files")
    parser.add_argument("--videos_dir", type=str, required=True, help="Path to the videos directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save the hallucination analysis results (JSON)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--stride", type=int, default=1, help="Stride for frame sampling")
    parser.add_argument("--sample_type", type=str, default="all", help="Type of sample to use")
    parser.add_argument("--type", type=str, default="CogvideoX", help="Type of video generation model")
    
    args = parser.parse_args()

    # 입력 검증
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
            print(f"✓ Successfully processed video {video_id} with {len(result.get('hallucination_results', []))} frame analysis")                
        except Exception as e:
            print(f"✗ Error processing video {video_id}: {e}")
            continue
    
    if not results:
        raise RuntimeError("No videos were successfully processed")
    
    video_model = args.type 
    output_path = Path(f"{args.output}/{video_model}/{args.sample_type}/_emergence_disappearance.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Hallucination detection complete! Results saved to: {output_path}")
    print(f"📊 Successfully processed {len(results)} out of {len(json_files)} videos")
    print(f"📋 Each video entry contains comprehensive frame analysis with emergence/disappearance detection")


if __name__ == "__main__":
    main()
