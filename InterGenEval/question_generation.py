"""
question_generation.py

Usage:
  python question_generation.py \
      --metas_dir metas \
      --output results/questions.json \
      --model gpt-4o
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re

from openai import OpenAI  # pip install openai
from tqdm import tqdm  # pip install tqdm

# System instruction for question generation
QUESTION_GENERATION_INSTRUCTION = """
Instruction

Role:
You are an expert at generating evaluation questions for video interaction understanding.
Your task is to generate 10 yes/no questions per interaction based on the input (video prompt and entities involved).

Input Format:

{
  "video_id": "xxx",
  "prompt": "<natural language video description>",
  "entity": ["<object1>", "<object2>", ...],
  "appearance": ["<object1 description>", "<object2 description>", ...],
  "colors": ["<color1>", "<color2>", ...]
}

entity[i], appearance[i], and colors[i] are aligned by index.

Key Interaction Extraction Rules

For each item, extract key_interaction from the prompt.

Use natural language structure of the prompt to identify key interactions.

Extract multiple key interactions per prompt if the prompt contains multiple interactions.

Question Generation Rules (Per Key Interaction)

Generate 10 yes/no questions for each item.key_interaction, following the structure below:

1. Pre-interaction state

Ask whether the subject is not yet interacting with the object.

Mention both subject and object, with bbox color and appearance.

2–5. Progressive interaction steps

Ask whether the subject begins, initiates, and progresses the action toward the object.

Use observable, frame-verifiable terms like “moves hand toward,” “makes contact,” “holds,” etc.

Mention both subject and object.

6. Post-interaction state

Ask whether the result or outcome of the interaction is visible.

Must mention both subject and object.

7. Action by subject

Ask whether the subject is performing an action.

Don't mention the specific action.

Mention only the subject.

8. Correct agent

Ask whether the subject is the one performing the interaction.

Mention both subject and object.

9. Correct recipient

Ask whether the object is being interacted with by the subject.

Mention both subject and object.

10. Recipient confirmation

Ask if the object is indeed the one recieving the action in the interaction.

Mention only the object.

Notes

Repeat the 10-question generation for each key_interaction in the input’s items list.

Always preserve index alignment of entities (i.e., appearance[0] with colors[0] for entity[0]).

Use short, present-tense, frame-checkable language.

Avoid speculative/future phrasing like "might" or "will".

Entity Mention Rule

When writing each question:

Always use appearance[i] + (colors[i] bbox) to refer to entities.

Match appearance and color using the same index as entity[i].

Example: If entity = ["boy", "girl"], appearance = ["young boy", "girl in a red dress"], and colors = ["green", "red"], then:

"young boy (green bbox)"

"girl in a red dress (red bbox)"

Output Format:
Return the same input JSON, with each item.questions field filled with exactly 10 generated questions as strings.

Example
Input:
{
    "video_id": "000",
    "prompt": "The man in a gray T-shirt with short black hair opens the silver fridge door in the kitchen.",
    "entity": [
      "man",
      "fridge"
    ],
    "appearance": [
      "man in a gray T-shirt with short black hair",
      "silver fridge door"
    ],
    "colors": [
      "red",
      "green"
    ]
  }
Output:
"items": [
    {
      "key_interaction": "The man opens the fridge door",
      "questions": [
        "Is the man in a gray T-shirt with short black hair (red bbox) not touching the silver fridge door (green bbox) before the interaction?",
        "Does the man in a gray T-shirt with short black hair (red bbox) move his hand toward the silver fridge door (green bbox) in the kitchen?",
        "Does the man in a gray T-shirt with short black hair (red bbox) make contact with the silver fridge door (green bbox) in the kitchen?",
        "While the man in a gray T-shirt with short black hair (red bbox) keeps hand on it, is the silver fridge door (green bbox) partially open?",
        "Is the opening or angle of the silver fridge door (green bbox) in the kitchen increasing by the man in a gray T-shirt with short black hair (red bbox)?",
        "Is the silver fridge door (green bbox) fully opened by the man in a gray T-shirt with short black hair (red bbox)?",
        "Is the man in a gray T-shirt with short black hair (red bbox) taking an action?",
        "Is the man in a gray T-shirt with short black hair (red bbox) the one operating the silver fridge door (green bbox) in the kitchen?",
        "Is the silver fridge door (green bbox) the one being opened by the man in a gray T-shirt with short black hair (red bbox)?",
        "Is the silver fridge door (green bbox) the item being opened?"
      ]
    }
  ]
"""


def natural_key(path_str: str) -> int:
    stem = Path(path_str).stem
    nums = re.findall(r'\d+', stem)
    return int(nums[-1]) if nums else 0


def load_json_metadata(json_path: Path) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # detailed_caption 제거
    if 'detailed_caption' in data:
        del data['detailed_caption']
    
    return data


def generate_questions_with_gpt(
    metadata: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    seed: int = 42
) -> Dict[str, Any]:
    gpt_input = {
        "video_id": metadata.get("video_id"),
        "prompt": metadata.get("prompt"),
        "entity": metadata.get("entity", []),
        "appearance": metadata.get("appearance", []),
        "colors": metadata.get("colors", [])
    }
    
    metadata_text = f"Input Data:\n{json.dumps(gpt_input, indent=2, ensure_ascii=False)}"
    
    messages = [
        {"role": "system", "content": QUESTION_GENERATION_INSTRUCTION},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": metadata_text}
            ]
        }
    ]
    
    print(f"Generating questions for video {metadata.get('video_id')}...")
    
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
    client: OpenAI,
    model_name: str
) -> Dict[str, Any]:
    json_path = metas_dir / f"{video_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    
    metadata = load_json_metadata(json_path)
    print(f"Loaded metadata for video {video_id}")
    
    try:
        question_result = generate_questions_with_gpt(
            metadata=metadata,
            client=client,
            model_name=model_name
        )
        
        print(f"✓ Completed question generation for video {video_id}")
        
        result = {
            "video_id": metadata.get("video_id"),
            "prompt": metadata.get("prompt"),
            "entity": metadata.get("entity", []),
            "appearance": metadata.get("appearance", []),
            "colors": metadata.get("colors", []),
            "items": question_result.get("items", [])
        }
        
    except Exception as e:
        print(f"✗ Error generating questions: {e}")
        result = {
            "video_id": metadata.get("video_id"),
            "prompt": metadata.get("prompt"),
            "entity": metadata.get("entity", []),
            "appearance": metadata.get("appearance", []),
            "colors": metadata.get("colors", []),
            "items": [
                {
                    "key_interaction": "interaction",
                    "questions": []
                }
            ],
            "error": str(e)
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate interaction questions using GPT")
    parser.add_argument("--metas_dir", type=str, required=True, help="Path to the metas directory containing JSON files")
    parser.add_argument("--output", type=str, required=True, help="Directory path to save the question generation results (individual JSON files per video)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    
    args = parser.parse_args()

    metas_dir = Path(args.metas_dir)
    
    if not metas_dir.exists():
        raise FileNotFoundError(f"Metas directory not found: {metas_dir}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = OpenAI() 
    json_files = sorted([f for f in metas_dir.iterdir() if f.suffix == '.json'], key=natural_key)
    print(f"Found {len(json_files)} video metadata files to process")
    
    if not json_files:
        raise RuntimeError(f"No valid JSON files found in {metas_dir}")
    
    success_count = 0

    for json_file in tqdm(json_files, desc="Processing videos", unit="video"):
        video_id = json_file.stem
        try:
            result = process_single_video(
                video_id=video_id,
                metas_dir=metas_dir,
                client=client,
                model_name=args.model
            )

            output_file = output_dir / f"{video_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            num_questions = len(result.get('items', [{}])[0].get('questions', [])) if result.get('items') else 0
            print(f"✓ Successfully processed video {video_id} with {num_questions} questions, saved to {output_file}")
            success_count += 1
                
        except Exception as e:
            print(f"✗ Error processing video {video_id}: {e}")
            continue
    
    if success_count == 0:
        raise RuntimeError("No videos were successfully processed")
    
    print(f"\n🎉 Question generation complete! Results saved to: {output_dir}")
    print(f"📊 Successfully processed {success_count} out of {len(json_files)} videos")
    print(f"📋 Each video has its own JSON file (e.g., 000.json, 001.json, ...)")


if __name__ == "__main__":
    main()
