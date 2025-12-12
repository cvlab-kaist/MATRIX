# IMPORT
import os
import json
import argparse
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from utils import (
    ensure_dir,
    pil_to_base64_jpg,
    draw_bboxes,
    crop_with_padding
)

# FN: CHOOSE BEST BBOX WITH OPENAI
def choose_best_bbox_with_openai(
    appearance_text: str,
    image_crops_b64: List[str],
    labels_for_crops: List[str],
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Send N crops as images + appearance text. Ask GPT-4o to select best index.
    Returns JSON like:
      {
        "best_idx": int,             # index in [0..N-1]
        "confidence": float,         # 0..1
        "rationale": "..."
      }
    If it cannot decide, it should return best_idx = -1 with low confidence.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build message content
    content = []
    content.append({
        "type": "text",
        "text": (
            "You will be given several image crops of candidate objects and a textual appearance description.\n"
            "Pick the single crop that best matches the description. If none match, choose -1.\n"
            "Always respond with strict JSON only:\n"
            "{ \"best_idx\": integer, \"confidence\": number, \"rationale\": string }"
        )
    })
    content.append({"type": "text", "text": f"Appearance description:\n{appearance_text}"})
    content.append({"type": "text", "text": "Candidates follow. Each has an index shown before the image."})
    for idx, (b64, lab) in enumerate(zip(image_crops_b64, labels_for_crops)):
        content.append({"type": "text", "text": f"Candidate index: {idx} (label: {lab})"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    # Ask for JSON output
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful vision-language matcher that returns JSON only."},
            {"role": "user", "content": content}
        ],
        temperature=temperature,
        max_tokens=400,
        response_format={"type": "json_object"}
    )
    # Parse returned JSON
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {"best_idx": -1, "confidence": 0.0, "rationale": f"parse_error: {type(e).__name__}: {e}"}
    # Safety: normalize fields
    if "best_idx" not in data: data["best_idx"] = -1
    if "confidence" not in data: data["confidence"] = 0.0
    if "rationale" not in data: data["rationale"] = ""
    return data

# MAIN
def main():
    # ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/data",
        help="Path to the data directory containing meta.json, bbox_meta.json and images/*.png",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model id (multimodal). e.g. gpt-4o, gpt-4o-mini",
    )
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.08,
        help="Padding ratio around each bbox for crop generation",
    )
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL = args.model
    PAD_RATIO = args.pad_ratio

    ensure_dir(os.path.join(DATA_DIR, "verify_vis"))

    # LOAD META
    with open(os.path.join(DATA_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(DATA_DIR, "bbox_meta.json"), "r") as f:
        bbox_meta = json.load(f)

    results = []

    for sample_idx, (sample_meta, sample_bbox_meta) in enumerate(zip(tqdm(meta), bbox_meta)):
        # LOAD IMAGE
        img_path = os.path.join(DATA_DIR, f"images/{sample_idx:03d}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # CANDIDATES
        candidates = sample_bbox_meta.get("objects", [])
        # Visualization (all boxes)
        vis = draw_bboxes(image, candidates, color=(255, 0, 0))
        vis_path = os.path.join(DATA_DIR, f"verify_vis/{sample_idx:03d}.png")
        vis.save(vis_path)

        sample_result = {
            "sample_idx": sample_idx,
            "image_path": img_path,
            "vis_path": vis_path,
            "objects": []
        }

        # FOR EACH OBJECT TO VERIFY
        for obj_idx, obj_meta in enumerate(sample_meta.get("objects", [])):
            obj_class = obj_meta["class"]
            obj_appearance = obj_meta.get("appearance", "")

            # FILTER CANDIDATES BY CLASS
            same_class_candidates = []
            same_class_indices = []
            for k, cand in enumerate(candidates):
                if cand.get("label") == obj_class:
                    same_class_candidates.append(cand)
                    same_class_indices.append(k)

            if not same_class_candidates:
                sample_result["objects"].append({
                    "obj_idx": obj_idx,
                    "class": obj_class,
                    "appearance": obj_appearance,
                    "match": {
                        "best_global_idx": -1,
                        "best_local_idx": -1,
                        "confidence": 0.0,
                        "rationale": "no candidate of the same class",
                    }
                })
                continue

            # BUILD CROPS FOR THESE CANDIDATES
            crops_b64 = []
            labels_for_crops = []
            for cand in same_class_candidates:
                bbox = cand["bbox"]
                crop = crop_with_padding(image, bbox, pad_ratio=PAD_RATIO)
                crops_b64.append(pil_to_base64_jpg(crop, quality=85))
                labels_for_crops.append(f"{cand.get('label','?')}, obj_id={cand.get('obj_id','?')}")

            # SELECT BEST CANDIDATE
            match = choose_best_bbox_with_openai(
                appearance_text=obj_appearance,
                image_crops_b64=crops_b64,
                labels_for_crops=labels_for_crops,
                model=MODEL,
                temperature=0.0
            )

            # MAP LOCAL BEST_IDX BACK TO GLOBAL CANDIDATE INDEX
            local_idx = int(match.get("best_idx", -1))
            if 0 <= local_idx < len(same_class_indices):
                global_idx = same_class_indices[local_idx]
                chosen = candidates[global_idx]
                best_box = chosen["bbox"]
                best_obj_id = chosen.get("obj_id", None)
            else:
                global_idx = -1
                best_box = None
                best_obj_id = None

            sample_result["objects"].append({
                "obj_idx": obj_idx,
                "class": obj_class,
                "appearance": obj_appearance,
                "match": {
                    "best_global_idx": global_idx,
                    "best_local_idx": local_idx,
                    "obj_id": best_obj_id,
                    "bbox": best_box,
                    "confidence": float(match.get("confidence", 0.0)),
                    "rationale": match.get("rationale", "")
                }
            })

        results.append(sample_result)

    # Save final mapping
    out_json = os.path.join(DATA_DIR, "verify_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved:")
    print(f"  - Visualizations: {DATA_DIR}/verify_vis/vis_XXX.png")
    print(f"  - Results JSON  : {out_json}")

if __name__ == "__main__":
    main()
