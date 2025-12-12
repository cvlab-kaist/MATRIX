# IMPORT
from os import makedirs
from os.path import join
import cv2
import json
import torch
import argparse
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', 
    type=str, 
    default='/path/to/dir',
    help='Path to the data directory'
)
args = parser.parse_args()

DATA_DIR = args.data_dir

# CREATE OUTPUT DIRECTORY
makedirs(join(DATA_DIR, 'bbox_vis'), exist_ok=True)

# LOAD META
with open(join(DATA_DIR, 'meta.json'), 'r') as f:
    meta = json.load(f)

# SET DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base",
).to(device)
grounding_dino.eval()

output_list = []
for sample_idx, sample_meta in enumerate(tqdm(meta)):
    # PREPARE PROMPT
    objects = [obj_info["class"] + "." for obj_info in sample_meta["objects"]]
    objects = list(set(objects))
    text_prompt = " ".join(objects)

    image = Image.open(join(DATA_DIR, f"images/{sample_idx:03d}.png")).convert("RGB")

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    # INFERENCE
    with torch.no_grad():
        outputs = grounding_dino(**inputs)

    # POST-PROCESSING
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    results = results[0]

    output_dict = {
        "sample_id": sample_idx,
        "text_prompt": text_prompt,
        "objects": [],
    }

    for obj_idx in range(len(results['scores'])):
        score = results['scores'][obj_idx].item()
        bbox = results['boxes'][obj_idx].tolist()
        label = results['labels'][obj_idx]

        output_dict['objects'].append({
            'label': label,
            'score': score,
            'bbox': bbox,
            'obj_id': obj_idx,
        })

        # BBOX VISUALIZATION
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[2]), int(bbox[3])
        image = cv2.imread(join(DATA_DIR, f"images/{sample_idx:03d}.png"))
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=5)
        cv2.imwrite(join(DATA_DIR, f'bbox_vis/{sample_idx:03d}_{obj_idx}_{label}_{score:.2f}.png'), image)

    output_list.append(output_dict)

# SAVE OUTPUT
with open(join(DATA_DIR, 'bbox_meta.json'), 'w') as f:
    json.dump(output_list, f, indent=4)