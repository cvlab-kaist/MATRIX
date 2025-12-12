# IMPORT
from os import makedirs
from os.path import join
import json
import argparse
import numpy as np
from PIL import Image
import imageio.v2 as iio
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

def _visualize_mask(image, mask, color, alpha=0.5):
    vis = image.copy()
    coords = np.where(mask > 0)
    vis[coords] = alpha * color + (1 - alpha) * vis[coords]
    return vis

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', 
    type=str, 
    default='/path/to/data',
    help='Path to the data directory'
)
args = parser.parse_args()

DATA_DIR = args.data_dir

# LOAD META
with open(join(DATA_DIR, 'verify_results.json'), 'r') as f:
    bbox_meta = json.load(f)

# LOAD SAM2 MODEL
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

# PROCESS SAMPLES
for sample_bbox_meta in tqdm(bbox_meta):
    # CREATE OUTPUT DIRECTORY
    makedirs(join(DATA_DIR, 'masks', f'{sample_bbox_meta["sample_idx"]:03d}'), exist_ok=True)
    makedirs(join(DATA_DIR, 'masks_vis', f'{sample_bbox_meta["sample_idx"]:03d}'), exist_ok=True)

    # LOAD IMAGE
    image = Image.open(join(DATA_DIR, 'images', f'{sample_bbox_meta["sample_idx"]:03d}.png')).convert("RGB")
    image = np.array(image)

    # SET IMAGE
    predictor.set_image(image)

    # PROCESS EACH BBOX
    for bbox_info in sample_bbox_meta['objects']:
        if bbox_info['match']['best_global_idx'] < 0:
            continue
        # EXTRACT BBOX INFO
        obj_idx = bbox_info['obj_idx'] + 1
        box = np.array(bbox_info['match']['bbox'])

        # PREDICT MASK
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False,
        )

        # SAVE MASK
        mask = masks[0].astype(np.uint8) * 255
        iio.imwrite(join(DATA_DIR, 'masks', f'{sample_bbox_meta["sample_idx"]:03d}', f'id{obj_idx}.png'), mask)

        # SAVE VISUALIZATION
        vis = _visualize_mask(image, masks[0], np.array([255, 0, 0], dtype=np.uint8), alpha=0.5)
        iio.imwrite(join(DATA_DIR, 'masks_vis', f'{sample_bbox_meta["sample_idx"]:03d}', f'id{obj_idx}.png'), vis)