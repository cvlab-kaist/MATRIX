# IMPORT
from os import makedirs
from os.path import join, exists
import json
import argparse
import numpy as np
import imageio.v2 as iio
import cv2

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    type=str,
    default='/path/to/data',
    help='Path to the data directory',
)
parser.add_argument(
    '--video_dir',
    type=str,
    default=None,
    help='Path to the video directory',
)
parser.add_argument(
    '--eval_dir',
    type=str,
    default=None,
    help='Path to the eval directory',
)

args = parser.parse_args()

DATA_DIR = args.data_dir
VIDEO_DIR = args.video_dir
EVAL_DIR = args.eval_dir

# CREATE OUTPUT DIRECTORY
makedirs(join(EVAL_DIR, 'track_vis'), exist_ok=True)
makedirs(join(EVAL_DIR, 'eval_metas'), exist_ok=True)

color_list = [
    {'color': 'red', 'value': np.array([255, 0, 0], dtype=np.uint8)},
    {'color': 'green', 'value': np.array([0, 255, 0], dtype=np.uint8)},
    {'color': 'blue', 'value': np.array([0, 0, 255], dtype=np.uint8)},
    {'color': 'yellow', 'value': np.array([255, 255, 0], dtype=np.uint8)},
    {'color': 'magenta', 'value': np.array([255, 0, 255], dtype=np.uint8)},
    {'color': 'cyan', 'value': np.array([0, 255, 255], dtype=np.uint8)},
]

with open(join(DATA_DIR, 'meta.json'), 'r') as f:
    meta = json.load(f)
for sample_idx, sample_meta in enumerate(meta):
    # GET NUMBER OF OBJECTS
    n_objects = len(sample_meta['objects'])

    eval_meta_dict = {
        'video_id': f'{sample_idx:03d}',
        'prompt': sample_meta['video_text_prompt'],
        'entity': [x['class'] for x in sample_meta['objects']],
        'appearance': [x['appearance'] for x in sample_meta['objects']],
        'colors': [color_list[i]['color'] for i in range(n_objects)]
    }

    # SAVE EVAL META
    with open(join(EVAL_DIR, 'eval_metas', f'{sample_idx:03d}.json'), 'w') as f:
        json.dump(eval_meta_dict, f, indent=4)

    video_path = join(VIDEO_DIR, f"{sample_idx:03d}.mp4")
    if not exists(video_path):
        print(f'[NOT FOUND] {video_path}')
        continue

    reader = iio.get_reader(video_path)
    writer = iio.get_writer(join(EVAL_DIR, 'track_vis', f"{sample_idx:03d}.mp4"), fps=8)
    for frame_idx, frame in enumerate(reader):
        for obj_idx, obj_info in enumerate(sample_meta['objects']):
            obj_id = f'id{obj_idx + 1}'
            obj_mask = iio.imread(join(EVAL_DIR, 'tracks', f"{sample_idx:03d}", obj_id, f"{frame_idx:03d}.png"))
            obj_mask = (obj_mask > 0).astype(np.uint8)

            if frame_idx == 0:
                coords = np.where(obj_mask > 0)
                frame[coords] = 0.5 * color_list[obj_idx]['value'] + 0.5 * frame[coords]
            if obj_mask.sum() > 0:
                # GET BOUNDING BOX
                bbox = cv2.boundingRect(obj_mask)
                # DRAW BOUNDING BOX
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    color_list[obj_idx]['value'].tolist(),
                    2
                )
        writer.append_data(frame)
    reader.close()
    writer.close()
    print(f'Processed video {sample_idx:03d}')