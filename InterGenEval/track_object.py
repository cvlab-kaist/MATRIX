# IMPORT
from os import makedirs, listdir
from os.path import join, exists
import argparse
import imageio.v2 as iio
import numpy as np
import cv2
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    type=str,
    default='/path/to/data',
    help='Path to the data directory'
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help='Path to the video directory'
)
parser.add_argument(
    '--eval_dir',
    type=str,
    default=None,
    help='Path to the evaluation directory'
)
parser.add_argument(
    "--center_crop",
    action='store_true',
    help='Whether to center crop the image to 1024x1024'
)
args = parser.parse_args()

DATA_DIR = args.data_dir
VIDEO_DIR = args.video_dir
EVAL_DIR = args.eval_dir

print('ARGS:')
print(f'DATA_DIR: {DATA_DIR}')
print(f'VIDEO_DIR: {VIDEO_DIR}')
print(f'EVAL_DIR: {EVAL_DIR}')
print(f'CENTER_CROP: {args.center_crop}')

# CREATE OUTPUT DIRECTORY
makedirs(join(EVAL_DIR, 'tracks'), exist_ok=True)

# LOAD MODEL
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")

# LOAD VIDEO IDS
video_paths = sorted(listdir(VIDEO_DIR))

for video_path in video_paths:
    video_id = video_path[:-4]
    video_path = join(VIDEO_DIR, video_path)
    if not video_path.endswith('.mp4'):
        continue

    reader = iio.get_reader(video_path)
    w, h = reader.get_meta_data()['size']
    ratio = w / h
    n_frames = reader.count_frames()

    # INITIALIZE INFERENCE STATE
    inference_state = predictor.init_state(video_path)

    obj_mask_paths = sorted(listdir(join(DATA_DIR, 'masks', video_id)))

    for obj_mask_path in obj_mask_paths:
        obj_id = obj_mask_path[:-4]
        if obj_id == 'merged':
            continue
        if exists(join(EVAL_DIR, 'tracks', video_id, obj_id, f'{n_frames - 1:03d}.png')):
            print(f'[ALREADY DONE] {video_id} {obj_id}')
            continue # skip if already done
        obj_mask = (iio.imread(join(DATA_DIR, 'masks', video_id, obj_mask_path)) > 0).astype(np.uint8)
        if args.center_crop:
            oh, ow = obj_mask.shape
            oratio = ow / oh
            if oratio > ratio: # crop width
                nh = h
                nw = int(oratio * nh)
            else: # crop height
                nw = w
                nh = int(nw / oratio)
            obj_mask = (cv2.resize(obj_mask, (nw, nh), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
            # center crop
            sh = (nh - h) // 2
            sw = (nw - w) // 2
            obj_mask = obj_mask[sh:sh+h, sw:sw+w]
        else:
            obj_mask = (cv2.resize(obj_mask, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

        if obj_mask.ndim == 3:
            print(f'[SKIP] {video_id} {obj_id} (3 channels)')
            continue
        makedirs(join(EVAL_DIR, 'tracks', video_id, obj_id), exist_ok=True)

        with torch.inference_mode():
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=obj_mask,
            )

            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
                out_mask = (out_mask_logits > 0).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
                iio.imwrite(join(EVAL_DIR, 'tracks', video_id, obj_id, f'{out_frame_idx:03d}.png'), out_mask)
        
        predictor.reset_state(inference_state)