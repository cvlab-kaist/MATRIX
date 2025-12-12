
import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any
import numpy as np
import cv2

def visualize_mask(frame, mask, color=np.array([255, 0, 0]), alpha=0.5):
    if mask is None:
        return frame.astype(np.uint8)
    coords = mask > 0
    frame[coords] = color * alpha + frame[coords] * (1 - alpha)
    edge = cv2.Canny(mask, 0, 1)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    coords = edge > 0
    frame[coords] = color
    return frame.astype(np.uint8)

def visualize_points(
    frame, 
    points, 
    pos_color=np.array([0, 0, 255]), 
    neg_color=np.array([0, 255, 0]),
    radius=15,
):
    for point, label in points:
        if label == 1: # positive point
            x, y = point
            cv2.circle(frame, (int(x), int(y)), radius, pos_color.tolist(), -1)
        elif label == 0: # negative point
            x, y = point
            cv2.circle(frame, (int(x), int(y)), radius, neg_color.tolist(), -1)
    return frame.astype(np.uint8)

def refine_bbox(ori_h, ori_w, new_h, new_w, bbox):
    scale_height = new_h / ori_h
    scale_width = new_w / ori_w
    x1, y1, x2, y2 = bbox
    return (int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pil_to_base64_jpg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def draw_bboxes(image: Image.Image, candidates: List[Dict[str, Any]], color=(0, 255, 0)) -> Image.Image:
    """
    Draw candidate boxes with indices on a copy of the image.
    candidates[i] should have 'bbox' [x1,y1,x2,y2] and 'label'.
    """
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    # Try to load a default font; fallback if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, obj in enumerate(candidates):
        x1, y1, x2, y2 = obj["bbox"]
        # Clamp to image bounds
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(image.width - 1, int(x2)); y2 = min(image.height - 1, int(y2))
        # Rectangle
        draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=4)
        # Label index box
        tag = f"{i}"
        tw, th = draw.textbbox((0,0), tag, font=font)[2:]
        pad = 4
        draw.rectangle([x1, y1 - th - 2*pad, x1 + tw + 2*pad, y1], fill=(0,0,0))
        draw.text((x1 + pad, y1 - th - pad), tag, fill=(255,255,255), font=font)
    return vis

def crop_with_padding(image: Image.Image, bbox, pad_ratio: float = 0.08) -> Image.Image:
    """
    Crop an area around bbox with padding (as a fraction of box size),
    clamped to image bounds.
    """
    x1, y1, x2, y2 = bbox
    x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * pad_ratio
    py = h * pad_ratio
    cx1 = int(max(0, x1 - px))
    cy1 = int(max(0, y1 - py))
    cx2 = int(min(image.width,  x2 + px))
    cy2 = int(min(image.height, y2 + py))
    return image.crop((cx1, cy1, cx2, cy2))