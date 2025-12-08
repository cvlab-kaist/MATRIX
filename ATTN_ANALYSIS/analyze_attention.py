import os 
import json 
import argparse 
import torch 
from diffusers import (
    CogVideoXPipeline, 
    CogVideoXImageToVideoPipeline,
    HunyuanVideoImageToVideoPipeline,
    HunyuanVideoTransformer3DModel,
    WanImageToVideoPipeline, 
    WanTransformer3DModel,
    DiffusionPipeline,
    AutoencoderKLWan
)
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video_2b import CogVideoXImg2VidPipeline
import cv2 
from utils.MATRIXWriter import MATRIXWriter
from diffusers.utils import export_to_video, load_image 
from difflib import SequenceMatcher
import random
import numpy as np
from PIL import Image
import re

def get_mask(masks_dir, image_id, prompt_text, mask_token, mask_path, H=30, W=52) :
    cur_mask_path = os.path.join(
        masks_dir, image_id, mask_token, mask_path 
    )
    mask = np.array(Image.open(cur_mask_path))
    mask = (mask > 0).astype(np.float32)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    return mask 

def find_token_span(decoded_tokens, phrase, threshold=0.8):
    phrase_tokens = phrase.split() 
    cleaned_tokens = [t.replace("_", "") for t in decoded_tokens]
    max_ratio = 0 
    best_span = None 
    for i in range(len(cleaned_tokens) - len(phrase_tokens) + 1):
        candidate = cleaned_tokens[i:i+len(phrase_tokens)]
        candidate_phrase = ' '.join(candidate)
        ratio = SequenceMatcher(None, ' '.join(phrase_tokens), candidate_phrase).ratio() 
        if ratio > max_ratio and ratio >= threshold:
            max_ratio = ratio 
            best_span = list(range(i, i + len(phrase_tokens)))
    return best_span 

def load_pipe(model, device):
    if model == "cogvideox_i2v_2b":
        pipe = CogVideoXImg2VidPipeline.from_pretrained(
            "NimVideo/cogvideox-2b-img2vid",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "cogvideox_i2v_5b":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "hunyuan-i2v":
        model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            model_id, transformer=transformer, 
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "wan-i2v":
        model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
    else:
        raise ValueError("Selected model is unavailable")
    
    if model != "wan-i2v":
        pipe.vae.enable_slicing() 
        pipe.vae.enable_tiling() 

    return pipe

def aggregate_mask_sequences(args, tokenizer, video_filename, prompt_text, entities):
    if tokenizer is not None:
        token_ids = tokenizer(prompt_text, return_tensors="pt")
        decoded = tokenizer.convert_ids_to_tokens(token_ids["input_ids"][0])
    cur_masks_path = os.path.join(args.masks_dir, video_filename)
    if not os.path.exists(cur_masks_path) : return 
    mask_tokens = sorted(os.listdir(cur_masks_path))

    mask_dict = {}
    for mask_token in mask_tokens:
        mask_dict[mask_token] = {}
        mask_paths = sorted(os.listdir(os.path.join(cur_masks_path, mask_token)))
        masks = [] 
        for idx in range(args.num_frames // 4 + 1):
            if idx == 0 : 
                mask = get_mask(args.masks_dir, video_filename, prompt_text, mask_token, mask_paths[idx], H=args.height // 16, W=args.width // 16)
                masks.append(mask)
            else:
                mask1 = get_mask(args.masks_dir, video_filename, prompt_text, mask_token, mask_paths[4 * idx - 3], H=args.height // 16, W=args.width // 16)
                mask2 = get_mask(args.masks_dir, video_filename, prompt_text, mask_token, mask_paths[4 * idx - 2], H=args.height // 16, W=args.width // 16)
                mask3 = get_mask(args.masks_dir, video_filename, prompt_text, mask_token, mask_paths[4 * idx - 1], H=args.height // 16, W=args.width // 16)
                mask4 = get_mask(args.masks_dir, video_filename, prompt_text, mask_token, mask_paths[4 * idx], H=args.height // 16, W=args.width // 16)

                union_mask_list =[mask1, mask2, mask3, mask4]
                stacked = torch.stack(union_mask_list, dim=0)
                union = stacked.max(dim=0).values 
                masks.append(union)
        masks = torch.cat(masks, dim=0).cuda() 
        best_span = None 
        save_type = "verb"
        
        matched_entity = None 
        for entity in entities.values():
            desc = entity.get("description", "None")
            subject = entity.get("subject", "None")
            if mask_token == subject or mask_token in desc.split():
                matched_entity = entity 
                break 
        
        if matched_entity is not None: 
            desc = matched_entity.get("description", "None")
            subject = matched_entity.get("subject", "None")
            if tokenizer is not None:
                span_desc = find_token_span(decoded, desc) if desc != "None" else None 
                span_subj = find_token_span(decoded, subject) if subject != "None" else None 
                best_span = span_desc or span_subj 
            save_type = "noun"
        else:
            if tokenizer is not None:
                best_span = find_token_span(decoded, mask_token)

        if best_span is None: 
            print("[ERROR] No valid span found. Skipping.")
        mask_dict[mask_token]["save_type"] = save_type 
        mask_dict[mask_token]["masks"] = masks 
        mask_dict[mask_token]["best_span"] = best_span 
    return mask_dict 

def main(args):
    device = args.device 
    output_dir = args.output_dir 
    model = args.model 
    prompt_path = args.prompt_path 

    pipe = load_pipe(model, device)
    pipe.enable_sequential_cpu_offload()

    tokenizer = pipe.tokenizer

    with open(prompt_path, "r") as f:
        data = json.load(f)

    frame_dir = os.path.join(args.data_dir, "images")
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, entry in enumerate((data)):
        video_filename = entry["video_id"]

        for i in range(len(entry["prompts"])):
            prompt_text = entry["prompts"][i]["prompt"]
            prompt_image = load_image(os.path.join(frame_dir, f"{video_filename}.png"))
        
        max_len = 50
        safe_prompt = re.sub(r'[^a-zA-Z0-9 _-]', '', prompt_text)  
        safe_prompt = safe_prompt[:max_len]
        os.makedirs(f"{output_dir}/{video_filename}", exist_ok=True)

        entities = entry["entities"]
        seed = 42 
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        generator = torch.manual_seed(seed)
        torch.cuda.empty_cache()
   
        if args.attn_whole:
            if args.model == "cogvideox_i2v_2b":
                num_layers = pipe.tranasformer.config.num_layers # 30
                attn_whole = MATRIXWriter(
                model="cogvideox",
                num_inference_steps=args.num_inference_steps,
                num_layers=num_layers, 
                text_len=args.text_len,
                latent_h=args.height // 16,
                latent_w=args.height // 16,
            )
            elif args.model == "hunyuan-i2v":
                num_layers = pipe.transformer.config.num_layers 
                num_layers += pipe.transformer.config.num_single_layers

                attn_whole = MATRIXWriter(
                    model="hunyuan",
                    num_inference_steps=args.num_inference_steps,
                    num_layers=num_layers,
                    text_len=args.text_len,
                    latent_h=args.height // 16, 
                    latent_w=args.width // 16,
                )
            elif args.model == "wan-i2v":
                num_layers = pipe.transformer.config.num_layers 
                attn_whole = MATRIXWriter(
                    model="wan",
                    num_inference_steps=args.num_inference_steps, 
                    num_layers=num_layers,
                    text_len=args.text_len,
                    latent_h=args.height // 16,
                    latent_w=args.width // 16,
                )
            else: 
                num_layers = pipe.transformer.config.num_layers # 42
                attn_whole = MATRIXWriter(
                    model="cogvideox",
                    num_inference_steps=args.num_inference_steps,
                    num_layers=num_layers, 
                    text_len=args.text_len, 
                    latent_h=args.height // 16,
                    latent_w=args.width // 16,

                )
            mask_dict = aggregate_mask_sequences(args, tokenizer, video_filename, prompt_text, entities)

        if args.attn_whole and mask_dict is not None:
            video_generate = pipe(
                prompt=prompt_text,
                image=prompt_image,
                height=args.height,
                width=args.width,
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                mask_dict=mask_dict if mask_dict else None,
                attn_whole=attn_whole if args.attn_whole else None,
            ).frames[0]

            export_to_video(video_generate, f"{output_dir}/{video_filename}/{safe_prompt}.mp4", fps=8)
        else:
            print(f"{prompt_text} has no available masks \n")

if __name__ == "__main__": 
    args = argparse.ArgumentParser() 
    args.add_argument(
        "--data_dir", type=str, default="/path/to/dir/"
    )
    args.add_argument(
        "--output_dir", type=str, default="generated_results"
    )
    args.add_argument(
        "--prompt_path", type=str, default="/path/to/json"
    )
    args.add_argument(
        "--model", type=str, default="cogvideox_i2v_5b",
        choices=[
            "cogvideox_i2v_2b", "cogvideox_i2v_5b", "hunyuan-i2v", "wan-i2v"
        ]
    )
    args.add_argument("--num_inference_steps", type=int, default=50)
    args.add_argument("--height", type=int, default=480)
    args.add_argument("--width", type=int, default=720)
    args.add_argument("--num_frames", type=int, default=49)
    args.add_argument("--device", type=str, default='cuda:0')
    args.add_argument("--guidance_scale", type=int, default=6)
    args.add_argument("--attn_whole", action="store_true")
    args.add_argument("--text_len", type=int, default=226)
    args.add_argument("--masks_dir", type=str, default="/path/to/masks_dir")

    args = args.parse_args() 
    main(args)