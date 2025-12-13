import torch
import torch.nn as nn
from PIL import Image

from modelscope import dataset_snapshot_download
import argparse 

from safetensors.torch import load_file
import json 
import os 
from pathlib import Path 
from typing import List, Dict 
from tqdm import tqdm

from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

def build_pipeline(
    checkpoint_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> WanVideoPipeline:
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P"
    device = "cuda"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                offload_device="cpu",
            ),
        ],
    )
    pipe.load_lora(pipe.dit, args.checkpoint_path, alpha=0.5)

    state = load_file(checkpoint_path)
    in_channels = 36 + 20
    out_channels = 5120 
    kernel_size = (1, 2, 2)
    stride = (1, 2, 2)

    new_patch_embed = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    with torch.no_grad():
        new_patch_embed.weight.copy_(state["patch_embedding.weight"])
        if "patch_embedding.bias" in state and new_patch_embed.bias is not None:
            new_patch_embed.bias.copy_(state["patch_embedding.bias"])
    
    pipe.dit.patch_embedding = new_patch_embed.to(dtype=dtype, device=pipe.device)
    pipe.enable_vram_management() 
    pipe.prompter.tokenizer.tokenizer.add_special_tokens(
        {"additional_special_tokens" :["<id1>", "<id2>", "<id3>", "<id4>", "<id5>"]}
    )

    return pipe 

def run_inference(
    checkpoint_path: str,
    eval_data_root: str, 
    output_dir: str,
    num_frames: int=81,
    fps: int=15,
    dtype: torch.dtype=torch.bfloat16, 
):
    os.makedirs(output_dir, exist_ok=True)
    pipe = build_pipeline(
        checkpoint_path = checkpoint_path, 
        dtype = dtype
    )

    with open(f"{eval_data_root}/eval_metas.json", "r") as f:
        data = json.load(f)
    
    for idx, eval_data in tqdm(enumerate(data)):
        video_id = eval_data["video_id"]
        prompt = eval_data["prompt"]
        input_image_path = os.path.join(args.eval_data_root, "images", f"{video_id}.png")
        mask_image_path = os.path.join(args.eval_data_root, "masks", f"{video_id}/merged.png")
        input_image = VideoData(input_image_path, height=480, width=832)[0]
        mask_image = VideoData(mask_image_path, height=480, width=832)[0]
        save_path = os.path.join(output_dir, f"{video_id}.mp4") 

        video = pipe(
            prompt=prompt,
            input_image=input_image,
            mask_image=mask_image,
            seed=42, 
            tiled=True,   
            cfg_scale=6.0,
            num_frames=num_frames,
        )
        save_video(video, save_path, fps=15, quality=5)

def get_parser():
    parser = argparse.ArgumentParser(
        description = "I2V inference script (image+mask -> video)"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_data_root", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser() 

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    run_inference(
        checkpoint_path=args.checkpoint_path,
        eval_data_root=args.eval_data_root,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        fps=args.fps,
        dtype=torch_dtype,
    )