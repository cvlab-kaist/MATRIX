import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union, List
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
# from typing import Optional, List
from typing_extensions import Literal
import torch.nn.functional as F 
from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader



class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_IDMaskImageEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_MaskImageEmbedderVAE(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_CfgMerger(),
        ]

        self.model_fn = model_fn_ours

    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)
    
    def aggregate_id_masks(self, input_masks, s=1/8):
        id_masks = []
        for msk in input_masks:             
            if msk.dim() == 4:
                msk = msk.max(dim=1).values    # [1, H, W]
            else:  # [C, H, W]
                msk = msk.max(dim=0).values    # [H, W]
                msk = msk.unsqueeze(0)         # [1, H, W]
            msk = msk.clamp(0, 1)
            msk = (msk > 0.5).float()
            msk = F.interpolate(msk.unsqueeze(0), scale_factor=(s, s), mode="nearest")
            id_masks.append(msk)

        id_masks = torch.stack(id_masks, dim=1).squeeze(0)  # [F, 1, H', W']
        return id_masks

    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        id1 = self.aggregate_id_masks(inputs["id1_masks"], s=1/16) 
        dummy = id1.new_zeros(id1.shape)
        id2 = self.aggregate_id_masks(inputs["id2_masks"], s=1/16) if inputs["id2_masks"] else dummy
        id3 = self.aggregate_id_masks(inputs["id3_masks"], s=1/16) if inputs["id3_masks"] else dummy 
        id4 = self.aggregate_id_masks(inputs["id4_masks"], s=1/16) if inputs["id4_masks"] else dummy
        id5 = self.aggregate_id_masks(inputs["id5_masks"], s=1/16) if inputs["id5_masks"] else dummy 
        id_first_masks = torch.stack([id1, id2, id3, id4, id5])
        id_first_masks = rearrange(id_first_masks, "b f c h w -> b c f h w")
        inputs["whole_id_masks"] = id_first_masks

        noise_pred, v2t_mask_pred, v2v_mask_pred = self.model_fn(
            **inputs,
            timestep=timestep,
        )

        id_num = 0
        id1_masks = self.aggregate_id_masks(inputs["id1_masks"])
        dummy = id1_masks.new_zeros((id1_masks.shape))

        id_num += 1
        if inputs["id2_masks"] :
            id2_masks = self.aggregate_id_masks(inputs["id2_masks"]) 
            id_num += 1
        else : 
            id2_masks = dummy 
        if inputs["id3_masks"]: 
            id3_masks = self.aggregate_id_masks(inputs["id3_masks"])
            id_num += 1
        else :
            id3_masks = dummy 
        if inputs["id4_masks"]: 
            id4_masks = self.aggregate_id_masks(inputs["id4_masks"]) 
            id_num += 1
        else: 
            id4_masks = dummy 
        if inputs["id5_masks"]:
            id5_masks = self.aggregate_id_masks(inputs["id5_masks"])
            id_num += 1
        else:
            id5_masks = dummy 
        id_masks = torch.stack([id1_masks, id2_masks, id3_masks, id4_masks, id5_masks])
        id_masks = rearrange(id_masks, "b f c h w -> b c f h w")

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)

        # 2) SGA / SPA mask loss
        L_sga = L_spa = 0.0

        if v2t_mask_pred is not None:
            logits = v2t_mask_pred[:id_num].float()
            target = id_masks[:id_num, :, :logits.shape[2]].float()
            pos = (target == 1).float().sum()
            neg = (target == 0).float().sum()
            pos_weight = (neg / pos.clamp(min=1)).clamp(max=10.0)

            bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            L_sga_bce = bce(logits, target)

            probs = torch.sigmoid(logits)
            # L_sga_mse = F.mse_loss(probs, target)
            L_sga = L_sga_bce 
        if v2v_mask_pred is not None:           
            logits = v2v_mask_pred[:id_num].float()
            target = id_masks[:id_num, :, :logits.shape[2]].float()

            pos = (target == 1).float().sum()
            neg = (target == 0).float().sum()
            pos_weight = (neg / pos.clamp(min=1)).clamp(max=10.0)

            bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            L_spa_bce = bce(logits, target)

            probs = torch.sigmoid(logits)
            # L_spa_mse = F.mse_loss(probs, target)

            L_spa = L_spa_bce 

        w = self.scheduler.training_weight(timestep)

        lambda_sga = 0.01  
        lambda_spa = 0.005 

        if v2t_mask_pred is not None:
            loss = loss + w * lambda_sga * L_sga

        if v2v_mask_pred is not None:
            loss = loss + w * lambda_spa * L_spa

        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            from ..models.longcat_video_dit import LayerNorm_FP32, RMSNorm_FP32
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                    LayerNorm_FP32: AutoWrappedModule,
                    RMSNorm_FP32: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
    
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")

        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        pipe.prompter.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens" :["<id1>", "<id2>", "<id3>", "<id4>", "<id5>"]})        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed=42,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "mask_image" : mask_image,
            "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi, _, _ = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega, _, _ = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)

        token_ids = pipe.prompter.tokenizer.tokenizer(prompt, return_tensors="pt")["input_ids"][0] 
        token_ids = [pipe.prompter.tokenizer.tokenizer.decode(token_id) for token_id in token_ids]
        return {"context": prompt_emb, "token_ids":token_ids}

class WanVideoUnit_IDEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            onload_model_names=("text_encoder")
        )
    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        token_ids = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        token_ids = [pipe.prompter.tokenizer.decode_prompt(token_id) for token_id in token_ids] 
        
class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}

class WanVideoUnit_IDMaskImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("id1_masks", "id2_masks", "id3_masks", "id4_masks", "id5_masks", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )
    def process(self, pipe: WanVideoPipeline, id1_masks, id2_masks, id3_masks, id4_masks, id5_masks, num_frames, height, width, tiled, tile_size, tile_stride):
        return_dict = {}
        if id1_masks is not None: 
            id1_masks = [pipe.preprocess_image(msk.resize((width, height))).to(pipe.device) for msk in id1_masks]
            return_dict["id1_masks"] = id1_masks 
        if id2_masks is not None:
            id2_masks = [pipe.preprocess_image(msk.resize((width, height))).to(pipe.device) for msk in id2_masks]
            return_dict["id2_masks"] = id2_masks 
        if id3_masks is not None:
            id3_masks = [pipe.preprocess_image(msk.resize((width, height))).to(pipe.device) for msk in id3_masks]
            return_dict["id3_masks"] = id3_masks 
        if id4_masks is not None:
            id4_masks = [pipe.preprocess_image(msk.resize((width, height))).to(pipe.device) for msk in id4_masks]
            return_dict["id4_masks"] = id4_masks 
        if id5_masks is not None:
            id5_masks = [pipe.preprocess_image(msk.resize((width, height))).to(pipe.device) for msk in id5_masks]
            return_dict["id5_masks"] = id5_masks 
        return return_dict 
    

class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}

class WanVideoUnit_MaskImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("mask_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
    def process(self, pipe: WanVideoPipeline, mask_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if mask_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        mask_image = pipe.preprocess_image(mask_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, 1:] = 0
        
        vae_input = torch.concat([mask_image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(mask_image.device)], dim=1)
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8) 
        msk = msk.transpose(1, 2)[0]
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype = pipe.torch_dtype, device = pipe.device)
        return {"mask_image" : y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["y"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value
        
def model_fn_ours(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    y: Optional[torch.Tensor] = None, # first image 
    mask_image : Optional[torch.Tensor] = None, # first mask image
    id1_mask : List = None,
    id2_masks : List = None, 
    id3_masks : List = None, 
    id4_masks : List = None, 
    id5_masks : List = None, 
    whole_id_masks = None,
    token_ids = None,
    use_unified_sequence_parallel: bool = False,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    v2t_layers=None, 
    v2v_layers=None,
    sga_loss=False,
    spa_loss=False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    # Mask Image Embedding
    if mask_image is not None and dit.require_vae_embedding:
        x = torch.cat([x, mask_image], dim=1)

    # Patchify
    x = dit.patch_embedding(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    def create_custom_forward(module):
        def custom_forward(*inputs, **kwargs):
            return module(*inputs, **kwargs)
        return custom_forward
    
    def create_custom_forward_vap(block, vap):
        def custom_forward(*inputs):
            return vap(block, *inputs)
        return custom_forward
    
    if v2t_layers is None :
        v2t_layers = []
    if v2v_layers is None:
        v2v_layers = []
    v2t_qks = []
    v2f_qks = []
    f2v_qks = []
    for block_id, block in enumerate(dit.blocks):
        # Block
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False, 
                    is_v2t_layer=True if v2t_layers is not None and block_id in v2t_layers else False, 
                    is_v2v_layer=True if v2v_layers is not None and block_id in v2v_layers else False,
                    sga_loss=sga_loss,
                    spa_loss=spa_loss,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
                is_v2t_layer=True if v2t_layers is not None and block_id in v2t_layers else False, 
                is_v2v_layer=True if v2v_layers is not None and block_id in v2v_layers else False,
                sga_loss=sga_loss,
                spa_loss=spa_loss,
            )
        else:
            x = block(
                x, context, t_mod, freqs,
                is_v2t_layer=True if v2t_layers is not None and block_id in v2t_layers else False, 
                is_v2v_layer=True if v2v_layers is not None and block_id in v2v_layers else False,
                sga_loss=sga_loss,
                spa_loss=spa_loss,
            )
    
        if block_id in v2t_layers and sga_loss:
            v2t_qks.append(block.cross_attn.v2t_weights)
            del block.cross_attn.v2t_weights
        if block_id in v2v_layers and spa_loss:
            # if hasattr(block.self_attn, "v2f_weights"):
            #     v2f_qks.append(block.self_attn.v2f_weights)
            #     del block.self_attn.v2f_weights
            if hasattr(block.self_attn, "f2v_weights"):
                f2v_qks.append(block.self_attn.f2v_weights)
                del block.self_attn.f2v_weights 
    
    if sga_loss :
        id_list = ["<id1>", "<id2>", "<id3>", "<id4>", "<id5>"]
        if len(v2t_qks) == 0: 
            v2t_pred = None
        else:
            v2t_pred = []
            for _id in id_list:
                indices = [j for j, tok in enumerate(token_ids) if tok == _id and j < 255]
                if len(indices) == 0: 
                    dummy = torch.zeros(v2t_qks[0].shape[0], (v2t_qks[0].shape[1] // (30 * 52) - 1) * 4 + 1 , 60, 104).to(x.device)
                    v2t_pred.append(dummy)
                    continue 
                v2t_pred_id = [qks[:, :, indices] for qks in v2t_qks]
                v2t_pred_id = [rearrange(qks, "b (f h w) n -> (b n) f h w", h = 30, w =52).mean(0) for qks in v2t_pred_id]
                
                v2t_pred_id = torch.stack(v2t_pred_id).mean(0, keepdim=True)
                v2t_pred_id = dit.seg_head.decode(v2t_pred_id, device=x.device)
                v2t_pred.append(v2t_pred_id.squeeze(0))
            v2t_pred = torch.stack(v2t_pred, dim=0)
            torch.cuda.empty_cache()
    else:
        v2t_pred = None
    
    if spa_loss : 
        f2v_pred = []
        for _id in range(5):
            id_first_mask = whole_id_masks[_id, :, 0, :, :].flatten().bool() 
            if id_first_mask.sum() == 0 or torch.isnan(id_first_mask).any():
                dummy = torch.zeros(f2v_qks[0].shape[0], (f2v_qks[0].shape[2]  // (30 * 52) - 1) * 4 + 1, 60, 104).to(x.device)
                f2v_pred.append(dummy)
                continue
            f2v_pred_id = [qks[:, id_first_mask, :] for qks in f2v_qks]
            f2v_pred_id = [rearrange(qks, "b n (f h w) -> b n f h w", h=30, w = 52).mean(1) for qks in f2v_pred_id]
            f2v_pred_id = torch.stack(f2v_pred_id).mean(0, keepdim=True) 
            f2v_pred_id = dit.seg_head.decode(f2v_pred_id, device=x.device)
            f2v_pred.append(f2v_pred_id.squeeze(0))
        f2v_pred = torch.stack(f2v_pred, dim=0)
        torch.cuda.empty_cache()
    else:
        v2f_pred = None 
        f2v_pred = None 
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    x = dit.unpatchify(x, (f, h, w))
    return x, v2t_pred, f2v_pred

