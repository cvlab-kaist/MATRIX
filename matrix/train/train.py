import torch, os, json
import torch.nn as nn
from typing import Dict, Any, List
import itertools
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_matrix import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import (
    DiffusionTrainingModule, 
    ModelLogger, 
    launch_training_task, 
    get_parser, 
    launch_data_process_task
)
from diffsynth.trainers.unified_dataset import (
    MATRIXDataset, 
    ImageCropAndResize, 
    ToAbsolutePath, 
    LoadImage, 
    SequencialProcess
)

from diffsynth.trainers.seg_head import SegHead

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_lora_target_modules(
    base_targets: str,
    v2t_layers: List[int] | None, 
    v2v_layers: List[int] | None,
) -> str:
    if (not v2t_layers) and (not v2v_layers):
        return base_targets 
    
    base = base_targets.split(",")
    modules: List[str] = [] 

    if v2t_layers :
        # cross-attn 
        for l, t in itertools.product(v2t_layers, base):
            modules.append(f"blocks.{l}.cross_attn.{t}")
    if v2v_layers :
        # self-attn 
        for l, t in itertools.product(v2v_layers, base):
            modules.append(f"blocks.{l}.self_attn.{t}")
    
    return ",".join(modules)

def extend_patch_embedding(dit: nn.Module) -> None:
    orig_patch = dit.patch_embedding 
    in_ch = orig_patch.in_channels 
    out_ch = orig_patch.out_channels 
    k = orig_patch.kernel_size 
    s = orig_patch.stride 
    p = orig_patch.padding 

    new_conv = nn.Conv3d(in_ch + 20, out_ch, kernel_size=k, stride=s, padding=p)
    
    with torch.no_grad():
        new_conv.weight.zero_() 
        new_conv.weight[:, :in_ch].copy_(orig_patch.weight)
        if orig_patch.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(orig_patch.bias)
    
    dit.patch_embedding = new_conv.to(type=torch.bfloat16)
    dit.patch_embedding.requires_grad_(True)

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, 
        model_id_with_origin_paths=None, 
        audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, 
        lora_target_modules="q,k,v,o,ffn.0,ffn.2", 
        lora_rank=32, 
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        v2t_layers: List[int] | None = None, 
        v2v_layers: List[int] | None = None,
        sga_loss: bool = False,
        spa_loss: bool = False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(
            model_paths, 
            model_id_with_origin_paths, 
            enable_fp8_training=False
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device="cpu", 
            model_configs=model_configs, 
            audio_processor_config=audio_processor_config
        )
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        extend_patch_embedding(models["dit"])
        models["dit"].to(dtype=torch.bfloat16, device=self.pipe.device)

        self.pipe.dit.seg_head = SegHead()

        v2t_layers = v2t_layers or []
        v2v_layers = v2v_layers or []
        lora_targets = build_lora_target_modules(
            base_targets=lora_target_modules,
            v2t_layers=v2t_layers,
            v2v_layers=v2v_layers,
        )

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_targets,
            lora_rank, 
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.v2t_layers=v2t_layers 
        self.v2v_layers=v2v_layers
        self.sga_loss = sga_loss 
        self.spa_loss = spa_loss
        self.pipe.prompter.tokenizer.tokenizer.add_special_tokens(
            {"additional_special_tokens" :["<id1>", "<id2>", "<id3>", "<id4>", "<id5>"]}
        )
            
    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"], 
            "height": data["video"][0].size[1], 
            "width": data["video"][0].size[0],
            "num_frames":len(data["video"]), 
            ## multi-id masks
            "id1_masks": data.get("id1_masks"),
            "id2_masks": data.get("id2_masks"),
            "id3_masks" : data.get("id3_masks"),
            "id4_masks" : data.get("id4_masks"),
            "id5_masks" : data.get("id5_masks"),
            "mask_image" : data.get("mask_image"),
            # diffusion cfg
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            # loss options
            "v2t_layers" : self.v2t_layers,
            "v2v_layers" : self.v2v_layers,
            "sga_loss" : self.sga_loss,
            "spa_loss" : self.spa_loss,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

def build_dataset(args): 
    main_op = MATRIXDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames,
        time_division_factor=4,
        time_division_remainder=1,
    )

    mask_base = args.dataset_base_path.replace("/videos", "/mask_annotation")

    special_ops = {
        "mask_image" : ToAbsolutePath(mask_base)  >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
        "id_masks" : SequencialProcess(ToAbsolutePath(mask_base) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16))
    }

    dataset = MATRIXDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=main_op,
        special_operator_map=special_ops,
    )

    return dataset 

def main():
    parser = get_parser() 
    args = parser.parse_args()
    v2t_layers = [int(p) for p in args.v2t_layers.split(",")] if args.v2t_layers else []
    v2v_layers= [int(p) for p in args.v2v_layers.split(",")] if args.v2v_layers else []

    dataset = build_dataset(args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        v2t_layers=v2t_layers, 
        v2v_layers=v2v_layers, 
        sga_loss=True if args.sga else False,
        spa_loss=True if args.spa else False,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)

if __name__ == "__main__":
    main()
