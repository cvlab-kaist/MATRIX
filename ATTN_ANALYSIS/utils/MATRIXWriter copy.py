import pandas as pd 
import torch 
import os 
from itertools import product 
from tqdm import tqdm
from utils.utils import (
    temporal_repeat,
    compute_iou, 
    compute_binary_cross_entropy, 
    compute_dice, 
    compute_focal_loss,
)
import numpy as np
N_HEAD = 24 
T = 13 
H = 30
W = 45
TEXT_LEN = 396
import time
def _compute_mask_scores(pred, target, threshold=0.5):
    iou = compute_iou(pred, target, threshold=threshold).mean().item() 
    bce_loss = compute_binary_cross_entropy(pred, target).mean().item() 
    dice_loss = compute_dice(pred, target).mean().item() 
    focal_loss = compute_focal_loss(pred, target).mean().item() 

    return iou, bce_loss, dice_loss, focal_loss 

def _compute_sigmoid_mask_scores(pred, target, threshold=0.5):
    iou = compute_iou(pred, target, threshold=threshold).mean().item() 
    return iou 

def _compute_track_scores(fore_attn, back_attn, gt_masks, nonmasks):
    f2v_tp = fore_attn[gt_masks]
    f2v_fn = fore_attn[nonmasks]
    f2v_fp = back_attn[gt_masks]
    f2v_tn = back_attn[nonmasks]
    
    f2v_tp_mean = f2v_tp.mean().item() 
    f2v_tp_sum = f2v_tp.sum().item() 
    f2v_fn_mean = f2v_fn.mean().item() 
    f2v_fn_sum = f2v_fn.sum().item() 

    f2v_fp_mean = f2v_fp.mean().item() 
    f2v_fp_sum = f2v_fp.sum().item()
    f2v_tn_mean = f2v_tn.mean().item() 
    f2v_tn_sum = f2v_tn.sum().item()
    
    return f2v_tp_mean, f2v_fn_mean, f2v_fp_mean, f2v_tn_mean, f2v_tp_sum, f2v_fn_sum, f2v_fp_sum, f2v_tn_sum

class MATRIXWriter(): 
    def __init__(
        self, 
        model="cogvideox",
        num_inference_steps=50,
        num_layers=42, 
        mode="max",
        text_len=226,
        latent_f=13, 
        latent_h=30,
        latent_w=45,
    ):
        self.model=model
        self.num_inference_steps=num_inference_steps 
        self.mode=mode 
        self.text_len=text_len 
        self.latent_f=latent_f 
        self.latent_h=latent_h 
        self.latent_w=latent_w 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        timesteps=[f"timestep{j}" for j in range(num_inference_steps)]
        sub_metric_rows=["iou", "dice", "bce", "focal", "f2t_tp", "f2t_tn", "f2t_fp", "f2t_fn"]
        index = pd.MultiIndex.from_product([timesteps, sub_metric_rows], names=["tiemstep", "metric"])
        self._attn_layer_by_words = {} # for f2t
        self._attn_head_by_words = {}  # for f2t 
        self._attn_layer_frames = {} # for f2v 
        self._attn_head_frames = {} # for f2v

    
    def reset(self, text_len, latent_f, latent_h, latent_w) :
        self.text_len = text_len 
        self.latent_f = latent_f 
        self.latent_h = latent_h 
        self.latent_w = latent_w 
    
    def _update_f2t(self, attn_weights, layer, timestep_idx, prompt, mask_dict, query_idx=0, scale=5.0, word="", device=0, mode="default"):
        device = self.device if device is None else torch.device(device)
        attn_weights = attn_weights.to(device)
        best_span = mask_dict["best_span"]
        masks = mask_dict["masks"] # [13, 30, 45] 
        save_type = mask_dict["save_type"]
        # query_mask = masks[query_idx].unsqueeze(0).to(device)
        query_mask = masks.to(device)
        mask_pos = query_mask.bool().squeeze(0) 
        mask_neg = ~mask_pos 

        if word not in self._attn_layer_by_words:
            self._attn_layer_by_words[word] = {} 
        if word not in self._attn_head_by_words: 
            self._attn_head_by_words[word] = {} 
        if mode == "default":
            f2t = attn_weights[:, :, :self.text_len].reshape(1, self.latent_h, self.latent_w, self.text_len).to(device) # since the query frame is single
        else:
            f2t = attn_weights.reshape(13, self.latent_h, self.latent_w, self.text_len).to(device)
        # split tokens 
        token_indices = torch.tensor(best_span, device=device) if best_span is not None else None 
        except_indices = torch.tensor(np.setdiff1d(np.arange(0, self.text_len), best_span), device=device) if best_span is not None else None

        # sig_f2t = 10 * (f2t - f2t.min()) / (f2t.max() - f2t.min()) - 5.0 
        # sig_f2t = sig_f2t.sigmoid()

        f2t_attn_map_masked = f2t.index_select(-1, token_indices).mean(dim=-1) # [N_HEAD, H, W]
        f2t_attn_map_nonmasked = f2t.index_select(-1, except_indices).mean(dim=-1) # [N_HEAD, H, W]
        # sig_f2t_masked = sig_f2t.index_select(-1, token_indices).mean(dim=-1)
        import cv2; cv2.imwrite("tmp.png", f2t_attn_map_masked[0].detach().cpu().numpy() * 255 * 255)

        # mean over heads
        # mean_sig_f2t = sig_f2t_masked.mean(0, keepdim=True) # [1, H, W]
        mean_f2t_attn_map_masked = f2t_attn_map_masked#.mean(0) # [H, W]
        mean_f2t_attn_map_nonmasked = f2t_attn_map_nonmasked#.mean(0)
    
        ## iou-like gt mask confidence scores -- layer-wise
        # iou, bce_loss, dice_loss, focal_loss = _compute_mask_scores(mean_f2t_attn_map_masked.unsqueeze(0), query_mask, threshold=f2t.mean())
        # sig_iou, sig_bce, sig_dice, sig_focal = _compute_mask_scores(mean_sig_f2t, query_mask)

        f2t_tp_mean, f2t_fn_mean, f2t_fp_mean, f2t_tn_mean, f2t_tp_sum, f2t_fn_sum, f2t_fp_sum, f2t_tn_sum = _compute_track_scores(
            mean_f2t_attn_map_masked,
            mean_f2t_attn_map_nonmasked, 
            mask_pos, 
            mask_neg 
        )

        for metric, value in [
            # ("iou", iou), ("bce", bce_loss), ("dice", dice_loss), ("focal", focal_loss),
            # ("sig_iou", sig_iou), ("sig_bce", sig_bce), ("sig_dice", sig_dice), ("sig_focal", sig_focal),
            ("f2t_tp_mean", f2t_tp_mean), ("f2t_fn_mean", f2t_fn_mean), ("f2t_fp_mean", f2t_fp_mean), ("f2t_tn_mean", f2t_tn_mean),
            ("f2t_tp_sum", f2t_tp_sum), ("f2t_fn_sum", f2t_fn_sum), ("f2t_fp_sum", f2t_fp_sum), ("f2t_tn_sum", f2t_tn_sum),
        ]:
            self._attn_layer_by_words[word][(f"timestep{timestep_idx}", metric, f"layer{layer:02d}")] = round(value, 5)
        torch.cuda.empty_cache()
    
        del f2t, f2t_attn_map_masked, f2t_attn_map_nonmasked
        del mean_f2t_attn_map_masked, mean_f2t_attn_map_nonmasked
        del query_mask, mask_pos, mask_neg, token_indices, except_indices
        # del iou, bce_loss, dice_loss, focal_loss#, f2t_tp, f2t_fn, f2t_fp, f2t_tn
        # del sig_iou, sig_bce, sig_dice, sig_focal, sig_f2t, sig_f2t_masked 
        del f2t_tp_mean, f2t_fn_mean, f2t_fp_mean, f2t_tn_mean, f2t_tp_sum, f2t_fn_sum ,f2t_fp_sum, f2t_tn_sum 
        torch.cuda.empty_cache()

    def _update_v2t(self, attn_weights, layer, timestep_idx, prompt, mask_dict, query_idx=0, scale=5.0, word="", device=0, mode="default"):
        device = self.device if device is None else torch.device(device)
        attn_weights = attn_weights.to(device)
        best_span = mask_dict["best_span"]
        masks = mask_dict["masks"] # [13, 30, 45] 
        save_type = mask_dict["save_type"]
        query_mask = masks[query_idx].unsqueeze(0).to(device)
        mask_pos = query_mask.bool().squeeze(0) 
        mask_neg = ~mask_pos 

        if word not in self._attn_layer_by_words:
            self._attn_layer_by_words[word] = {} 
        if word not in self._attn_head_by_words: 
            self._attn_head_by_words[word] = {} 

        if mode == "default":
            f2t = attn_weights[:, :, :self.text_len].reshape(1, self.latent_h, self.latent_w, self.text_len).to(device) # since the query frame is single
        else:
            f2t = attn_weights.reshape(1, self.latent_h, self.latent_w, self.text_len).to(device)
        breakpoint()
        # split tokens 
        token_indices = torch.tensor(best_span, device=device) if best_span is not None else None 
        except_indices = torch.tensor(np.setdiff1d(np.arange(0, self.text_len), best_span), device=device) if best_span is not None else None

        sig_f2t = 10 * (f2t - f2t.min()) / (f2t.max() - f2t.min()) - 5.0 
        sig_f2t = sig_f2t.sigmoid()

        f2t_attn_map_masked = f2t.index_select(-1, token_indices).mean(dim=-1) # [N_HEAD, H, W]
        f2t_attn_map_nonmasked = f2t.index_select(-1, except_indices).mean(dim=-1) # [N_HEAD, H, W]
        sig_f2t_masked = sig_f2t.index_select(-1, token_indices).mean(dim=-1)

        # mean over heads
        mean_sig_f2t = sig_f2t_masked.mean(0, keepdim=True) # [1, H, W]
        mean_f2t_attn_map_masked = f2t_attn_map_masked.mean(0) # [H, W]
        mean_f2t_attn_map_nonmasked = f2t_attn_map_nonmasked.mean(0)
    
        ## iou-like gt mask confidence scores -- layer-wise
        iou, bce_loss, dice_loss, focal_loss = _compute_mask_scores(mean_f2t_attn_map_masked.unsqueeze(0), query_mask, threshold=f2t.mean())
        sig_iou, sig_bce, sig_dice, sig_focal = _compute_mask_scores(mean_sig_f2t, query_mask)

        f2t_tp_mean, f2t_fn_mean, f2t_fp_mean, f2t_tn_mean, f2t_tp_sum, f2t_fn_sum, f2t_fp_sum, f2t_tn_sum = _compute_track_scores(
            mean_f2t_attn_map_masked,
            mean_f2t_attn_map_nonmasked, 
            mask_pos, 
            mask_neg 
        )

        for metric, value in [
            ("iou", iou), ("bce", bce_loss), ("dice", dice_loss), ("focal", focal_loss),
            ("sig_iou", sig_iou), ("sig_bce", sig_bce), ("sig_dice", sig_dice), ("sig_focal", sig_focal),
            ("f2t_tp_mean", f2t_tp_mean), ("f2t_fn_mean", f2t_fn_mean), ("f2t_fp_mean", f2t_fp_mean), ("f2t_tn_mean", f2t_tn_mean),
            ("f2t_tp_sum", f2t_tp_sum), ("f2t_fn_sum", f2t_fn_sum), ("f2t_fp_sum", f2t_fp_sum), ("f2t_tn_sum", f2t_tn_sum),
        ]:
            self._attn_layer_by_words[word][(f"timestep{timestep_idx}", metric, f"layer{layer:02d}")] = round(value, 5)
        torch.cuda.empty_cache()
    
        del f2t, f2t_attn_map_masked, f2t_attn_map_nonmasked
        del mean_f2t_attn_map_masked, mean_f2t_attn_map_nonmasked
        del query_mask, mask_pos, mask_neg, token_indices, except_indices
        del iou, bce_loss, dice_loss, focal_loss#, f2t_tp, f2t_fn, f2t_fp, f2t_tn
        del sig_iou, sig_bce, sig_dice, sig_focal, sig_f2t, sig_f2t_masked 
        del f2t_tp_mean, f2t_fn_mean, f2t_fp_mean, f2t_tn_mean, f2t_tp_sum, f2t_fn_sum ,f2t_fp_sum, f2t_tn_sum 
        torch.cuda.empty_cache()
    def _update_f2v(self, attn_weights, layer, timestep_idx, mask_dict, query_idx=0, word="", frame_idx=None, device=0, mode="default"):
        attn_weights = attn_weights.to(device)
        masks_reshaped_pos = mask_dict["masks"][query_idx].reshape(-1).bool().to(device)
        masks_reshaped_neg = ~masks_reshaped_pos
        masks = mask_dict["masks"].to(device)
        mask_pos = masks.bool()
        masks_neg = ~mask_pos

        
        # attention split
        if mode == "default":
            f2v = attn_weights[:, :, TEXT_LEN:]  # [N_HEAD, HW, THW]
        elif mode == "separate":
            f2v = attn_weights
        frame_tokens = self.latent_h * self.latent_w

        if word not in self._attn_layer_frames:
            self._attn_layer_frames[word] = {}
        if word not in self._attn_head_frames:
            self._attn_head_frames[word] = {}

        # define split ranges
        ranges = {
            "whole": slice(0, self.latent_f * frame_tokens),
            "first": slice(0, frame_tokens),
            "self": slice(query_idx * frame_tokens, (query_idx + 1) * frame_tokens),
            "cross": None,  
        }

        for key, r in ranges.items():
            if key == "cross":
                # cross = whole - self frame
                idx = torch.arange(self.latent_f * frame_tokens, device=device)
                self_idx = torch.arange(query_idx * frame_tokens, (query_idx + 1) * frame_tokens, device=device)
                cross_idx = torch.tensor([i for i in idx.tolist() if i not in self_idx.tolist()], device=device)
                f2v_split = f2v[:, :, cross_idx]
            else:
                f2v_split = f2v[:, :, r] if r is not None else f2v
            
            if key == "whole":
                selected_frames = self.latent_f
            elif key == "first":
                selected_frames = 1
            elif key == "self":
                selected_frames = 1
            elif key == "cross":
                selected_frames = self.latent_f - 1

            # masked/unmasked average
            f2v_masked = f2v_split[:, masks_reshaped_pos].mean(dim=1).to(device) 
            f2v_nonmasked = f2v_split[:, masks_reshaped_neg].mean(dim=1).to(device) 
            # f2v_masked = f2v_masked.reshape(self.num_heads, self.latent_f, self.latent_h, self.latent_w)
            # f2v_nonmasked = f2v_nonmasked.reshape(self.num_heads, self.latent_f, self.latent_h, self.latent_w)
            f2v_masked = f2v_masked.reshape(self.num_heads, selected_frames, self.latent_h, self.latent_w)
            f2v_nonmasked = f2v_nonmasked.reshape(self.num_heads, selected_frames, self.latent_h, self.latent_w)

            mean_f2v_masked = f2v_masked.mean(0)
            mean_f2v_nonmasked = f2v_nonmasked.mean(0)
            
            # split에 따른 mask 선택
            if key == "whole":
                mask_selected = masks  # [latent_f, H, W]
            elif key == "first":
                mask_selected = masks[0:1]  # [1, H, W]
            elif key == "self":
                mask_selected = masks[query_idx:query_idx+1]  # [1, H, W]
            elif key == "cross":
                cross_indices = [i for i in range(self.latent_f) if i != query_idx]
                mask_selected = masks[cross_indices]  # [latent_f-1, H, W]
            mask_selected_neg = ~mask_selected.bool()

            # iou, bce_loss, dice_loss, focal_loss = _compute_mask_scores(
            #     mean_f2v_masked, mask_selected, threshold=f2v.mean()
            # )
            # f2v_tp, f2v_fn, f2v_fp, f2v_tn = _compute_track_scores(
            #     mean_f2v_masked, mean_f2v_nonmasked, mask_selected.bool(), ~mask_selected.bool()
            # )
            f2v_tp_mean, f2v_fn_mean, f2v_fp_mean, f2v_tn_mean, f2v_tp_sum, f2v_fn_sum, f2v_fp_sum, f2v_tn_sum = _compute_track_scores(
                mean_f2v_masked, mean_f2v_nonmasked, mask_selected.bool(), mask_selected_neg
            )

            for metric, value in [
                # ("iou", iou), ("bce", bce_loss), ("dice", dice_loss), ("focal", focal_loss),
                ("f2v_tp_mean", f2v_tp_mean), ("f2v_fn_mean", f2v_fn_mean), ("f2v_fp_mean", f2v_fp_mean), ("f2v_tn_mean", f2v_tn_mean),
                ("f2v_tp_sum", f2v_tp_sum), ("f2v_fn_sum", f2v_fn_sum), ("f2v_fp_sum", f2v_fp_sum), ("f2v_tn_sum", f2v_tn_sum),
            ]:
                self._attn_layer_frames[word][(f"timestep{timestep_idx}", key, metric, f"layer{layer:02d}")] = round(value, 5)
            
        del f2v, f2v_masked, f2v_nonmasked
        del mean_f2v_masked, mean_f2v_nonmasked
        del mask_selected, mask_selected_neg
        # del iou, bce_loss, dice_loss, focal_loss#, f2t_tp, f2t_fn, f2t_fp, f2t_tn
        del f2v_tp_mean, f2v_fn_mean, f2v_fp_mean, f2v_tn_mean, f2v_tp_sum, f2v_fn_sum ,f2v_fp_sum, f2v_tn_sum 

        torch.cuda.empty_cache()

    def update_separate(self, attn_weights, layer, timestep_idx, prompt, mask_dict, query_idx=0, scale=5.0, word="", device=0):
        best_span = mask_dict["best_span"]
        masks = mask_dict["masks"].cpu() 
        save_type = mask_dict["save_type"]
        if best_span is not None:
            self._update_f2t(
                attn_weights[1], layer, timestep_idx, prompt, mask_dict, query_idx, scale, word, device, mode="separate"
            )
        self._update_f2v(attn_weights[0], layer, timestep_idx, mask_dict, query_idx, word, device, mode="separate")
        torch.cuda.empty_cache()
        
    def update(self, attn_weights, layer, timestep_idx, prompt, mask_dict, query_idx=0, scale=5.0, word="", device=0):
        best_span = mask_dict["best_span"]
        masks = mask_dict["masks"].cpu() 
        save_type = mask_dict["save_type"]

        if best_span is not None:
            self._update_f2t(attn_weights, layer, timestep_idx, prompt, mask_dict, query_idx, scale, word, device)
        self._update_f2v(attn_weights, layer, timestep_idx, mask_dict, query_idx, word, device, mode="default")
        torch.cuda.empty_cache()

    def _write_f2v(self, output_dir="hunyuan", prompt="", save_type="", word="", query_idx=0):
        os.makedirs(os.path.join(output_dir, prompt, "layer", str(f"{query_idx:2d}"), save_type), exist_ok=True)
        layer_records = self._attn_layer_frames[word]
        layer_index_set = sorted(set([k[:3] for k in layer_records]))
        layer_columns = sorted(set([k[3] for k in layer_records]))

        layer_data = {
            col: [layer_records.get((idx[0], idx[1], idx[2], col), None) for idx in layer_index_set]
            for col in layer_columns
        }
        multi_index = pd.MultiIndex.from_tuples(layer_index_set, names=["timestep", "type",  "metric"])
        layer_df = pd.DataFrame(layer_data, index=multi_index)

        layer_df.to_excel(os.path.join(output_dir, prompt, "layer", str(f"{query_idx:2d}"), save_type, f'{word}.xlsx'), engine="openpyxl")
 
    def _write_f2t(self, output_dir="hunyuan", prompt="", save_type="", word="", query_idx=0): 
        os.makedirs(os.path.join(output_dir, prompt, "layer", str(f"{query_idx:2d}"), save_type), exist_ok=True)
        layer_records = self._attn_layer_by_words[word]
        layer_index_set = sorted(set([k[:2] for k in layer_records]))
        layer_columns = sorted(set([k[2] for k in layer_records]))

        layer_data = {
            col: [layer_records.get((idx[0], idx[1], col), None) for idx in layer_index_set]
            for col in layer_columns
        }
        multi_index = pd.MultiIndex.from_tuples(layer_index_set, names=["timestep",  "metric"])
        layer_df = pd.DataFrame(layer_data, index=multi_index)

        layer_df.to_excel(os.path.join(output_dir, prompt, "layer", str(f"{query_idx:2d}"), save_type, f'{word}.xlsx'), engine="openpyxl")

    def write(self, output_dir="hunyuan", prompt="", save_type="", word="", query_idx=0):
        import re

        max_len = 50
        safe_prompt = re.sub(r'[^a-zA-Z0-9 _-]', '', prompt)  
        safe_prompt = safe_prompt[:max_len]
        # breakpoint()
        if word in self._attn_layer_by_words and word in self._attn_head_by_words: 
            self._write_f2t(output_dir, safe_prompt, save_type, word, query_idx)
        # os.makedirs(os.path.join(output_dir, prompt, "layer", str(f"{query_idx:2d}"), save_type), exist_ok=True)
        self._write_f2v(f"{output_dir}_f2v", safe_prompt, save_type, word, query_idx)