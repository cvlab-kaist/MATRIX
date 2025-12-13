# MATRIX: Mask Track Alignment for Interaction-aware Video Generation
<a href="https://arxiv.org/pdf/2510.07310"><img src="https://img.shields.io/badge/arXiv-2510.07310-%23B31B1B"></a>
<a href="https://cvlab-kaist.github.io/MATRIX"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>  
<br>

This is the official implmentation of the paper *"MATRIX: Mask Track Alignment for Interaction-aware Video Generation"*

by [Siyoon Jin](https://JinSY515.github.io/my-page/), [Seongchan Kim](https://deep-overflow.github.io/), [Dahyun Chung](https://scholar.google.com/citations?user=EU52riMAAAAJ&hl=ko), [Jaeho Lee](https://scholar.google.com/citations?user=rfDpohEAAAAJ&hl=ko), [Hyunwook Choi](https://scholar.google.com/citations?user=vqSp0lwAAAAJ&hl=ko), [Jisu Nam](https://nam-jisu.github.io), [Jiyoung Kim](https://scholar.google.com/citations?user=DqG-ybIAAAAJ&hl=ko) and [Seungryong Kim](https://cvlab.kaist.ac.kr/members/faculty)

# Introduction 
![](images/fig_teaser.png)<br>

### 🤔 How do Video Diffusion Transformers semantically bind text and video, and how is this binding propagated to support interactions? 

MATRIX identifies interaction-dominant layers in video DiTs and introduces a simple yet effective regularization that aligns their attention to multi-instance mask tracks, resulting in more interaction-aware video generation.

MATRIX introduces :

🔎  **Novel Analysis** specifically designed to quantify semantic grounding and propagation 

🚀 **Simple yet Effective Loss Design** that aligns the attention in interaction-dominant layers with multi-instance mask tracks 

🏅 **Novel InterGenEval Metrics** designed to evaluate interaction-awareness of the generated video.



# Semantic Grounding & Propagation Analysis 
## Analysis on Generated Videos
For video DiT backbone models, including CogVideoX-2B-I2V, CogVideoX-5B-I2V, HunyuanVideo-I2V, Wan2.1-14B-I2V, we provide analysis framework. 
Additional details and settings are available in ``ATTN_ANALYSIS`` directory
```
bash analyze_attention.sh 
```
**Options**
- ``--model`` : video backbone model, choices = ['cogvideox_i2v_2b', 'cogvideox_i2v_5b', 'wan-i2v', 'hunyuan-i2v']
- ``--height`` : height of the generated video (e.g., 480, must be multiple of 16) 
- ``--width`` : width of the generated video (e.g., 720, must be multiple of 16)
- ``--text_len`` : length of the text embedding (e.g., 226 for CogVideoX)
- ``--num_frames`` : number of frames (e.g., 49)
- ``--output_dir`` : output directory to save generated videos 
- ``--data_dir`` : directory of RGB frames 
- ``--masks_dir`` : directory of mask frames 
- ``--prompt_path`` : path to prompt json file 

# MATRIX
 
## 🔧 Installation 
```
git clone https://github.com/cvlab-kaist/MATRIX.git 
cd MATRIX

conda create -n matrix python=3.11 -y
conda activate matrix
pip install -r requirements.txt 

cd diffsynth
pip install -e .
```

## 📹 Dataset Preparation
The code assumes a dataset structure like:

```text
DATA_ROOT/
  videos/
    000001.mp4
    000002.mp4
    ...
  mask_annotation/
    000001/
      <id1>/
        000.png
        001.png
        ...
      <id2>/
        ...
      <id3>/
      <id4>/
      <id5>/
      merged/
    000002/
      <id1>/
      ...
  metadata.csv   (or .json / .jsonl)
```
- `videos/` contains the input videos used for training.
- Each row in `metadata.csv` references a video file via the `video` field, e.g.,:
```text
video,prompt
000001.mp4,"a <id1> person passes a ball to another <id2> person"
```
- Each id masks should be paired with the corresponding ids. 
- `merged` stores color-coded union masks that aggregate all IDs (id1–id5) into a single mask image. Each ID is assigned a fixed, unique color (e.g., all pixels belonging to id1 share the same color, all id2 pixels share another color, etc.), so instance regions are distinguishable in one palette image.

For detailed preparation, please refer to [DATA_PREPARATION](DATA_PREPARATION/README.md)

### 🔥 Training
Below is an example command to launch training.
Adjust paths, GPU index, and hyperparameters for your environment:
```bash
bash matrix/train/train.sh
```
- `--dataset_base_path`
Root folder that contains final_videos_16fps/.

- `--dataset_metadata_path`
Path to metadata.csv (or .json, .jsonl) describing the training samples.

- `--height, --width`
Target spatial resolution. Must be compatible with the base Wan2.1 I2V model (e.g., 480×832).

- `--output_path`
Where fine-tuned LoRA checkpoints and logs will be written.

- `--trainable_models`
Which submodules to train (e.g., dit, dit.patch_embedding, seg_head).

- `--v2t_layers, --v2v_layers`
DiT block indices where interaction-aware supervision is applied:

  - `v2t` = video-to-text (semantic grounding alignment)

  - `v2v` = video-to-video (temporal/propagation alignment)

- `--sga, --spa` Flags to enable: SGA (Semantic Grounding Alignment),  SPA (Semantic Propagation Alignment)
# InterGenEval

For detailed usage and examples, please refer to [InterGenEval](InterGenEval/README.md)

# Citation
If you find this research useful, please consider citing:
```
@misc{jin2025matrixmasktrackalignment,
      title={MATRIX: Mask Track Alignment for Interaction-aware Video Generation}, 
      author={Siyoon Jin and Seongchan Kim and Dahyun Chung and Jaeho Lee and Hyunwook Choi and Jisu Nam and Jiyoung Kim and Seungryong Kim},
      year={2025},
      eprint={2510.07310},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.07310}, 
}
```