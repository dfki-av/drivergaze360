# DriverGaze360: Omnidirectional Driver Attention with Object-Level Guidance

<p align="center">
  <a href="https://dfki-av.github.io/drivergaze360/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://arxiv.org/abs/2512.14266" target="_blank"><img src="https://img.shields.io/badge/arXiv-2512.14266-b31b1b"></a>
  <a href="https://huggingface.co/datasets/dfki-av/drivergaze360" target="_blank"><img src="https://img.shields.io/badge/Hugging Face-Dataset-FFD21E"></a>
  <a href="https://github.com/dfki-av/drivergaze360" target="_blank"><img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white"></a>
  <img src="https://img.shields.io/badge/Conference-CVPR%202026-4b44ce">
  <img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-blue">
</p>


<video src="https://github.com/dfki-av/drivergaze360/blob/gh-pages/static/videos/supplementary_video.mp4?raw=true" controls preload></video>

## Setup

### Installation

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Downloading Dataset from HuggingFace 🤗

The dataset is available on HuggingFace 🤗 at: https://huggingface.co/datasets/dfki-av/drivergaze360 


## Training and Inference 

### Run training

```
uv run \
torchrun --standalone --nproc-per-node=gpu \
main.py --model DriverGaze360 \
```

### Run Inference

```
uv run \
torchrun --standalone --nproc-per-node=gpu \
main.py \
--model DriverGaze360 \
--inference \
--video-path VIDEO_PATH \
--video-outpath VIDEO_OUTPATH \
--cktp CKPT
```

### Configuration

```
usage: main.py [-h] [--no-logs] [--save-dir SAVE_DIR] [--model MODEL] [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--w-nss W_NSS] [--w-kld W_KLD] [--w-cc W_CC] [--w-mse W_MSE] [--w-sal W_SAL] [--w-ss W_SS] [--use-amp] [--resume] [--ckpt CKPT]
               [--num-workers NUM_WORKERS] [-T T] [--overlap OVERLAP] [--frame-stride FRAME_STRIDE] [--train-path TRAIN_PATH] [--val-path VAL_PATH] [--img-size IMG_SIZE IMG_SIZE] [--weighted-samples] 
Training script for DriverGaze360

options:
  -h, --help            show this help message and exit
  --no-logs             disable logging
  --save-dir SAVE_DIR   save directory for outputs

Model Config:
  --model MODEL         Model architecture
  --num-epochs NUM_EPOCHS
                        Number of training epochs
  --batch-size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --w-nss W_NSS         Weight for NSS loss
  --w-kld W_KLD         Weight for KLD loss
  --w-cc W_CC           Weight for cross-correlation loss
  --w-mse W_MSE         Weight for MSE loss
  --w-sal W_SAL         Weight for Saliency loss
  --w-ss W_SS           Weight for Sementic Segmentation loss
  --use-amp             Use mixed precision
  --resume              Resume training from ckpt
  --ckpt CKPT           Model Checkpoint

Dataset Config:
  --num-workers NUM_WORKERS
                        Number of data loader workers
  -T T                  Number of consecutive frames
  --overlap OVERLAP     Number of overlapping frames
  --frame-stride FRAME_STRIDE
                        Stride between frames
  --train-path TRAIN_PATH
                        Path to training data
  --val-path VAL_PATH   Path to validation data
  --img-size IMG_SIZE IMG_SIZE
                        Input image size (H, W)
  --weighted-samples    Use weighted sampler with stored KLDs

Inference:
  --inference           Perform inference on a video
  --video-path VIDEO_PATH
                        Path of video folder
  --video-outpath VIDEO_OUTPATH
                        Save path
```

## TODOs:
- [ ] Add data processing scripts
- [ ] Add training scripts
- [ ] Add inference scripts

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{govil_2025,
  title        = {DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance},
  author       = {Shreedhar Govil and Didier Stricker and Jason Rambach},
  year         = {2025},
  eprint       = {2512.14266},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2512.14266}
}
