# DriverGaze360: Omnidirectional Driver Attention with Object-Level Guidance

<p align="center">
  <a href="https://dfki-av.github.io/drivergaze360/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://arxiv.org/abs/2512.14266" target="_blank"><img src="https://img.shields.io/badge/arXiv-2512.14266-b31b1b"></a>
  <a href="https://huggingface.co/datasets/dfki-av/drivergaze360" target="_blank"><img src="https://img.shields.io/badge/Hugging Face-Dataset-FFD21E"></a>
  <a href="https://github.com/dfki-av/drivergaze360" target="_blank"><img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white"></a>
  <img src="https://img.shields.io/badge/Conference-CVPR%202026-4b44ce">
  <img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-blue">
</p>


https://github.com/user-attachments/assets/71f40095-a92f-453c-a4d8-494a8034034f


## Setup

### Installation

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Downloading from HuggingFace 🤗

The dataset and checkpoints are available on HuggingFace 🤗 at:
- Dataset: https://huggingface.co/datasets/dfki-av/drivergaze360
- Checkpoints: https://huggingface.co/dfki-av/drivergaze360-net

### Preparing the data

The release ships one folder per recording iteration, with the frames packed into
videos and a tar archive:

```
<root>/C011/001/001/
├── is.tar                # Instance Segmentation
├── dt.mp4                # Depth maps
├── rgb.mp4               # RGB output
├── saliency.mp4          # Saliency maps
└── sim_gaze_df.csv       # Simulator data on ego car and eye gaze positions
```

The data loader reads individual frames, so unpack every recording first:

```
scripts/prepare_dataset.sh <dataset-root> [jobs]
```

This walks the whole tree and turns each recording into:

```
<root>/C011/001/001/
├── rgb/000001.jpg ...
├── saliency/000001.jpg ...
├── DT/000001.jpg ...
├── IS/000001.png ...
└── sim_gaze_df.csv
```

Frames are numbered from `000001` to match the `Frame` column of
`sim_gaze_df.csv`. Recordings whose folders already exist are skipped, so the
script can be re-run after an interruption. Requires `ffmpeg` and `tar`.

Point `--train-path` / `--val-path` at the unpacked splits when training.

## Training and Inference 

### Run training

```
uv run \
torchrun --standalone --nproc-per-node=gpu \
main.py --model DriverGaze360 \
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
- [X] Add data processing scripts
- [X] Add training scripts

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@InProceedings{Govil_2026_CVPR,
    author    = {Govil, Shreedhar and Stricker, Didier and Rambach, Jason},
    title     = {DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {39786-39795}
}
```

## Acknowledgments

This work was partially funded by the European Union's Horizon Europe Research and Innovation Programme under Grant Agreement No. 101076360 (BERTHA) and by the German Federal Ministry of Research, Technology and Space under Grant Agreement No. 16IW24009 (COPPER). The authors would like to express their sincere appreciation to Prateek Kumar Sharma, for his support with data collection and the implementation of driving scenarios. We also gratefully acknowledge Ruben Abad, Alex Levy, and Prof. Antonio M. López from the Computer Vision Center (CVC) for their methodological guidance and for providing the code used to implement the goal-directed navigation routes applied in collecting part of the dataset presented in this study. Finally, we sincerely thank all the participants who contributed to the dataset collection, as well as our colleagues at DFKI for their valuable feedback and support throughout this project.

![](https://github.com/dfki-av/drivergaze360/blob/gh-pages/static/images/funding_logo.png)

The views and opinions expressed in this publication are solely those of the author(s) and do not necessarily reflect those of the European Union or the European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor the granting authority can be held responsible for them. 
