import argparse
import os
from datetime import datetime

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group

from train import Trainer

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup():
    init_process_group(backend="nccl", device_id=torch.device(LOCAL_RANK))
    torch.cuda.set_device(LOCAL_RANK)


def cleanup():
    destroy_process_group()


def main(run_id, args):
    setup()
    try:
        trainer = Trainer(run_id, args)
        trainer.train()
    finally:
        cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for DriverGaze360-Net")

    parser.add_argument("--no-logs", action="store_true", help="disable logging")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./perception_runs",
        help="save directory for outputs",
    )

    # === Model Config Group ===
    model_group = parser.add_argument_group("Model Config")
    model_group.add_argument(
        "--model", type=str, default="DriverGaze360", help="Model architecture"
    )
    model_group.add_argument(
        "--num-epochs", type=int, default=10, help="Number of training epochs"
    )
    model_group.add_argument("--batch-size", type=int, default=5, help="Batch size")
    model_group.add_argument("--lr", type=float, default=1.0e-6, help="Learning rate")
    model_group.add_argument(
        "--w-nss", type=float, default=0, help="Weight for NSS loss"
    )
    model_group.add_argument(
        "--w-kld", type=float, default=1.0, help="Weight for KLD loss"
    )
    model_group.add_argument(
        "--w-cc", type=float, default=1.0, help="Weight for cross-correlation loss"
    )
    model_group.add_argument(
        "--w-mse", type=float, default=0, help="Weight for MSE loss"
    )
    model_group.add_argument(
        "--w-sal", type=float, default=1.0, help="Weight for Saliency loss"
    )
    model_group.add_argument(
        "--w-ss",
        type=float,
        default=2.0,
        help="Weight for Sementic Segmentation loss",
    )
    model_group.add_argument(
        "--use-amp",
        action="store_true",
        help="Use mixed precision",
    )

    # ==== Train Configs ====
    model_group.add_argument(
        "--resume", action="store_true", help="Resume training from ckpt"
    )
    model_group.add_argument(
        "--ckpt", type=str, default="", required=False, help="Model Checkpoint"
    )

    # === Dataset Config Group ===
    data_group = parser.add_argument_group("Dataset Config")
    data_group.add_argument(
        "--num-workers", type=int, default=2, help="Number of data loader workers"
    )
    data_group.add_argument(
        "-T", type=int, default=16, help="Number of consecutive frames"
    )
    data_group.add_argument(
        "--overlap", type=int, default=8, help="Number of overlapping frames"
    )
    data_group.add_argument(
        "--frame-stride", type=int, default=1, help="Stride between frames"
    )
    data_group.add_argument(
        "--train-path",
        type=str,
        default="./drivergaze360_dataset/train",
        help="Path to training data",
    )
    data_group.add_argument(
        "--val-path",
        type=str,
        default="./drivergaze360_dataset/val",
        help="Path to validation data",
    )
    data_group.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=(224, 1120),
        help="Input image size (H, W)",
    )
    data_group.add_argument(
        "--weighted-samples",
        action="store_true",
        help="Use weighted sampler with stored KLDs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    job_id = os.getenv("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d%H%M%S"))
    job_name = os.getenv("SLURM_JOB_NAME", "local")
    run_id = job_id + "_" + job_name
    main(run_id, args)
