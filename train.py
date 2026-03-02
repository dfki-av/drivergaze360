import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)
from torcheval.metrics import Mean
from torcheval.metrics.toolkit import sync_and_compute
from tqdm import tqdm

from dataset.dataset import DG360Dataset
from losses import Criterion
from models.model_factory import build_model
from utils import MetricAggregator, Metrics, TFLogger, clip_percentiles

torch.manual_seed(123)
np.random.seed(123)

LOCAL_RANK = int(os.environ["LOCAL_RANK"])


class Trainer:
    def __init__(self, run_id, args):
        self.run_id = run_id
        if LOCAL_RANK == 0:
            self.logger = TFLogger(run_id, args.save_dir, args=args)

        self.epochs_run = 0
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.img_size = args.img_size

        self.device = torch.device(
            f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
        )

        self.model = build_model(args)

        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[LOCAL_RANK])

        self.use_amp = args.use_amp

        self.use_amp and LOCAL_RANK == 0 and print("Using AMP")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.metrics = Metrics()

        self.log_interval = 500
        self.train_step = 0
        self.val_step = 0

        self.setup_dataloader(args)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.criterion = Criterion(
            w_sal=args.w_sal,
            w_ss=args.w_ss,
            w_nss=args.w_nss,
            w_cc=args.w_cc,
            w_kld=args.w_kld,
            w_mse=args.w_mse,
        ).to(self.device)

        if args.resume:
            self.load_checkpoint(args.ckpt)
            LOCAL_RANK == 0 and print(f"Resuming training from {args.ckpt}")

    def setup_dataloader(self, args):
        self.train_ds = DG360Dataset(
            args.train_path,
            args.T,
            args.frame_stride,
            args.overlap,
            args.img_size,
            self.device,
        )
        self.val_ds = DG360Dataset(
            args.val_path,
            args.T,
            args.frame_stride,
            args.overlap,
            args.img_size,
            self.device,
        )

        self.train_sampler = DistributedSampler(self.train_ds, shuffle=True)
        self.train_weights = torch.ones(len(self.train_ds), device=self.device)
        if args.weighted_samples:
            LOCAL_RANK == 0 and print("Using KLD Weighted training instead of sampling")
            train_weights = self.train_ds.get_sample_weights()
            self.train_weights, _, _ = clip_percentiles(klds=train_weights)
            self.train_weights = self.train_weights.to(self.device)
            assert len(self.train_weights) == len(self.train_ds), (
                "Weights length not same as dataset"
            )

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            sampler=self.train_sampler,
            shuffle=False,
        )

        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            sampler=DistributedSampler(
                self.val_ds,
                shuffle=False,
            ),
            num_workers=args.num_workers,
            pin_memory=False,
        )

    def save_checkpoint(self, epoch):
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
        }
        out_path = os.path.join(self.logger.save_dir, f"model_ep_{epoch}.pt")
        torch.save(
            checkpoint,
            out_path,
        )
        print(f"Epoch {epoch} | Training checkpoint saved at: {out_path}")

    def load_checkpoint(self, save_path):
        assert os.path.exists(save_path), "File does not exist"
        checkpoint = torch.load(save_path)
        self.model.module.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epochs_run = checkpoint["epoch"]
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def train(self):
        for epoch in range(self.epochs_run, self.num_epochs):
            self.train_sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.val(epoch)
            if LOCAL_RANK == 0:
                print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
                print(
                    "Val metrics: ",
                    ", ".join(f"{k}: {v.item():.4f}" for k, v in val_metrics.items()),
                )
                self.save_checkpoint(epoch)
                self.logger.log_epoch(train_loss, val_loss, val_metrics, epoch)

            dist.barrier()

    def _fix_batch(self, batch):
        """removes invalid saliency and fixation maps from the batch"""
        gt_saliency = batch["gts"]["sal"]
        fixation = batch["fixation"]

        gt_valid_mask = gt_saliency.view(gt_saliency.size(0), -1).sum(dim=1) > 0  # [B]
        fixation_valid_mask = fixation.view(fixation.size(0), -1).sum(dim=1) > 0  # [B]

        # Combine masks (both must be valid)
        valid_mask = gt_valid_mask & fixation_valid_mask  # [B]

        for key, vals in batch.items():
            if key == "gts":
                for task in vals:
                    batch[key][task] = vals[task][valid_mask]
            else:
                batch[key] = vals[valid_mask]
        batch["valid_count"] = valid_mask.sum().item()
        return batch

    def train_one_epoch(self, epoch):
        self.model.train()

        train_loss = Mean(device=self.device)

        for batch_id, batch in tqdm(
            enumerate(self.train_dl),
            total=len(self.train_dl),
            desc=f"Epoch {epoch} | Train",
            disable=LOCAL_RANK != 0,
        ):
            self.optimizer.zero_grad()

            batch = self._fix_batch(batch)

            gts = batch["gts"]
            rgb = batch["rgb"].to(self.device)
            tasks = batch["tasks"]
            direction = batch["direction"]
            valid_count = batch["valid_count"]
            idx = batch["idx"]
            weights = self.train_weights[idx].flatten()

            if valid_count == 0:
                continue

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                preds = self.model(rgb, tasks=tasks, direction=direction)
                losses = self.criterion(
                    gts=batch["gts"],
                    preds=preds,
                    fixation=batch["fixation"],
                    weights=weights,
                )

                loss = valid_count * losses["loss"]  # Scale loss with the mask count

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss.update(loss)

            if LOCAL_RANK == 0:
                self.logger.log(losses, "train", step=self.train_step)

                if self.train_step % self.log_interval == 0 and valid_count > 0:
                    if "sal" in preds:
                        self.logger.log_images(
                            rgb,
                            preds["sal"],
                            gts["sal"],
                            phase="train/sal",
                            step=self.train_step,
                        )
                    if "ss" in preds:
                        self.logger.log_segmentation(
                            rgb,
                            preds["ss"],
                            gts["ss"],
                            phase="train/ss",
                            step=self.train_step,
                        )
            self.train_step += 1

        train_loss = sync_and_compute(train_loss)
        return train_loss

    @torch.no_grad()
    def val(self, epoch):
        self.model.eval()

        val_loss = Mean(device=self.device)
        val_metrics = MetricAggregator(self.device)

        for batch_id, batch in tqdm(
            enumerate(self.val_dl),
            total=len(self.val_dl),
            desc=f"Epoch {epoch} | Eval",
            disable=LOCAL_RANK != 0,
        ):
            batch = self._fix_batch(batch)

            gts = batch["gts"]
            rgb = batch["rgb"].to(self.device)
            tasks = batch["tasks"]
            direction = batch["direction"]
            valid_count = batch["valid_count"]

            if valid_count == 0:
                continue

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                preds = self.model(rgb, tasks=tasks, direction=direction)
                losses = self.criterion(
                    gts=batch["gts"],
                    preds=preds,
                    fixation=batch["fixation"],
                    weights=1,
                )

                batch_metrics = self.metrics(
                    gts=gts, preds=preds, fixation=batch["fixation"]
                )

            loss = valid_count * losses["loss"]

            val_loss.update(loss)

            val_metrics.update(batch_metrics, valid_count)

            if LOCAL_RANK == 0:
                self.logger.log(losses, "val", step=self.val_step)

                if valid_count > 0 and self.val_step % self.log_interval == 0:
                    if "sal" in preds:
                        self.logger.log_images(
                            rgb,
                            preds["sal"],
                            gts["sal"],
                            phase="val/sal",
                            step=self.val_step,
                        )
                    if "ss" in preds:
                        self.logger.log_segmentation(
                            rgb,
                            preds["ss"],
                            gts["ss"],
                            phase="val/ss",
                            step=self.val_step,
                        )
            self.val_step += 1

        val_loss = sync_and_compute(val_loss)

        val_metrics = val_metrics.compute()

        return val_loss, val_metrics
