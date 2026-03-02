import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


def create_label_colormap():
    n_classes = 19
    cmap = plt.get_cmap("tab20")

    colors = (np.array([cmap(i)[:3] for i in range(n_classes)]) * 255).astype(np.uint8)

    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0:n_classes] = colors
    colormap[0] = [0, 0, 0]  # Ensure class 0 is black

    return colormap


COLORS = create_label_colormap()


class TFLogger:
    def __init__(self, run_id, save_dir, args=None):
        self.enabled = not args.no_logs
        if not self.enabled:
            save_dir = "/tmp/nologs"

        self.run_id = run_id
        self.save_dir = os.path.join(save_dir, self.run_id)
        self.setup_save_dir()
        self.writer = SummaryWriter(self.save_dir)

        self.writer.add_text("config", f"```yaml\n{yaml.dump(vars(args))}\n```")

    def setup_save_dir(self):
        os.makedirs(self.save_dir, exist_ok=True)
        shutil.copytree(
            ".",
            self.save_dir,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".vscode", "*.pt", "*.jpg", "*.png"
            ),
            dirs_exist_ok=True,
        )

    def log_epoch(self, train_loss, val_loss, val_metrics, epoch):
        self.log(
            {"train_loss": train_loss, "val_loss": val_loss},
            "epoch",
            epoch,
        )

        self.log(
            val_metrics,
            "epoch/val_metrics",
            epoch,
        )

    def log(self, losses, phase, step):
        if not self.enabled:
            return
        for k, v in losses.items():
            self.writer.add_scalar(f"{phase}/{k}", v.item(), step)

    def watch_model(self, model, step):
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"model/weights/{name}", param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(f"model/grads/{name}", param.grad, step)

    def log_segmentation(self, rgb, pred_logits, gt_mask, phase, step):
        if not self.enabled:
            return
        # Take the first sample in batch
        rgb_img = rgb[0, -1, :].detach().cpu()  # [3, H, W]
        gt_mask = gt_mask[0].detach().cpu().numpy()  # [H, W]
        pred_logits = pred_logits[0].detach().cpu().numpy()  # [C, H, W]

        # Argmax over channels -> predicted labels
        pred_mask = np.argmax(pred_logits, axis=0).astype(np.uint8)  # [H, W]

        # Unnormalize RGB (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(rgb_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(rgb_img.device)
        rgb_img = rgb_img * std + mean
        rgb_np = rgb_img.cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        rgb_np = (255 * np.clip(rgb_np, 0, 1)).astype(np.uint8)

        # Map masks to color images
        gt_colormap = COLORS[gt_mask]
        pred_colormap = COLORS[pred_mask]

        # Overlay masks on RGB
        gt_overlay = cv2.addWeighted(rgb_np, 0.5, gt_colormap, 0.5, 0)
        pred_overlay = cv2.addWeighted(rgb_np, 0.5, pred_colormap, 0.5, 0)

        # Convert to tensorboard-friendly format (CHW, float32 in [0,1])
        gt_tensor = torch.tensor(gt_overlay).permute(2, 0, 1).float() / 255.0
        pred_tensor = torch.tensor(pred_overlay).permute(2, 0, 1).float() / 255.0

        self.writer.add_image(f"{phase}/gt", gt_tensor, step)
        self.writer.add_image(f"{phase}/pred", pred_tensor, step)

    def log_images(self, rgb, pred_saliency, gt_saliency, phase, step):
        if not self.enabled:
            return
        rgb_img = rgb[0, -1, :].detach().cpu()  # [3, H, W]
        pred_map = pred_saliency[0].detach().cpu()  # [1, H, W]
        gt_map = gt_saliency[0].detach().cpu()  # [1, H, W]

        # Normalize saliency maps
        pred_map = pred_map.clone()
        gt_map = gt_map.clone()

        pred_map -= pred_map.min()
        pred_map /= pred_map.max() + 1e-8

        gt_map -= gt_map.min()
        gt_map /= gt_map.max() + 1e-8

        # Unnormalize RGB image (ImageNet mean and std)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(rgb_img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(rgb_img.device)
        rgb_img = rgb_img * std + mean  # unnormalize
        rgb_np = rgb_img.cpu().permute(1, 2, 0).numpy()  # shape: [H, W, 3]
        rgb_np = (255 * np.clip(rgb_np, 0, 1)).astype(np.uint8)

        # Generate heatmaps
        pred_colormap = cv2.applyColorMap(
            (pred_map.cpu().squeeze().numpy() * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        gt_colormap = cv2.applyColorMap(
            (gt_map.cpu().squeeze().numpy() * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        pred_overlay = cv2.cvtColor(
            cv2.resize(
                cv2.addWeighted(
                    cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR), 0.5, pred_colormap, 0.5, 0
                ),
                None,
                fx=1,
                fy=1,
                interpolation=cv2.INTER_AREA,
            ),
            cv2.COLOR_BGR2RGB,
        )
        gt_overlay = cv2.cvtColor(
            cv2.resize(
                cv2.addWeighted(
                    cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR), 0.5, gt_colormap, 0.5, 0
                ),
                None,
                fx=1,
                fy=1,
                interpolation=cv2.INTER_AREA,
            ),
            cv2.COLOR_BGR2RGB,
        )

        # Convert to tensorboard-friendly format (CHW, float32)
        pred_tensor = torch.tensor(pred_overlay).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.tensor(gt_overlay).permute(2, 0, 1).float() / 255.0

        self.writer.add_image(f"{phase}/gt", gt_tensor, step)
        self.writer.add_image(f"{phase}/pred", pred_tensor, step)
