import os
from functools import cache
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image, decode_jpeg, decode_png, read_file
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from utils.data import get_bin_masks, get_instance_masks, get_seen_objects

LOCAL_RANK = int(os.environ["LOCAL_RANK"])


@cache
def load_csv_from_path(data_dir):
    data_file = os.path.join(data_dir, "sim_gaze_df.csv")
    return pd.read_csv(data_file)


class DG360Dataset(Dataset):
    def __init__(
        self,
        datapath,
        T,
        frame_stride,
        overlap,
        img_size,
        data_device=torch.device("cuda"),
    ):
        self.dataset_path = datapath
        self.clip_len = T
        self.frame_stride = frame_stride
        self.overlap = overlap
        self.img_size = img_size
        self.device = data_device

        self.make_paths()
        # check data exists
        self.check_files()
        self.make_samples()

        self.rgb_transforms = v2.Compose(
            [
                v2.Resize(size=self.img_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

        self.gt_transforms = v2.Compose(
            [
                v2.Resize(size=self.img_size),
                v2.ToDtype(torch.float32, scale=True),
            ],
        )

        self.is_transform = v2.Compose(
            [
                v2.Resize(
                    size=self.img_size, interpolation=v2.InterpolationMode.NEAREST_EXACT
                )
            ]
        )

        # Subtract this mask for the IS mask to remove the car
        self.sub_is = (
            self.is_transform(
                decode_image("config/self_mask.png", mode="GRAY").to(self.device)
            )
            / 255.0
        )

    def make_samples(self):
        # For sampling subsequences: store all valid (clip_idx, start_idx) pairs
        min_start = 30 * 1  # Skip the first 1 seconds x 30 FPS
        self.samples = []
        for data_idx, data_len in enumerate(self.data_lens):
            max_start = data_len - self.clip_len * self.frame_stride
            for start_idx in range(min_start, max_start, self.overlap):
                self.samples.append((data_idx, start_idx))

    def make_paths(self):
        self.data_paths = []
        for user in os.listdir(self.dataset_path):
            user_path = os.path.join(self.dataset_path, user)
            for session in os.listdir(user_path):
                session_path = os.path.join(user_path, session)
                for iteration in os.listdir(session_path):
                    iter_path = os.path.join(session_path, iteration)
                    self.data_paths.append(iter_path)
        self.data_paths.sort()

    def check_files(self):
        self.data_lens = []
        for data_path in tqdm(
            self.data_paths, desc="Checking files", disable=LOCAL_RANK != 0
        ):
            rgb_path = os.path.join(data_path, "rgb")
            gt_path = os.path.join(data_path, "saliency")
            is_path = os.path.join(data_path, "IS")
            dt_path = os.path.join(data_path, "DT")
            metadata_path = os.path.join(data_path, "sim_gaze_df.csv")

            assert os.path.isdir(rgb_path), f"Path error: {rgb_path=}"
            assert os.path.isdir(gt_path), f"Path error: {gt_path=}"
            assert os.path.isdir(is_path), f"Path error: {is_path=}"
            assert os.path.isdir(dt_path), f"Path error: {dt_path=}"
            assert os.path.exists(metadata_path), f"Path error: {data_path=}"

            metadata = load_csv_from_path(data_path)
            rgb_files = [Path(file).stem for file in sorted(os.listdir(rgb_path))]
            gt_files = [Path(file).stem for file in sorted(os.listdir(gt_path))]
            is_files = [Path(file).stem for file in sorted(os.listdir(is_path))]
            metadata_index = [f"{int(i):06d}" for i in list(metadata.Frame)]
            dt_files = [Path(file).stem for file in sorted(os.listdir(dt_path))]

            num_rgb = len(rgb_files)
            num_gt = len(gt_files)
            num_is = len(is_files)
            num_metadata = len(metadata_index)
            num_dt = len(dt_files)

            assert num_rgb == num_gt, f"{data_path=}, {num_rgb=}, {num_gt=}"
            assert num_rgb == num_is, f"{data_path=}, {num_rgb=}, {num_is=}"
            assert num_rgb == num_metadata, f"{data_path=}, {num_rgb=}, {num_metadata=}"
            assert num_rgb == num_dt, f"{data_path=}, {num_rgb=}, {num_dt=}"
            assert rgb_files == gt_files, f"file names different {data_path=}"
            assert rgb_files == dt_files, f"file names different {data_path=}"
            assert rgb_files == is_files, f"file names different {data_path=}"
            assert rgb_files == metadata_index, f"index different {data_path=}"

            self.data_lens.append(num_metadata)

    def get_avg_saliency(self) -> torch.Tensor:
        # Calculate the weights
        avg_file = Path("weights/average_gt_map.pt")
        if not avg_file.exists() and LOCAL_RANK == 0:
            running_sum = None
            total_count = 0
            for data_idx, start_idx in tqdm(
                self.samples, desc="Calculating avg saliency"
            ):
                gt = self._load_clip(
                    data_idx,
                    [start_idx + (self.clip_len - 1) * self.frame_stride],
                    "saliency",
                    mode="GRAY",
                    transforms=self.gt_transforms,
                )
                gt = gt / (gt.sum() + 1e-8)  # Normalize to probability map
                if running_sum is None:
                    running_sum = gt.clone()
                else:
                    running_sum += gt
                total_count += 1
            avg_map = running_sum / total_count  # [1, H, W]
            avg_map = avg_map / avg_map.max()
            img = to_pil_image(avg_map.squeeze(0).cpu())  # remove batch dim
            img.save("weights/average_gt_map.png")
            torch.save(avg_map.cpu(), avg_file)

        dist.barrier()

        avg_map = torch.load(avg_file)
        avg_map = self.gt_transforms(avg_map)
        return avg_map

    def get_sample_weights(self) -> torch.Tensor:
        weights_file = Path(
            f"weights/train_kld_weights_{self.clip_len}_{self.frame_stride}_{self.overlap}.pt"
        )
        avg_map = self.get_avg_saliency().to(self.device)
        if not weights_file.exists() and LOCAL_RANK == 0:
            avg_map = avg_map.view(1, -1)
            avg_map = avg_map / (avg_map.sum() + 1e-8)
            klds = []
            for data_idx, start_idx in tqdm(
                self.samples, desc="Calculating KLD weights"
            ):
                gt = self._load_clip(
                    data_idx,
                    [start_idx + (self.clip_len - 1) * self.frame_stride],
                    "saliency",
                    mode="GRAY",
                    transforms=self.gt_transforms,
                ).view(1, -1)
                gt = gt / (gt.sum() + 1e-8)
                kld = F.kl_div((avg_map + 1e-8).log(), gt, reduction="batchmean")
                klds.append(kld.item())

            klds = torch.tensor(klds)

            torch.save(klds, weights_file)

        dist.barrier()  # wait till weights are calculated

        klds = torch.load(weights_file)

        return klds

    def __len__(self):
        return len(self.samples)

    def _load_clip(
        self,
        data_idx,
        indices,
        kind: Literal["rgb", "saliency", "IS", "DT"],
        mode="RGB",
        transforms: v2.Transform = None,
    ) -> torch.Tensor:
        data_path = self.data_paths[data_idx]
        ext = "png" if kind in ("IS", "SS") else "jpg"

        images = [
            read_file(os.path.join(data_path, kind, f"{idx + 1:06d}.{ext}"))
            for idx in indices
        ]

        if ext == "jpg":
            clip = decode_jpeg(images, mode=mode, device=self.device)
        else:
            clip = [decode_png(image, mode=mode) for image in images]

        clip = transforms(torch.stack(clip).to(self.device))
        return clip

    def _load_metadata(self, data_idx, indices):
        data_df = load_csv_from_path(self.data_paths[data_idx])
        data_rows = data_df.loc[indices]
        return data_rows[
            [
                "Frame",
                "gaze_x",
                "gaze_y",
                "steer",
                "throttle",
                "brake",
                "gear",
                "loc_x",
                "loc_y",
                "loc_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "vel_x",
                "vel_y",
                "vel_z",
            ]
        ]

    def _make_fixation_mask(
        self, current: pd.DataFrame, future: pd.DataFrame, src_img_size=(6400, 720)
    ):
        """
        Convert gaze_x, gaze_y to binary fixation maps using vectorized operations.
        """
        W_src, H_src = src_img_size
        H, W = self.img_size
        scale_x = W / W_src
        scale_y = H / H_src

        combined = pd.concat([current, future], ignore_index=True)

        # Convert to scaled pixel coordinates
        x = (combined["gaze_x"].to_numpy() * scale_x).round().astype(int)
        y = (combined["gaze_y"].to_numpy() * scale_y).round().astype(int)

        # Filter out-of-bounds indices
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x = x[valid]
        y = y[valid]

        # Create the mask and set the fixation points
        mask = torch.zeros((1, H, W), dtype=torch.float32)
        mask[0, y, x] = 1.0

        return mask


    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with the following keys:
                - "rgb" (Tensor): RGB clip of shape [T, C, H, W], where
                - "gt" (Tensor): Saliency map for the last frame, shape [1, H, W]
                - "fixation" (Tensor): Binary fixation map for the last frame, shape [1, H, W]
                - "idx" (Tensor): Sample index, shape [1]
                - "sal" (Tensor): Saliency GT for the last frame, shape [1, H, W]
                - "dt" (Tensor): Driver attention map for the last frame, shape [1, H, W]
                - "ss" (Tensor): Attended Semantic segmentation GT for the last frame, shape [C, H, W]
        """
        data_idx, start_idx = self.samples[idx]

        indices = [start_idx + i * self.frame_stride for i in range(self.clip_len)]

        rgb_clip = self._load_clip(
            data_idx, indices, "rgb", mode="RGB", transforms=self.rgb_transforms
        )

        sal_clip = self._load_clip(
            data_idx,
            indices[-1:],
            "saliency",
            mode="GRAY",
            transforms=self.gt_transforms,
        ).squeeze(0)

        dt_clip = self._load_clip(
            data_idx,
            indices[-1:],
            "DT",
            mode="GRAY",
            transforms=self.gt_transforms,
        ).squeeze(0)

        is_img = self._load_clip(
            data_idx, indices[-1:], "IS", mode="RGB", transforms=self.is_transform
        ).squeeze(0)

        is_mask = (get_bin_masks(is_img=is_img) - self.sub_is).clamp(0, 255)
        sal_is = get_seen_objects(is_img, is_mask, saliency=sal_clip)
        ss_clip = get_instance_masks(sal_is)

        metadata = self._load_metadata(data_idx, indices)

        future_indices = [
            start_idx + i * self.frame_stride
            for i in range(self.clip_len, 2 * self.clip_len)
        ]
        max_frame_idx = self.data_lens[data_idx] - 1
        future_indices = [i for i in future_indices if i <= max_frame_idx]

        future_metadata = self._load_metadata(data_idx, future_indices)

        fixation_mask = self._make_fixation_mask(metadata, future_metadata).to(
            self.device
        )

        gts = {
            "sal": sal_clip,
            "dt": dt_clip,
            "ss": ss_clip,
        }

        ret_dict = {
            "rgb": rgb_clip,
            "gts": gts,
            "fixation": fixation_mask,
            "idx": torch.tensor([idx]).to(self.device),
        }

        return ret_dict