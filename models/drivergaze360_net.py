# Code inspired from https://github.com/ykotseruba/SCOUT
import os
from typing import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import swin_transformer, video
import torch.distributed as dist

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


class DriverGaze360(nn.Module):
    def __init__(
        self,
        heads=["ss", "sal"],
    ):
        super().__init__()

        self.backbone = VideoSwinBackbone()
        self.decoder = DecoderSwin(heads=heads, num_classes=7)

    def forward(self, x, tasks, direction, **kwargs):
        # x: B T C H W
        x = x.permute(0, 2, 1, 3, 4)  # B C T H W

        b_out = self.backbone(x)

        out = {}

        for head in self.decoder.heads:
            out[f"{head}"] = self.decoder(b_out, head=head)

        return out


class VideoSwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        if LOCAL_RANK == 0:
            backbone = video.swin3d_s(weights=video.Swin3D_S_Weights.KINETICS400_V1)
        if torch.distributed.is_initialized():
            dist.barrier()

        backbone = video.swin3d_s(weights=video.Swin3D_S_Weights.KINETICS400_V1)

        self.patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for layer in backbone.features:
            if isinstance(layer, swin_transformer.PatchMerging):
                self.downsamples.append(layer)
            else:
                self.layers.append(layer)
        self.downsamples.append(None)

    def forward(self, x):
        # x: B C T H W
        x = self.patch_embed(x)  # B _T _H _W C
        x = self.pos_drop(x)  # B _T _H _W C

        out = []
        for layer, downsample in zip(self.layers, self.downsamples):
            x = layer(x.contiguous()).contiguous()  # B _T _H _W C
            out.append(x.permute(0, 4, 1, 2, 3))
            if downsample:
                x = downsample(x)
        return out[::-1]


class DecoderSwin(nn.Module):
    def __init__(self, num_layers=4, heads=["ss", "sal"], num_classes=7):
        super(DecoderSwin, self).__init__()

        self.upsampling = nn.Upsample(
            scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )

        self.convtsp1 = nn.Sequential(
            nn.Conv3d(
                768, 384, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False
            ),
            nn.ReLU(),
            self.upsampling,
        )

        x = 1 if num_layers == 1 else 3

        self.convtsp2 = nn.Sequential(
            nn.Conv3d(
                384,
                192,
                kernel_size=(x, 3, 3),
                stride=(x, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.ReLU(),
            self.upsampling,
        )

        x = 1 if num_layers < 4 else 5

        self.convtsp3 = nn.Sequential(
            nn.Conv3d(
                192,
                96,
                kernel_size=(x, 3, 3),
                stride=(x, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.ReLU(),
            self.upsampling,
        )

        self.num_classes = num_classes

        self.heads = heads

        if "sal" in heads:
            self.sal_head = self.setup_heads(num_layers, 1, nn.Sigmoid)
        if "ss" in heads:
            self.ss_head = self.setup_heads(num_layers, self.num_classes, nn.Identity)
        if "dt" in heads:
            self.dt_head = self.setup_heads(num_layers, 1, nn.Sigmoid)

    def setup_heads(self, num_layers, out_channels, activation):
        x = 1 if num_layers < 3 else 5
        layers = [
            (
                "conv3_1",
                nn.Conv3d(
                    96,
                    64,
                    kernel_size=(x, 3, 3),
                    stride=(x, 1, 1),
                    padding=(0, 1, 1),
                    bias=False,
                ),
            ),
            ("relu_1", nn.ReLU()),
            ("up_1", self.upsampling),
            (
                "conv3_2",
                nn.Conv3d(
                    64,
                    32,
                    kernel_size=(1, 3, 3),
                    stride=(2, 1, 1),
                    padding=(0, 1, 1),
                    bias=False,
                ),
            ),
            ("relu_2", nn.ReLU()),
            ("up_2", self.upsampling),
        ]
        if num_layers == 1:
            layers.append(
                (
                    "conv3_3",
                    nn.Conv3d(
                        32,
                        out_channels,
                        kernel_size=(1, 1, 1),
                        stride=(2, 1, 1),
                        bias=True,
                    ),
                )
            )
        else:
            layers.append(
                (
                    "conv3_3",
                    nn.Conv3d(
                        32,
                        out_channels,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1),
                        bias=True,
                    ),
                )
            )

        layers.append(("acc", activation()))

        conv_layer = nn.Sequential(OrderedDict(layers))
        return conv_layer

    def forward(self, y, head):
        if not isinstance(y, list):
            raise ValueError("ERROR: input to decoder should be a list!")
        if head not in self.heads:
            raise TypeError(f" {head} not in {self.heads} ")

        if len(y) >= 1:
            z = self.convtsp1(y[0])

        if len(y) >= 2:
            z = torch.cat((z, y[1]), 2)

        z = self.convtsp2(z)

        if len(y) >= 3:
            z = torch.cat((z, y[2]), 2)

        z = self.convtsp3(z)

        if len(y) == 4:
            z = torch.cat((z, y[3]), 2)

        out_channels = self.num_classes if head == "ss" else 1

        if head == "sal":
            z = self.sal_head(z)
        elif head == "ss":
            z = self.ss_head(z)
        elif head == "dt":
            z = self.dt_head(z)

        z = z.view(z.size(0), out_channels, z.size(3), z.size(4))
        return z


if __name__ == "__main__":
    with torch.no_grad():
        B, T = 2, 8
        task = torch.randn((B, T, 6)).cuda()
        x = torch.randn(B, T, 3, 224, 1120).cuda()
        model = DriverGaze360(img_size=(224, 1120), T=T, use_context=True).cuda()
        # model.compile()
        out = model(x, tasks=task, direction=torch.tensor([0] * B).cuda())
        print([v.shape for k, v in out.items()])
