"""Project code: a small ResNet and the deployment pipeline that serves it.

The pipeline is what production runs: raw uint8 images (N, H, W, C) in, class
probabilities out, with the preprocessing inside the model so it can't drift.
"""
from __future__ import annotations

import torch
from torch import nn

CLASSES = ("red", "green", "blue")
MEAN = (0.45, 0.45, 0.45)
STD = (0.25, 0.25, 0.25)


class BasicBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.shortcut = nn.Identity()
        if stride != 1 or channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride, bias=False), nn.BatchNorm2d(channels_out)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, classes: int = 3, width: int = 16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.layer1 = BasicBlock(width, width)
        self.layer2 = BasicBlock(width, 2 * width, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * width, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layer2(self.layer1(self.stem(images)))
        return self.fc(torch.flatten(self.pool(x), 1))


class InferencePipeline(nn.Module):
    """uint8 NHWC images -> normalized NCHW floats -> ResNet -> probabilities."""

    def __init__(self, model: ResNet):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN))
        self.register_buffer("std", torch.tensor(STD))

    def forward(self, images_uint8: torch.Tensor) -> dict[str, torch.Tensor]:
        n, h, w, c = images_uint8.shape
        # Normalize as a (pixels, channels) matrix, in the model's dtype (buffers follow .to()).
        # Try `.view(n, h * w, c)`: fine for contiguous batches, but decoders and loaders hand
        # over non-contiguous ones, and the layout test catches the crash.
        pixels = images_uint8.reshape(n, h * w, c).to(self.mean.dtype) / 255
        pixels = (pixels - self.mean) / self.std
        logits = self.model(pixels.reshape(n, h, w, c).permute(0, 3, 1, 2))
        return {"logits": logits, "probs": logits.softmax(-1)}


def make_images(n: int, size: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 NHWC images whose label is the dominant color channel, with noise and a gradient."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 3, (n,), generator=g)
    base = torch.rand(n, size, size, 3, generator=g) * 0.5
    base[torch.arange(n), :, :, labels] += 0.4
    base += torch.linspace(0, 0.1, size).view(1, size, 1, 1)
    return (base.clamp(0, 1) * 255).to(torch.uint8), labels
