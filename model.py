"""Model definitions for the underpainting to photorealistic translation task."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int, *, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    """Create a standard convolutional block used throughout the U-Net."""

    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ResidualBlock(nn.Module):
    """Residual block used in the bottleneck to preserve fine detail."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple arithmetic
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)


class UpBlock(nn.Module):
    """Upsampling block combining nearest-neighbour upsampling and convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_block(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.block(x)


class UnderpaintingToPhotoModel(nn.Module):
    """U-Net style encoder/decoder network with skip connections.

    The network receives a stack of feature maps describing the underpainting and
    synthesises a 3-channel RGB image.  Skip connections ensure that structural
    information present in the conditioning input is preserved all the way to the
    output layers, enforcing alignment between the rough painting and the final
    photorealistic rendering.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3, feature_channels: Tuple[int, ...] | None = None) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = (64, 128, 256, 512, 512)

        self.in_channels = in_channels
        self.out_channels = out_channels

        enc_layers: List[nn.Module] = []
        prev_channels = in_channels
        for channels in feature_channels:
            enc_layers.append(conv_block(prev_channels, channels, stride=2))
            enc_layers.append(conv_block(channels, channels))
            prev_channels = channels
        self.encoder = nn.ModuleList(enc_layers)

        self.bottleneck = nn.Sequential(
            ResidualBlock(feature_channels[-1]),
            ResidualBlock(feature_channels[-1]),
        )

        dec_layers: List[nn.Module] = []
        reversed_channels = list(reversed(feature_channels))
        for idx in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[idx]
            skip_ch = reversed_channels[idx + 1]
            dec_layers.append(UpBlock(in_ch, skip_ch))
            dec_layers.append(conv_block(skip_ch * 2, skip_ch))
        self.decoder = nn.ModuleList(dec_layers)

        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_channels[0], out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        # Encoder: downsample while storing skip activations.
        for idx in range(0, len(self.encoder), 2):
            out = self.encoder[idx](out)
            out = self.encoder[idx + 1](out)
            skips.append(out)

        out = self.bottleneck(out)

        # Decoder: upsample and concatenate skip connections.
        for idx in range(0, len(self.decoder), 2):
            out = self.decoder[idx](out)
            skip = skips.pop()
            # Spatial alignment: padding can cause off-by-one differences; adjust if needed.
            if skip.shape[-2:] != out.shape[-2:]:
                diff_y = skip.shape[-2] - out.shape[-2]
                diff_x = skip.shape[-1] - out.shape[-1]
                out = nn.functional.pad(out, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
            out = torch.cat([out, skip], dim=1)
            out = self.decoder[idx + 1](out)

        return self.output_layer(out)


__all__ = ["UnderpaintingToPhotoModel"]

