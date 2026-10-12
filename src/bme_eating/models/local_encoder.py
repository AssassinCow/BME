from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            input_channels,
            input_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=input_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_channels, output_channels, 1, bias=False)
        self.norm = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv1d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = self.depthwise(values)
        values = self.pointwise(values)
        values = self.norm(values)
        values = self.dropout(self.activation(values))
        return self.activation(values + residual)


class LocalEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        stem_channels: int,
        stem_kernel: int,
        stem_stride: int,
        block_channels: Sequence[int],
        block_kernels: Sequence[int],
        block_strides: Sequence[int],
        block_dilations: Sequence[int],
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                stem_channels,
                stem_kernel,
                stride=stem_stride,
                padding=(stem_kernel - 1) // 2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.SiLU(),
        )
        blocks: list[nn.Module] = []
        previous = stem_channels
        for channels, kernel, stride, dilation in zip(
            block_channels, block_kernels, block_strides, block_dilations
        ):
            blocks.append(
                DepthwiseResidualBlock(
                    previous, channels, kernel, stride, dilation, dropout
                )
            )
            previous = channels
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.Linear(previous * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values))
        mean = values.mean(dim=-1)
        standard_deviation = values.std(dim=-1, unbiased=False)
        maximum = values.amax(dim=-1)
        last = values[..., -1]
        return self.projection(torch.cat((mean, standard_deviation, maximum, last), dim=-1))
