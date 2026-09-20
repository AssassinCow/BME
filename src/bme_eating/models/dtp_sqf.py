from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def logits_to_probability_arrays(
    output: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert autocast logits to NumPy-compatible float32 probabilities."""

    return tuple(
        torch.sigmoid(output[name].float()).cpu().numpy()
        for name in ("state_logit", "start_logit", "end_logit")
    )


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


class DyadicPool(nn.Module):
    def __init__(self, embedding_dim: int, token_dim: int, bucket_counts: Sequence[int], dropout: float):
        super().__init__()
        self.bucket_counts = tuple(int(value) for value in bucket_counts)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 5 + 1, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, token_dim),
            nn.LayerNorm(token_dim),
        )

    @staticmethod
    def summarize(embedding: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(embedding.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (embedding * weights).sum(dim=1) / count
        variance = ((embedding - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
        standard_deviation = torch.sqrt(variance.clamp_min(1e-8))
        maximum = embedding.masked_fill(weights <= 0, -torch.inf).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

        reversed_valid = torch.flip(valid, dims=(1,))
        reverse_index = torch.argmax(reversed_valid.to(torch.int64), dim=1)
        last_index = embedding.shape[1] - 1 - reverse_index
        last = embedding.gather(
            1,
            last_index[:, None, None].expand(-1, 1, embedding.shape[-1]),
        ).squeeze(1)
        any_valid = valid.any(dim=1, keepdim=True)
        last = last * any_valid.to(last.dtype)

        time = torch.linspace(-1.0, 1.0, embedding.shape[1], device=embedding.device)
        time = time.view(1, -1, 1)
        time_mean = (time * weights).sum(dim=1, keepdim=True) / count.unsqueeze(1)
        centered_time = time - time_mean
        denominator = (centered_time.square() * weights).sum(dim=1).clamp_min(1e-6)
        slope = (
            centered_time * (embedding - mean.unsqueeze(1)) * weights
        ).sum(dim=1) / denominator
        valid_fraction = valid.to(embedding.dtype).mean(dim=1, keepdim=True)
        return torch.cat(
            (mean, standard_deviation, maximum, last, slope, valid_fraction), dim=-1
        )

    def forward(self, embedding: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        cursor = embedding.shape[1]
        for width in self.bucket_counts:
            start = cursor - width
            if start < 0:
                raise ValueError("Embedding sequence is shorter than configured dyadic buckets")
            summary = self.summarize(embedding[:, start:cursor], valid[:, start:cursor])
            tokens.append(self.projection(summary))
            cursor = start
        return torch.stack(tokens, dim=1)


def _bucket_average(values: torch.Tensor, valid: torch.Tensor, widths: Sequence[int]) -> torch.Tensor:
    output: list[torch.Tensor] = []
    cursor = values.shape[1]
    for width in widths:
        start = cursor - width
        weights = valid[:, start:cursor].to(values.dtype)
        average = (values[:, start:cursor] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        output.append(average)
        cursor = start
    return torch.stack(output, dim=1)


def _bucket_geometry(counts: Sequence[int], block_seconds: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    durations = tuple(int(count) * int(block_seconds) for count in counts)
    ages: list[int] = []
    elapsed = 0
    for duration in durations:
        ages.append(elapsed)
        elapsed += duration
    return durations, tuple(ages)


class DTPSQF(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        embedding_dim = int(config["embedding_dim"])
        token_dim = int(config["token_dim"])
        dropout = float(config["dropout"])
        self.motion_block_seconds = int(config.get("motion_block_seconds", 3))
        self.ppg_block_seconds = int(config.get("ppg_block_seconds", 15))
        self.motion_bucket_counts = tuple(int(value) for value in config["motion_bucket_counts"])
        self.ppg_bucket_counts = tuple(int(value) for value in config["ppg_bucket_counts"])
        self.future_context_seconds = int(config.get("future_context_seconds", 0))
        if self.motion_block_seconds <= 0 or self.ppg_block_seconds <= 0:
            raise ValueError("DTP block durations must be positive")
        if not self.motion_bucket_counts or min(self.motion_bucket_counts) <= 0:
            raise ValueError("Motion bucket counts must be non-empty and positive")
        if not self.ppg_bucket_counts or min(self.ppg_bucket_counts) <= 0:
            raise ValueError("PPG bucket counts must be non-empty and positive")
        if self.future_context_seconds and (
            self.future_context_seconds % self.motion_block_seconds != 0
            or self.future_context_seconds % self.ppg_block_seconds != 0
        ):
            raise ValueError("Future context must be divisible by both DTP block durations")
        self.motion_bucket_durations, self.motion_bucket_ages = _bucket_geometry(
            self.motion_bucket_counts, self.motion_block_seconds
        )
        self.ppg_bucket_durations, self.ppg_bucket_ages = _bucket_geometry(
            self.ppg_bucket_counts, self.ppg_block_seconds
        )

        self.motion_encoder = LocalEncoder(
            input_channels=12,
            stem_channels=48,
            stem_kernel=9,
            stem_stride=2,
            block_channels=(64, 96, 128),
            block_kernels=(7, 5, 3),
            block_strides=(2, 2, 2),
            block_dilations=(1, 2, 4),
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.ppg_encoder = LocalEncoder(
            input_channels=2,
            stem_channels=64,
            stem_kernel=1,
            stem_stride=1,
            block_channels=(64, 96, 128),
            block_kernels=(15, 9, 5),
            block_strides=(5, 3, 2),
            block_dilations=(1, 1, 2),
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.motion_pool = DyadicPool(
            embedding_dim, token_dim, self.motion_bucket_counts, dropout
        )
        self.ppg_pool = DyadicPool(embedding_dim, token_dim, self.ppg_bucket_counts, dropout)
        self.future_motion_pool = DyadicPool(embedding_dim, token_dim, (1,), dropout)
        self.future_ppg_pool = DyadicPool(embedding_dim, token_dim, (1,), dropout)
        self.ppg_gate = nn.Sequential(
            nn.Linear(embedding_dim + 8, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.ppg_projection = nn.Linear(token_dim, token_dim)
        self.ppg_missing = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.future_motion_missing = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.future_ppg_missing = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.query = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.modality_embedding = nn.Embedding(4, token_dim)
        self.maximum_scale = max(
            len(self.motion_bucket_counts), len(self.ppg_bucket_counts)
        ) + 1
        self.scale_embedding = nn.Embedding(self.maximum_scale + 1, token_dim)
        self.time_projection = nn.Sequential(nn.Linear(3, token_dim), nn.Tanh())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=int(config["attention_heads"]),
            dim_feedforward=int(config["feedforward_dim"]),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["transformer_layers"])
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.state_head = nn.Sequential(nn.Linear(token_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.boundary_head = nn.Sequential(
            nn.Linear(token_dim, 64), nn.SiLU(), nn.Linear(64, 2)
        )

    @staticmethod
    def _encode_blocks(encoder: nn.Module, blocks: torch.Tensor) -> torch.Tensor:
        batch, count, channels, samples = blocks.shape
        encoded = encoder(blocks.reshape(batch * count, channels, samples))
        return encoded.reshape(batch, count, -1)

    def _ppg_tokens(
        self,
        embedding: torch.Tensor,
        quality: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.ppg_gate(torch.cat((embedding, quality), dim=-1)).squeeze(-1)
        tokens = self.ppg_pool(embedding, valid > 0)
        bucket_gate = _bucket_average(gate, valid, self.ppg_bucket_counts).unsqueeze(-1)
        tokens = bucket_gate * self.ppg_projection(tokens) + (1.0 - bucket_gate) * self.ppg_missing
        return tokens, gate

    def _decorate(
        self,
        tokens: torch.Tensor,
        modality: list[int],
        scale: list[int],
        duration_seconds: list[float],
        age_seconds: list[float],
        valid_fraction: torch.Tensor,
    ) -> torch.Tensor:
        device = tokens.device
        modality_tensor = torch.tensor(modality, device=device)
        scale_tensor = torch.tensor(scale, device=device)
        duration = torch.tensor(duration_seconds, device=device, dtype=tokens.dtype).unsqueeze(0)
        age = torch.tensor(age_seconds, device=device, dtype=tokens.dtype).unsqueeze(0)
        duration = duration.expand(tokens.shape[0], -1)
        age = age.expand(tokens.shape[0], -1)
        time_features = torch.stack(
            (torch.log1p(duration), torch.log1p(torch.abs(age)), valid_fraction),
            dim=-1,
        )
        return (
            tokens
            + self.modality_embedding(modality_tensor).unsqueeze(0)
            + self.scale_embedding(scale_tensor).unsqueeze(0)
            + self.time_projection(time_features)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        motion_embedding = self._encode_blocks(self.motion_encoder, batch["motion_blocks"])
        ppg_embedding = self._encode_blocks(self.ppg_encoder, batch["ppg_blocks"])
        motion_valid = batch["motion_valid"] > 0
        ppg_valid = batch["ppg_valid"] > 0
        motion_tokens = self.motion_pool(motion_embedding, motion_valid)
        ppg_tokens, ppg_gate = self._ppg_tokens(
            ppg_embedding, batch["ppg_quality"], batch["ppg_valid"]
        )

        batch_size = motion_tokens.shape[0]
        if self.future_context_seconds > 0:
            future_motion_embedding = self._encode_blocks(
                self.motion_encoder, batch["future_motion_blocks"]
            )
            future_motion_summary = DyadicPool.summarize(
                future_motion_embedding, batch["future_motion_valid"] > 0
            )
            future_motion_token = self.future_motion_pool.projection(future_motion_summary).unsqueeze(1)
            future_ppg_embedding = self._encode_blocks(
                self.ppg_encoder, batch["future_ppg_blocks"]
            )
            future_ppg_summary = DyadicPool.summarize(
                future_ppg_embedding, batch["future_ppg_valid"] > 0
            )
            future_ppg_token = self.future_ppg_pool.projection(future_ppg_summary).unsqueeze(1)
            future_ppg_gate = self.ppg_gate(
                torch.cat((future_ppg_embedding, batch["future_ppg_quality"]), dim=-1)
            ).mean(dim=1, keepdim=True)
            future_ppg_token = (
                future_ppg_gate * self.ppg_projection(future_ppg_token)
                + (1.0 - future_ppg_gate) * self.future_ppg_missing
            )
            future_motion_valid = batch["future_motion_valid"].mean(dim=1, keepdim=True)
            future_ppg_valid = batch["future_ppg_valid"].mean(dim=1, keepdim=True)
        else:
            future_motion_token = self.future_motion_missing.expand(batch_size, -1, -1)
            future_ppg_token = self.future_ppg_missing.expand(batch_size, -1, -1)
            future_motion_valid = torch.zeros((batch_size, 1), device=motion_tokens.device)
            future_ppg_valid = torch.zeros((batch_size, 1), device=motion_tokens.device)

        query = self.query.expand(batch_size, -1, -1)
        tokens = torch.cat(
            (query, motion_tokens, ppg_tokens, future_motion_token, future_ppg_token), dim=1
        )
        motion_bucket_valid = torch.stack(
            [
                block.float().mean(dim=1)
                for block in self._split_valid(motion_valid, self.motion_bucket_counts)
            ],
            dim=1,
        )
        ppg_bucket_valid = torch.stack(
            [block.float().mean(dim=1) for block in self._split_valid(ppg_valid, self.ppg_bucket_counts)],
            dim=1,
        )
        valid_fraction = torch.cat(
            (
                torch.ones((batch_size, 1), device=tokens.device),
                motion_bucket_valid,
                ppg_bucket_valid,
                future_motion_valid,
                future_ppg_valid,
            ),
            dim=1,
        )
        tokens = self._decorate(
            tokens,
            modality=[0]
            + [1] * len(self.motion_bucket_counts)
            + [2] * len(self.ppg_bucket_counts)
            + [3, 3],
            scale=[0]
            + list(range(1, len(self.motion_bucket_counts) + 1))
            + list(range(1, len(self.ppg_bucket_counts) + 1))
            + [self.maximum_scale, self.maximum_scale],
            duration_seconds=[0]
            + list(self.motion_bucket_durations)
            + list(self.ppg_bucket_durations)
            + [self.future_context_seconds] * 2,
            age_seconds=[0]
            + list(self.motion_bucket_ages)
            + list(self.ppg_bucket_ages)
            + [-self.future_context_seconds] * 2,
            valid_fraction=valid_fraction,
        )
        encoded = self.transformer(tokens)
        query_output = self.output_norm(encoded[:, 0])
        boundaries = self.boundary_head(query_output)
        return {
            "state_logit": self.state_head(query_output).squeeze(-1),
            "start_logit": boundaries[:, 0],
            "end_logit": boundaries[:, 1],
            "ppg_gate": ppg_gate,
        }

    @staticmethod
    def _split_valid(valid: torch.Tensor, widths: Sequence[int]) -> list[torch.Tensor]:
        blocks: list[torch.Tensor] = []
        cursor = valid.shape[1]
        for width in widths:
            start = cursor - width
            blocks.append(valid[:, start:cursor])
            cursor = start
        return blocks
