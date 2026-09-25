from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bme_eating.models.dtp_sqf import LocalEncoder


def temporal_receptive_field(dilations: Sequence[int], kernel_size: int = 3) -> int:
    if kernel_size <= 0 or not dilations:
        raise ValueError("A positive kernel and at least one dilation are required")
    if min(int(value) for value in dilations) <= 0:
        raise ValueError("Dilations must be positive")
    return 1 + (kernel_size - 1) * sum(int(value) for value in dilations)


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TimewiseGroupNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, steps = values.shape
        values = values.transpose(1, 2).reshape(batch * steps, channels, 1)
        values = self.normalization(values)
        return values.reshape(batch, steps, channels).transpose(1, 2)


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=self.dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.normalization = TimewiseGroupNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(F.pad(values, (2 * self.dilation, 0)))
        update = self.pointwise(update)
        update = self.dropout(F.silu(self.normalization(update)))
        return values + update


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(channels, dilation, dropout)
            for dilation in self.dilations
        )
        self.output_norm = TimewiseGroupNorm(channels)

    @property
    def receptive_field_steps(self) -> int:
        return temporal_receptive_field(self.dilations)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        return F.silu(self.output_norm(values)).transpose(1, 2)


class CausalCompletedBlockPool(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, factor: int) -> None:
        super().__init__()
        self.factor = int(factor)
        if self.factor <= 0:
            raise ValueError("Long-context pooling factor must be positive")
        self.projection = nn.Sequential(
            nn.Linear(3 * input_dim + 1, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels = values.shape
        complete = steps // self.factor
        if complete == 0:
            empty = values.new_zeros((batch, 0, self.projection[0].out_features))
            return empty, valid.new_zeros((batch, 0))
        values = values[:, : complete * self.factor].reshape(
            batch, complete, self.factor, channels
        )
        valid = valid[:, : complete * self.factor].reshape(batch, complete, self.factor)
        weights = valid.unsqueeze(-1).to(values.dtype)
        count = weights.sum(dim=2).clamp_min(1.0)
        mean = (values * weights).sum(dim=2) / count
        maximum = values.masked_fill(weights <= 0, -torch.inf).amax(dim=2)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        last = values[:, :, -1]
        valid_fraction = valid.to(values.dtype).mean(dim=2, keepdim=True)
        pooled = self.projection(torch.cat((mean, maximum, last, valid_fraction), dim=-1))
        return pooled, valid_fraction.squeeze(-1)

    def hold_completed(
        self, low_rate: torch.Tensor, high_steps: int
    ) -> torch.Tensor:
        if low_rate.shape[1] == 0:
            return low_rate.new_zeros((low_rate.shape[0], high_steps, low_rate.shape[-1]))
        indices = torch.div(
            torch.arange(high_steps, device=low_rate.device) + 1,
            self.factor,
            rounding_mode="floor",
        ) - 1
        valid = indices >= 0
        indices = indices.clamp(0, low_rate.shape[1] - 1)
        held = low_rate.index_select(1, indices)
        return held * valid.view(1, -1, 1).to(held.dtype)


class StatsFusionStateModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        if int(config.get("future_context_seconds", 0)) != 0:
            raise ValueError("StatsFusion state inference must remain causal")
        self.motion_block_seconds = int(config.get("motion_block_seconds", 3))
        self.ppg_block_seconds = int(config.get("ppg_block_seconds", 15))
        self.statistics_columns = tuple(config.get("stable_feature_columns", ()))
        if len(self.statistics_columns) != 12:
            raise ValueError("StatsFusion requires the fixed 12 statistical features")
        self.use_motion = bool(config.get("use_motion", True))
        self.use_ppg = bool(config.get("use_ppg", True))
        self.use_statistics = bool(config.get("use_statistics", True))
        self.use_long_context = bool(config.get("use_long_context", True))
        if not self.use_motion and not self.use_ppg:
            raise ValueError("At least one raw sensor branch must be enabled")

        motion_dim = int(config.get("motion_embedding_dim", 64))
        ppg_dim = int(config.get("ppg_embedding_dim", 48))
        hidden_dim = int(config.get("hidden_dim", 64))
        statistics_dim = int(config.get("statistics_dim", 32))
        dropout = float(config.get("dropout", 0.1))
        gate_bias = float(config.get("gate_initial_bias", -2.0))
        motion_dilations = config.get("motion_dilations", (1, 2, 4, 8, 16, 32))
        ppg_dilations = config.get("ppg_dilations", (1, 2, 4, 8))
        statistics_dilations = config.get("statistics_dilations", (1, 2, 4, 8))
        long_dilations = config.get("long_dilations", (1, 2, 4, 8, 16, 32))

        self.motion_encoder = LocalEncoder(
            input_channels=12,
            stem_channels=48,
            stem_kernel=9,
            stem_stride=2,
            block_channels=(64, 96, 128),
            block_kernels=(7, 5, 3),
            block_strides=(2, 2, 2),
            block_dilations=(1, 2, 4),
            embedding_dim=motion_dim,
            dropout=dropout,
        )
        self.ppg_encoder = LocalEncoder(
            input_channels=2,
            stem_channels=48,
            stem_kernel=15,
            stem_stride=5,
            block_channels=(64, 96, 128),
            block_kernels=(9, 7, 5),
            block_strides=(3, 2, 2),
            block_dilations=(1, 1, 2),
            embedding_dim=ppg_dim,
            dropout=dropout,
        )
        self.motion_tcn = CausalTCN(motion_dim, hidden_dim, motion_dilations, dropout)
        self.ppg_tcn = CausalTCN(ppg_dim, hidden_dim, ppg_dilations, dropout)
        self.ppg_missing = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.ppg_gate_network = nn.Sequential(
            nn.Linear(hidden_dim + 8, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        self.statistics_projection = nn.Sequential(
            nn.Linear(24, 32),
            nn.SiLU(),
            nn.Linear(32, statistics_dim),
            nn.LayerNorm(statistics_dim),
            nn.SiLU(),
        )
        self.statistics_tcn = CausalTCN(
            statistics_dim, statistics_dim, statistics_dilations, dropout
        )
        self.statistics_to_hidden = nn.Linear(statistics_dim, hidden_dim, bias=False)

        self.short_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        pool_factor = int(config.get("long_pool_factor", 5))
        if self.motion_block_seconds * pool_factor != self.ppg_block_seconds:
            raise ValueError(
                "PPG blocks and long-context pooling must share the completed-block cadence"
            )
        self.long_pool = CausalCompletedBlockPool(
            hidden_dim + statistics_dim, hidden_dim, pool_factor
        )
        self.long_tcn = CausalTCN(hidden_dim, hidden_dim, long_dilations, dropout)
        self.long_to_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.long_gate_network = nn.Linear(2 * hidden_dim, hidden_dim)
        self.statistics_gate_network = nn.Sequential(
            nn.Linear(hidden_dim + statistics_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.constant_(self.ppg_gate_network[-1].bias, gate_bias)
        nn.init.constant_(self.long_gate_network.bias, gate_bias)
        nn.init.constant_(self.statistics_gate_network[-1].bias, gate_bias)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.onset_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.offset_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    @staticmethod
    def _encode_blocks(encoder: nn.Module, blocks: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, samples = blocks.shape
        encoded = encoder(blocks.reshape(batch * steps, channels, samples))
        return encoded.reshape(batch, steps, -1)

    @staticmethod
    def _align_ppg(values: torch.Tensor, indices: torch.Tensor, steps: int) -> torch.Tensor:
        if indices.shape != (values.shape[0], steps):
            raise ValueError("ppg_to_motion_index has an incompatible shape")
        safe = indices.clamp(0, max(0, values.shape[1] - 1))
        aligned = values.gather(1, safe.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
        return aligned * (indices >= 0).unsqueeze(-1).to(aligned.dtype)

    @property
    def short_receptive_field_seconds(self) -> int:
        return self.motion_tcn.receptive_field_steps * self.motion_block_seconds

    @property
    def long_receptive_field_seconds(self) -> int:
        return self.long_tcn.receptive_field_steps * self.ppg_block_seconds

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        motion_blocks = batch["motion_blocks"]
        motion_valid = batch["motion_valid"].to(motion_blocks.dtype)
        steps = motion_blocks.shape[1]
        motion = self._encode_blocks(self.motion_encoder, motion_blocks)
        motion = self.motion_tcn(motion * motion_valid.unsqueeze(-1))
        motion_fusion_valid = motion_valid
        if not self.use_motion:
            motion = torch.zeros_like(motion)
            motion_fusion_valid = torch.zeros_like(motion_valid)

        ppg_blocks = batch["ppg_blocks"]
        ppg_quality = batch["ppg_quality"].to(ppg_blocks.dtype)
        ppg_valid = batch["ppg_valid"].to(ppg_blocks.dtype)
        ppg = self.ppg_tcn(self._encode_blocks(self.ppg_encoder, ppg_blocks))
        ppg_gate = ppg_valid * torch.sigmoid(
            self.ppg_gate_network(torch.cat((ppg, ppg_quality), dim=-1)).squeeze(-1)
        )
        ppg = ppg_gate.unsqueeze(-1) * ppg + (1.0 - ppg_valid).unsqueeze(-1) * self.ppg_missing
        ppg_indices = batch.get("ppg_to_motion_index")
        if ppg_indices is None:
            if ppg.shape[1] != steps:
                raise ValueError("Unaligned PPG tokens require ppg_to_motion_index")
            ppg_aligned = ppg
            ppg_gate_aligned = ppg_gate
            ppg_valid_aligned = ppg_valid
        else:
            ppg_aligned = self._align_ppg(ppg, ppg_indices, steps)
            ppg_gate_aligned = self._align_ppg(
                ppg_gate.unsqueeze(-1), ppg_indices, steps
            ).squeeze(-1)
            ppg_valid_aligned = self._align_ppg(
                ppg_valid.unsqueeze(-1), ppg_indices, steps
            ).squeeze(-1)
        if not self.use_ppg:
            ppg_aligned = torch.zeros_like(ppg_aligned)
            ppg_gate_aligned = torch.zeros_like(ppg_gate_aligned)
            ppg_fusion_valid = torch.zeros_like(ppg_valid_aligned)
        else:
            ppg_fusion_valid = ppg_valid_aligned

        short = self.short_fusion(
            torch.cat(
                (
                    motion,
                    ppg_aligned,
                    motion_fusion_valid.unsqueeze(-1),
                    ppg_fusion_valid.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        statistics = self.statistics_projection(batch["statistics"].to(short.dtype))
        statistics = self.statistics_tcn(statistics)
        if not self.use_statistics:
            statistics = torch.zeros_like(statistics)
        quality = torch.stack((motion_fusion_valid, ppg_fusion_valid), dim=-1)
        statistics_gate = torch.sigmoid(
            self.statistics_gate_network(torch.cat((short, statistics, quality), dim=-1))
        )

        combined = torch.cat((short, statistics), dim=-1)
        combined_valid = torch.maximum(motion_fusion_valid, ppg_fusion_valid)
        low, _ = self.long_pool(combined, combined_valid)
        if low.shape[1]:
            long = self.long_pool.hold_completed(self.long_tcn(low), steps)
        else:
            long = short.new_zeros(short.shape)
        if not self.use_long_context:
            long = torch.zeros_like(long)
        long_gate = torch.sigmoid(self.long_gate_network(torch.cat((short, long), dim=-1)))
        final = self.final_norm(
            short
            + long_gate * self.long_to_hidden(long)
            + statistics_gate * self.statistics_to_hidden(statistics)
        )
        missing_fraction = 1.0 - 0.5 * (motion_valid + ppg_valid_aligned)
        return {
            "state_logit": self.state_head(final).squeeze(-1),
            "onset_logit": self.onset_head(final).squeeze(-1),
            "offset_logit": self.offset_head(final).squeeze(-1),
            "ppg_gate": ppg_gate_aligned,
            "statistics_gate": statistics_gate.mean(dim=-1),
            "long_gate": long_gate.mean(dim=-1),
            "missing_fraction": missing_fraction,
        }
