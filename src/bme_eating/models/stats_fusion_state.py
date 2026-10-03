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


class TimewiseLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(values.transpose(1, 2).contiguous())
        return normalized.transpose(1, 2)


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
        self.normalization = TimewiseLayerNorm(channels)
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
            CausalDepthwiseBlock(channels, dilation, dropout) for dilation in self.dilations
        )
        self.output_norm = TimewiseLayerNorm(channels)

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
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        block_end_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, channels = values.shape
        if valid.shape != (batch, steps):
            raise ValueError("Completed-block validity has an incompatible shape")
        if block_end_indices.ndim != 2 or block_end_indices.shape[0] != batch:
            raise ValueError("long_block_end_indices has an incompatible shape")
        block_count = block_end_indices.shape[1]
        if block_count == 0:
            empty = values.new_zeros((batch, 0, self.projection[0].out_features))
            return empty, valid.new_zeros((batch, 0))

        endpoint_valid = (block_end_indices >= self.factor - 1) & (block_end_indices < steps)
        offsets = torch.arange(
            1 - self.factor,
            1,
            device=values.device,
            dtype=block_end_indices.dtype,
        )
        member_indices = block_end_indices.unsqueeze(-1) + offsets.view(1, 1, -1)
        member_indices = member_indices.clamp(0, max(0, steps - 1))
        flat_indices = member_indices.reshape(batch, block_count * self.factor)
        pooled_values = values.gather(
            1,
            flat_indices.unsqueeze(-1).expand(-1, -1, channels),
        ).reshape(batch, block_count, self.factor, channels)
        pooled_valid = valid.gather(1, flat_indices).reshape(batch, block_count, self.factor)
        pooled_valid = pooled_valid * endpoint_valid.unsqueeze(-1).to(pooled_valid.dtype)
        weights = pooled_valid.unsqueeze(-1).to(values.dtype)
        count = weights.sum(dim=2).clamp_min(1.0)
        mean = (pooled_values * weights).sum(dim=2) / count
        maximum = pooled_values.masked_fill(weights <= 0, -torch.inf).amax(dim=2)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        valid_bool = pooled_valid > 0
        reverse_index = valid_bool.flip(dims=(2,)).to(torch.int64).argmax(dim=2)
        last_index = self.factor - 1 - reverse_index
        gather_index = last_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, channels)
        last = pooled_values.gather(2, gather_index).squeeze(2)
        last = torch.where(valid_bool.any(dim=2, keepdim=True), last, torch.zeros_like(last))
        valid_fraction = pooled_valid.to(values.dtype).mean(dim=2, keepdim=True)
        pooled = self.projection(torch.cat((mean, maximum, last, valid_fraction), dim=-1))
        block_has_valid = endpoint_valid & valid_bool.any(dim=2)
        pooled = pooled * block_has_valid.unsqueeze(-1).to(pooled.dtype)
        return pooled, valid_fraction.squeeze(-1)

    def hold_completed(self, low_rate: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2 or indices.shape[0] != low_rate.shape[0]:
            raise ValueError("Completed-block hold indices have an incompatible shape")
        if low_rate.shape[1] == 0:
            return low_rate.new_zeros((low_rate.shape[0], indices.shape[1], low_rate.shape[-1]))
        if torch.any(indices >= low_rate.shape[1]):
            raise ValueError("Completed-block hold refers to a missing low-rate token")
        valid = indices >= 0
        indices = indices.clamp(0, low_rate.shape[1] - 1)
        held = low_rate.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, low_rate.shape[-1]),
        )
        return held * valid.unsqueeze(-1).to(held.dtype)

    def hold_completed_validity(
        self, low_rate_validity: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if indices.ndim != 2 or indices.shape[0] != low_rate_validity.shape[0]:
            raise ValueError("Completed-block validity hold indices have an incompatible shape")
        if low_rate_validity.shape[1] == 0:
            return low_rate_validity.new_zeros((low_rate_validity.shape[0], indices.shape[1]))
        if torch.any(indices >= low_rate_validity.shape[1]):
            raise ValueError("Completed-block validity hold refers to a missing token")
        valid = indices >= 0
        safe = indices.clamp(0, low_rate_validity.shape[1] - 1)
        held = low_rate_validity.gather(1, safe)
        return held * valid.to(held.dtype)


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
        self.separate_motion_branches = bool(config.get("separate_motion_branches", False))
        self.use_invariant_motion_branch = bool(config.get("use_invariant_motion_branch", False))
        self.use_mixstyle = bool(config.get("use_mixstyle", False))
        self.mixstyle_probability = float(config.get("mixstyle_probability", 0.15))
        self.mixstyle_alpha = float(config.get("mixstyle_alpha", 0.3))
        self.use_quality_conditioned_fusion = bool(
            config.get("use_quality_conditioned_fusion", False)
        )
        self.use_local_cross_scale_attention = bool(
            config.get("use_local_cross_scale_attention", False)
        )
        self.local_attention_window_steps = int(config.get("local_attention_window_steps", 8))
        if not 0.0 <= self.mixstyle_probability <= 1.0:
            raise ValueError("mixstyle_probability must lie in [0, 1]")
        if self.mixstyle_alpha <= 0:
            raise ValueError("mixstyle_alpha must be positive")
        if self.local_attention_window_steps < 0:
            raise ValueError("local_attention_window_steps must be non-negative")
        stable_features = tuple(str(value) for value in config.get("stable_feature_columns", ()))
        self.ppg_statistics_index = (
            stable_features.index("local_ppg_valid_fraction")
            if "local_ppg_valid_fraction" in stable_features
            else None
        )
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

        def motion_encoder(input_channels: int) -> LocalEncoder:
            return LocalEncoder(
                input_channels=input_channels,
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

        if self.separate_motion_branches:
            self.acc_encoder = motion_encoder(6)
            self.gyro_encoder = motion_encoder(6)
            self.acc_tcn = CausalTCN(motion_dim, hidden_dim, motion_dilations, dropout)
            self.gyro_tcn = CausalTCN(motion_dim, hidden_dim, motion_dilations, dropout)
            self.acc_missing = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.gyro_missing = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.gyro_gate_network = nn.Sequential(
                nn.Linear(2 * hidden_dim + 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.motion_merge_norm = nn.LayerNorm(hidden_dim)
            nn.init.constant_(self.gyro_gate_network[-1].bias, gate_bias)
        else:
            self.motion_encoder = motion_encoder(12)
            self.motion_tcn = CausalTCN(motion_dim, hidden_dim, motion_dilations, dropout)
            self.motion_missing = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if self.use_invariant_motion_branch:
            self.invariant_motion_encoder = motion_encoder(6)
            self.invariant_motion_tcn = CausalTCN(motion_dim, hidden_dim, motion_dilations, dropout)
            self.invariant_gate_network = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.constant_(self.invariant_gate_network[-1].bias, gate_bias)
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
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
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
            nn.Linear(hidden_dim + statistics_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.constant_(self.ppg_gate_network[-1].bias, gate_bias)
        nn.init.constant_(self.long_gate_network.bias, gate_bias)
        nn.init.constant_(self.statistics_gate_network[-1].bias, gate_bias)

        if self.use_quality_conditioned_fusion:
            self.quality_condition_network = nn.Sequential(
                nn.Linear(hidden_dim + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.quality_residual = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.constant_(self.quality_condition_network[-1].bias, gate_bias)
        if self.use_local_cross_scale_attention:
            attention_heads = 4 if hidden_dim % 4 == 0 else 1
            self.cross_scale_attention = nn.MultiheadAttention(
                hidden_dim,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_scale_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.cross_scale_gate = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
            nn.init.constant_(self.cross_scale_gate.bias, gate_bias)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.onset_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.offset_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.proposal_head = (
            nn.Linear(hidden_dim, 1) if bool(config.get("use_proposal_head", False)) else None
        )

    @staticmethod
    def _encode_blocks(encoder: nn.Module, blocks: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, samples = blocks.shape
        encoded = encoder(blocks.reshape(batch * steps, channels, samples))
        return encoded.reshape(batch, steps, -1)

    @staticmethod
    def _align_ppg(values: torch.Tensor, indices: torch.Tensor, steps: int) -> torch.Tensor:
        if indices.shape != (values.shape[0], steps):
            raise ValueError("ppg_to_motion_index has an incompatible shape")
        if torch.any(indices >= values.shape[1]):
            raise ValueError("ppg_to_motion_index refers to a missing completed block")
        safe = indices.clamp(0, max(0, values.shape[1] - 1))
        aligned = values.gather(1, safe.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
        return aligned * (indices >= 0).unsqueeze(-1).to(aligned.dtype)

    def _apply_mixstyle(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        if not self.training or not self.use_mixstyle or values.shape[0] < 2:
            return values
        if torch.rand((), device=values.device) >= self.mixstyle_probability:
            return values
        weights = valid.to(values.dtype).unsqueeze(-1)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (values * weights).sum(dim=1, keepdim=True) / count
        variance = ((values - mean) ** 2 * weights).sum(dim=1, keepdim=True) / count
        std = variance.clamp_min(1e-6).sqrt()
        shift = int(
            torch.randint(1, values.shape[0], (), device=values.device).item()
        )
        permutation = torch.arange(values.shape[0], device=values.device).roll(shift)
        lam = torch.distributions.Beta(
            self.mixstyle_alpha, self.mixstyle_alpha
        ).sample((values.shape[0], 1, 1)).to(values.device, values.dtype)
        mixed_mean = lam * mean + (1.0 - lam) * mean[permutation]
        mixed_std = lam * std + (1.0 - lam) * std[permutation]
        normalized = (values - mean) / std
        return (normalized * mixed_std + mixed_mean) * weights + values * (1.0 - weights)

    @property
    def short_receptive_field_seconds(self) -> int:
        tcn = self.acc_tcn if self.separate_motion_branches else self.motion_tcn
        return tcn.receptive_field_steps * self.motion_block_seconds

    @property
    def long_receptive_field_seconds(self) -> int:
        return self.long_tcn.receptive_field_steps * self.ppg_block_seconds

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        motion_blocks = batch["motion_blocks"]
        motion_valid = batch["motion_valid"].to(motion_blocks.dtype)
        steps = motion_blocks.shape[1]
        acc_valid = motion_blocks[:, :, 6:9].mean(dim=(2, 3)).to(motion_blocks.dtype)
        gyro_valid = motion_blocks[:, :, 9:12].mean(dim=(2, 3)).to(motion_blocks.dtype)
        acc_present = acc_valid > 0
        gyro_present = gyro_valid > 0
        if self.separate_motion_branches:
            acc_blocks = torch.cat((motion_blocks[:, :, :3], motion_blocks[:, :, 6:9]), dim=2)
            gyro_blocks = torch.cat((motion_blocks[:, :, 3:6], motion_blocks[:, :, 9:12]), dim=2)
            acc = self.acc_tcn(self._encode_blocks(self.acc_encoder, acc_blocks))
            gyro = self.gyro_tcn(self._encode_blocks(self.gyro_encoder, gyro_blocks))
            acc = torch.where(acc_present.unsqueeze(-1), acc, self.acc_missing)
            gyro = torch.where(gyro_present.unsqueeze(-1), gyro, self.gyro_missing)
            gyro_gate = gyro_present.unsqueeze(-1).to(gyro.dtype) * torch.sigmoid(
                self.gyro_gate_network(
                    torch.cat(
                        (acc, gyro, acc_valid.unsqueeze(-1), gyro_valid.unsqueeze(-1)), dim=-1
                    )
                )
            )
            motion = self.motion_merge_norm(acc + gyro_gate * gyro)
            motion_fusion_valid = torch.maximum(acc_valid, gyro_valid)
        else:
            motion = self._encode_blocks(self.motion_encoder, motion_blocks)
            motion = self.motion_tcn(motion)
            motion = torch.where((motion_valid > 0).unsqueeze(-1), motion, self.motion_missing)
            motion_fusion_valid = motion_valid
            gyro_gate = torch.zeros_like(motion)
        invariant_gate = torch.zeros_like(motion)
        if self.use_invariant_motion_branch:
            invariant_blocks = batch.get("motion_invariant_blocks")
            if invariant_blocks is None:
                raise ValueError(
                    "The physical-unit invariant motion branch requires motion_invariant_blocks"
                )
            invariant_blocks = invariant_blocks.to(motion_blocks.dtype)
            invariant = self.invariant_motion_tcn(
                self._encode_blocks(self.invariant_motion_encoder, invariant_blocks)
            )
            invariant_gate = (motion_fusion_valid > 0).unsqueeze(-1).to(
                motion.dtype
            ) * torch.sigmoid(self.invariant_gate_network(torch.cat((motion, invariant), dim=-1)))
            motion = motion + invariant_gate * invariant
        if not self.use_motion:
            motion = torch.zeros_like(motion)
            motion_fusion_valid = torch.zeros_like(motion_valid)
            acc_valid = torch.zeros_like(acc_valid)
            gyro_valid = torch.zeros_like(gyro_valid)

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
            ppg_gate_aligned = self._align_ppg(ppg_gate.unsqueeze(-1), ppg_indices, steps).squeeze(
                -1
            )
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
                    acc_valid.unsqueeze(-1),
                    gyro_valid.unsqueeze(-1),
                    ppg_fusion_valid.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        statistics_input = batch["statistics"].to(short.dtype)
        if not self.use_ppg and self.ppg_statistics_index is not None:
            statistics_input = statistics_input.clone()
            statistics_input[..., self.ppg_statistics_index] = 0.0
            statistics_input[..., 12 + self.ppg_statistics_index] = 1.0
        statistics_missing = statistics_input[..., 12:]
        if not self.use_ppg and self.ppg_statistics_index is not None:
            active_statistics = torch.ones(
                statistics_missing.shape[-1], dtype=torch.bool, device=statistics_missing.device
            )
            active_statistics[self.ppg_statistics_index] = False
            statistics_missing = statistics_missing[..., active_statistics]
        statistics_missing_fraction = statistics_missing.mean(dim=-1)
        statistics_reliability = 1.0 - statistics_missing_fraction
        statistics = self.statistics_projection(statistics_input)
        statistics = self.statistics_tcn(statistics)
        if not self.use_statistics:
            statistics = torch.zeros_like(statistics)
            statistics_reliability = torch.zeros_like(statistics_reliability)
        quality = torch.stack(
            (acc_valid, gyro_valid, ppg_fusion_valid, statistics_reliability), dim=-1
        )
        short = self._apply_mixstyle(short, torch.maximum(motion_fusion_valid, ppg_fusion_valid) > 0)
        quality_conditioned_gate = torch.zeros_like(short)
        if self.use_quality_conditioned_fusion:
            quality_conditioned_gate = torch.sigmoid(
                self.quality_condition_network(torch.cat((short, quality), dim=-1))
            )
            short = short + quality_conditioned_gate * self.quality_residual(short)
        statistics_gate = torch.sigmoid(
            self.statistics_gate_network(torch.cat((short, statistics, quality), dim=-1))
        )

        statistics_gate_scalar = statistics_gate.mean(dim=-1, keepdim=True)
        gated_statistics = (
            statistics * statistics_gate_scalar * statistics_reliability.unsqueeze(-1)
        )
        combined = torch.cat((short, gated_statistics), dim=-1)
        combined_valid = torch.maximum(
            torch.maximum(motion_fusion_valid, ppg_fusion_valid), statistics_reliability
        )
        block_end_indices = batch.get("long_block_end_indices")
        if block_end_indices is None:
            raise ValueError("StatsFusion requires session-phased long_block_end_indices")
        low, low_validity = self.long_pool(combined, combined_valid, block_end_indices)
        if low.shape[1]:
            if ppg_indices is None:
                raise ValueError("StatsFusion requires session-phased completed-block mapping")
            long = self.long_pool.hold_completed(self.long_tcn(low), ppg_indices)
            long_validity = self.long_pool.hold_completed_validity(
                low_validity, ppg_indices
            )
        else:
            long = short.new_zeros(short.shape)
            long_validity = combined_valid.new_zeros(combined_valid.shape)
        if not self.use_long_context:
            long = torch.zeros_like(long)
            long_validity = torch.zeros_like(long_validity)
        cross_scale = torch.zeros_like(short)
        if self.use_local_cross_scale_attention and long.shape[1]:
            steps = long.shape[1]
            indices = torch.arange(steps, device=long.device)
            distance = indices.unsqueeze(1) - indices.unsqueeze(0)
            allowed = (distance >= 0) & (distance <= self.local_attention_window_steps)
            attention_mask = ~allowed
            key_padding_mask = ~(long_validity > 0)
            no_valid = key_padding_mask.all(dim=1)
            safe_key_padding_mask = key_padding_mask.clone()
            safe_key_padding_mask[no_valid] = False
            cross_scale, _ = self.cross_scale_attention(
                short,
                long,
                long,
                attn_mask=attention_mask,
                key_padding_mask=safe_key_padding_mask,
                need_weights=False,
            )
            cross_scale = self.cross_scale_projection(cross_scale)
            cross_scale = torch.where(
                no_valid[:, None, None], torch.zeros_like(cross_scale), cross_scale
            )
            cross_scale_gate = torch.sigmoid(
                self.cross_scale_gate(torch.cat((short, long), dim=-1))
            )
            cross_scale = (
                cross_scale_gate
                * cross_scale
                * long_validity.unsqueeze(-1).to(cross_scale.dtype)
            )
        long_gate = torch.sigmoid(self.long_gate_network(torch.cat((short, long), dim=-1)))
        final = self.final_norm(
            short
            + long_gate * self.long_to_hidden(long)
            + cross_scale
            + statistics_gate
            * statistics_reliability.unsqueeze(-1)
            * self.statistics_to_hidden(statistics)
        )
        active_validity: list[torch.Tensor] = []
        if self.use_motion:
            active_validity.extend((acc_valid, gyro_valid))
        if self.use_ppg:
            active_validity.append(ppg_valid_aligned)
        if self.use_statistics:
            active_validity.append(1.0 - statistics_missing_fraction)
        missing_fraction = 1.0 - torch.stack(active_validity, dim=0).mean(dim=0)
        result = {
            "state_hidden": final,
            "state_logit": self.state_head(final).squeeze(-1),
            "onset_logit": self.onset_head(final).squeeze(-1),
            "offset_logit": self.offset_head(final).squeeze(-1),
            "ppg_gate": ppg_gate_aligned,
            "statistics_gate": statistics_gate.mean(dim=-1),
            "long_gate": long_gate.mean(dim=-1),
            "missing_fraction": missing_fraction,
            "active_modality_missing_fraction": missing_fraction,
            "motion_valid_fraction": motion_valid,
            "motion_present": (motion_valid > 0).to(motion_valid.dtype),
            "acc_valid_fraction": acc_valid,
            "gyro_valid_fraction": gyro_valid,
            "gyro_gate": gyro_gate.mean(dim=-1),
            "invariant_gate": invariant_gate.mean(dim=-1),
            "modality_quality_gate": quality_conditioned_gate.mean(dim=-1),
            "ppg_valid_fraction": ppg_valid_aligned,
            "statistics_missing_fraction": statistics_missing_fraction,
        }
        if self.proposal_head is not None:
            result["proposal_logit"] = self.proposal_head(final).squeeze(-1)
        return result
