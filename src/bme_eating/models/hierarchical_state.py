from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bme_eating.models.dtp_sqf import LocalEncoder


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TimewiseGroupNorm(nn.Module):
    """Apply GroupNorm independently at every causal time step."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, steps = values.shape
        normalized = self.normalization(
            values.transpose(1, 2).reshape(batch * steps, channels, 1)
        )
        return normalized.reshape(batch, steps, channels).transpose(1, 2)


class CausalDualBranchBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=self.dilation,
            groups=channels,
            bias=False,
        )
        self.local = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
            bias=False,
        )
        self.projection = nn.Conv1d(2 * channels, channels, kernel_size=1, bias=False)
        self.norm = TimewiseGroupNorm(channels)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal(values: torch.Tensor, convolution: nn.Conv1d, dilation: int) -> torch.Tensor:
        return convolution(F.pad(values, (2 * dilation, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dilated = F.silu(self._causal(values, self.dilated, self.dilation))
        local = F.silu(self._causal(values, self.local, 1))
        update = self.projection(torch.cat((dilated, local), dim=1))
        update = self.dropout(F.silu(self.norm(update)))
        return values + update


class CausalTemporalBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            CausalDualBranchBlock(channels, int(dilation), dropout) for dilation in dilations
        )
        self.output_norm = TimewiseGroupNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        return F.silu(self.output_norm(values)).transpose(1, 2)


class HierarchicalStateModel(nn.Module):
    """Causal state model retaining the full block-resolution history."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.motion_block_seconds = int(config.get("motion_block_seconds", 3))
        self.ppg_block_seconds = int(config.get("ppg_block_seconds", 15))
        self.motion_bucket_counts = tuple(int(v) for v in config["motion_bucket_counts"])
        self.ppg_bucket_counts = tuple(int(v) for v in config["ppg_bucket_counts"])
        self.future_context_seconds = int(config.get("future_context_seconds", 0))
        if self.future_context_seconds != 0:
            raise ValueError("The hierarchical state model must remain causal")
        if sum(self.motion_bucket_counts) != 127:
            raise ValueError("Hierarchical motion history must contain exactly 127 blocks")
        if sum(self.ppg_bucket_counts) != 31:
            raise ValueError("Hierarchical PPG history must contain exactly 31 blocks")

        local_dim = int(config.get("local_embedding_dim", 64))
        channels = int(config.get("tcn_channels", 64))
        embedding_dim = int(config.get("state_embedding_dim", 32))
        dropout = float(config.get("dropout", 0.1))
        self.stable_feature_columns = tuple(config.get("stable_feature_columns", ()))
        self.use_stable_state_features = bool(config.get("use_stable_state_features", False))

        self.motion_encoder = LocalEncoder(
            input_channels=12,
            stem_channels=48,
            stem_kernel=9,
            stem_stride=2,
            block_channels=(64, 96, 128),
            block_kernels=(7, 5, 3),
            block_strides=(2, 2, 2),
            block_dilations=(1, 2, 4),
            embedding_dim=local_dim,
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
            embedding_dim=local_dim,
            dropout=dropout,
        )
        self.motion_tcn = CausalTemporalBranch(
            local_dim, channels, config.get("motion_dilations", (1, 2, 4, 8, 16, 32)), dropout
        )
        self.ppg_tcn = CausalTemporalBranch(
            local_dim, channels, config.get("ppg_dilations", (1, 2, 4, 8)), dropout
        )
        self.ppg_gate = nn.Sequential(
            nn.Linear(local_dim + 8, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.ppg_missing = nn.Parameter(torch.zeros(1, 1, local_dim))
        self.stable_projection = None
        stable_dim = 0
        if self.use_stable_state_features:
            if not self.stable_feature_columns:
                raise ValueError("Stable state features require configured feature columns")
            self.stable_projection = nn.Sequential(
                nn.Linear(len(self.stable_feature_columns), 32),
                nn.SiLU(),
                nn.Linear(32, 16),
                nn.LayerNorm(16),
            )
            stable_dim = 16
        fusion_dim = 2 * channels + 8 + 2 + stable_dim
        self.state_projection = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.state_head = nn.Linear(embedding_dim, 1)
        self.start_head = nn.Sequential(nn.Linear(embedding_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.end_head = nn.Sequential(nn.Linear(embedding_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.history_head = nn.Linear(channels, 1)

    @staticmethod
    def _encode_blocks(encoder: nn.Module, blocks: torch.Tensor) -> torch.Tensor:
        batch, count, channels, samples = blocks.shape
        encoded = encoder(blocks.reshape(batch * count, channels, samples))
        return encoded.reshape(batch, count, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        motion = self._encode_blocks(self.motion_encoder, batch["motion_blocks"])
        motion = motion * batch["motion_valid"].unsqueeze(-1).to(motion.dtype)
        motion_sequence = self.motion_tcn(motion)

        ppg = self._encode_blocks(self.ppg_encoder, batch["ppg_blocks"])
        ppg_valid = batch["ppg_valid"].unsqueeze(-1).to(ppg.dtype)
        gate = self.ppg_gate(torch.cat((ppg, batch["ppg_quality"].to(ppg.dtype)), dim=-1))
        gate = gate * ppg_valid
        ppg = gate * ppg + (1.0 - gate) * self.ppg_missing
        ppg_sequence = self.ppg_tcn(ppg)

        ppg_quality = batch["ppg_quality"].to(ppg.dtype)
        ppg_valid_fraction = batch["ppg_valid"].to(ppg.dtype).mean(dim=1, keepdim=True)
        motion_valid_fraction = batch["motion_valid"].to(ppg.dtype).mean(dim=1, keepdim=True)
        fusion = [
            motion_sequence[:, -1],
            ppg_sequence[:, -1],
            ppg_quality[:, -1],
            ppg_valid_fraction,
            motion_valid_fraction,
        ]
        if self.stable_projection is not None:
            if "stable_features" not in batch:
                raise ValueError("Stable state feature tensor is missing from the batch")
            fusion.append(self.stable_projection(batch["stable_features"].to(ppg.dtype)))
        embedding = self.state_projection(torch.cat(fusion, dim=-1))
        return {
            "state_logit": self.state_head(embedding).squeeze(-1),
            "start_logit": self.start_head(embedding).squeeze(-1),
            "end_logit": self.end_head(embedding).squeeze(-1),
            "state_embedding": embedding,
            "state_history_logit": self.history_head(motion_sequence).squeeze(-1),
            "ppg_gate": gate.squeeze(-1),
            "missing_fraction": 1.0 - 0.5 * (ppg_valid_fraction + motion_valid_fraction),
        }
