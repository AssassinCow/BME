from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bme_eating.models.stats_fusion_state import (
    CausalTCN,
    StatsFusionStateModel,
)


class CausalSeparableBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 3, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.depthwise(F.pad(features.transpose(1, 2), (2, 0)))
        values = F.silu(self.pointwise(values)).transpose(1, 2)
        return features + self.dropout(values)


class CompletedScale(nn.Module):
    receptive_field_steps = 5

    def __init__(self, factor: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.factor = factor
        self.temporal = nn.Sequential(CausalSeparableBlock(channels, dropout),
                                      CausalSeparableBlock(channels, dropout))
        self.head = nn.Linear(channels, 1)

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, masks = [], []
        for sequence, mask, position in zip(features, valid, positions):
            ends = torch.nonzero((position + 1) % self.factor == 0).flatten()
            ends = ends[ends >= self.factor - 1]
            if not len(ends):
                outputs.append(sequence.new_zeros(len(sequence)))
                masks.append(mask.new_zeros(len(sequence)))
                continue
            members = ends[:, None] + torch.arange(
                1 - self.factor, 1, device=sequence.device
            )
            weights = mask[members]
            pooled = (sequence[members] * weights[..., None]).sum(1)
            pooled = pooled / weights.sum(1).clamp_min(1)[:, None]
            low = self.head(self.temporal(pooled[None])).squeeze(0).squeeze(-1)
            mapping = torch.searchsorted(ends, torch.arange(len(sequence), device=ends.device),
                                         right=True) - 1
            eligible = mapping >= 0
            mapping = mapping.clamp_min(0)
            outputs.append(low[mapping] * eligible)
            masks.append(weights.mean(1)[mapping] * eligible)
        return torch.stack(outputs), torch.stack(masks)


def candidate_roi(
    semantic: torch.Tensor,
    local: torch.Tensor,
    times: torch.Tensor,
    valid: torch.Tensor,
    start_ms: int,
    end_ms: int,
    decision_ms: int,
) -> torch.Tensor:
    if not start_ms < end_ms <= decision_ms:
        raise ValueError("Candidate ROI has invalid causal coordinates")
    parts = ((start_ms - 60_000, start_ms, 4), (start_ms, end_ms, 16),
             (end_ms, min(end_ms + 30_000, decision_ms), 4))
    tokens = []
    combined = torch.cat((semantic, local), -1)
    for left, right, bins in parts:
        edges = torch.linspace(left, right, bins + 1, dtype=torch.float64, device=times.device)
        for index in range(bins):
            selected = (times > edges[index]) & (times <= edges[index + 1]) & (times <= decision_ms)
            weights = valid[selected].to(combined.dtype)
            values = combined[selected]
            if len(values) and bool((weights > 0).any()):
                mean = (values * weights[:, None]).sum(0) / weights.sum().clamp_min(1)
                maximum = values.masked_fill(weights[:, None] <= 0, -torch.inf).amax(0)
                fraction = weights.mean()
            else:
                mean = combined.new_zeros(combined.shape[-1])
                maximum = mean
                fraction = combined.new_zeros(())
            center = (float(edges[index]) + float(edges[index + 1])) / 2
            relative = (center - start_ms) / max(end_ms - start_ms, 3000)
            tokens.append(torch.cat((mean, maximum, fraction[None],
                                     combined.new_tensor([relative]))))
    return torch.stack(tokens)


class JointCandidateHead(nn.Module):
    def __init__(self, channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Linear(4 * channels + 2, channels)
        self.temporal = CausalTCN(channels, channels, (1, 1), dropout)
        self.query = nn.Parameter(torch.randn(1, 1, channels) * 0.02)
        self.pool = nn.MultiheadAttention(channels, 4, dropout=dropout, batch_first=True)
        self.metadata = nn.Sequential(nn.Linear(12, channels), nn.SiLU())
        self.classifier = nn.Sequential(nn.Linear(2 * channels, 128), nn.SiLU(),
                                        nn.Dropout(dropout), nn.Linear(128, 64), nn.SiLU())
        self.event = nn.Linear(64, 1)
        self.quality = nn.Linear(64, 1)

    def forward(self, roi: torch.Tensor, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.temporal(self.input(roi))
        invalid = roi[..., -2] <= 0
        all_invalid = invalid.all(1)
        invalid = invalid.clone()
        invalid[all_invalid, 0] = False
        pooled, _ = self.pool(self.query.expand(len(roi), -1, -1), values, values,
                              key_padding_mask=invalid, need_weights=False)
        pooled = pooled[:, 0].masked_fill(all_invalid[:, None], 0)
        hidden = self.classifier(torch.cat((pooled, self.metadata(metadata)), -1))
        return {"event_logit": self.event(hidden).squeeze(-1),
                "quality_logit": self.quality(hidden).squeeze(-1)}


class JointBoundaryHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(channels + 1, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, local: torch.Tensor, relative_seconds: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((local, relative_seconds[:, None] / 900), -1))


class StatsFusionJointEvent(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        channels = int(model["hidden_dim"])
        self.encoder = StatsFusionStateModel(model)
        self.scales = nn.ModuleDict({str(seconds): CompletedScale(seconds // 3, channels,
                                                               float(model["dropout"]))
                                     for seconds in (3, 15, 60)})
        self.onset = nn.Linear(channels, 1)
        self.offset = nn.Linear(channels, 1)
        self.candidate_head = JointCandidateHead(channels, float(model["dropout"]))
        self.boundary_head = JointBoundaryHead(channels)
        self.contrastive_projection = nn.Linear(channels, 32)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        outputs = self.encoder.forward_features(batch)
        valid = torch.maximum(outputs["motion_valid_fraction"], outputs["ppg_valid_fraction"])
        outputs["sensor_valid"] = valid
        positions = batch["timeline_position"]
        proposals = {}
        for scale, module in self.scales.items():
            proposals[scale] = module(outputs["semantic_features"], valid, positions)
        outputs["proposal_scales"] = proposals
        outputs["onset_logit"] = self.onset(outputs["local_features"]).squeeze(-1)
        outputs["offset_logit"] = self.offset(outputs["local_features"]).squeeze(-1)
        return outputs


def quality_focal_loss(logits: torch.Tensor, target: torch.Tensor, beta: float = 2) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (
        target - logits.sigmoid()).abs().pow(beta)


def event_contrastive_loss(
    projected: torch.Tensor, event_ids: torch.Tensor, phases: torch.Tensor,
    *, temperature: float = 0.1, maximum_pairs: int = 128,
) -> tuple[torch.Tensor, int]:
    eligible = ((event_ids >= 0) | (event_ids == -2)) & (phases >= 0)
    indices = torch.nonzero(eligible).flatten()
    if len(indices) > maximum_pairs:
        indices = indices[torch.linspace(0, len(indices) - 1, maximum_pairs,
                                         device=indices.device).long()]
    values = F.normalize(projected[indices], dim=-1)
    identities, stages = event_ids[indices], phases[indices]
    losses = []
    for anchor in range(len(indices)):
        if int(identities[anchor]) < 0:
            continue
        positive = (identities == identities[anchor]) & (stages == stages[anchor])
        positive[anchor] = False
        negative = identities != identities[anchor]
        if bool(positive.any()) and bool(negative.any()):
            scores = values[anchor] @ values.T / temperature
            denominator = torch.logsumexp(scores[positive | negative], 0)
            losses.append(-(scores[positive] - denominator).mean())
    if not losses:
        return projected.sum() * 0, 0
    return torch.stack(losses).mean(), len(losses)


def loss_schedule(epoch: int) -> dict[str, float]:
    ramp = min(1.0, max(0.0, (epoch - 2) / 3))
    jitter = 0.5 if 3 <= epoch <= 5 else max(0.0, (8 - epoch) / 6) if epoch >= 6 else 0
    return {"event": ramp, "quality": ramp, "boundary": ramp,
            "contrastive": float(epoch >= 6), "truth_jitter": jitter}


def gaussian_distribution(times: torch.Tensor, endpoint_ms: int, sigma_seconds: float = 9):
    residual = (times.double() - endpoint_ms).float()
    weights = torch.exp(-0.5 * (residual / (sigma_seconds * 1000)) ** 2)
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)


def distribution_endpoint(logits: torch.Tensor, times: torch.Tensor) -> tuple[int, float]:
    probability = logits.float().softmax(0)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    entropy = float(entropy / max(math.log(len(times)), 1))
    origin = int(times[0])
    relative = times.double() - origin
    weights = probability.double()
    endpoint = origin + int((weights * relative).sum().div(weights.sum()).round())
    return endpoint, entropy
