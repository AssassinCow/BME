from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from bme_eating.models.stats_fusion_state import CausalTCN, StatsFusionStateModel
from bme_eating.protocol import capabilities


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
            pooled = pooled / weights.sum(1).clamp_min(
                torch.finfo(sequence.dtype).tiny if getattr(self, "fractional_mean", False) else 1)[:, None]
            low = self.head(self.temporal(pooled[None])).squeeze(0).squeeze(-1)
            mapping = torch.searchsorted(ends, torch.arange(len(sequence), device=ends.device),
                                         right=True) - 1
            eligible = mapping >= 0
            mapping = mapping.clamp_min(0)
            outputs.append(low[mapping] * eligible)
            masks.append(weights.mean(1)[mapping] * eligible)
        return torch.stack(outputs), torch.stack(masks)


class _CandidateInitialization(nn.Module):
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


class _BoundaryInitialization(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(channels + 1, 32), nn.SiLU(), nn.Linear(32, 2))


class _JointInitialization(nn.Module):
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
        self.candidate_head = _CandidateInitialization(channels, float(model["dropout"]))
        self.boundary_head = _BoundaryInitialization(channels)
        self.contrastive_projection = nn.Linear(channels, 32)


class _RegionInitialization(_CandidateInitialization):
    def __init__(self, channels=64, dropout=0.1):
        super().__init__(channels, dropout)
        self.regions = nn.Embedding(5, channels)
        self.register_buffer("region_ids", torch.tensor([0] * 4 + [1] * 16 + [2] * 4
                                                        + [3] * 8 + [4] * 8))


class JointBoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(132, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, local, relative_seconds, embedding, validity, onset, offset):
        if embedding.ndim == 1:
            embedding = embedding[None].expand(len(local), -1)
        scalars = torch.stack((relative_seconds / 900, validity, onset, offset), -1)
        return self.head(torch.cat((local, embedding, scalars), -1))


def fractional_mean(values, weights, dim=0):
    total = weights.sum(dim=dim)
    numerator = (values * weights.unsqueeze(-1)).sum(dim=dim)
    mean = numerator / total.clamp_min(torch.finfo(values.dtype).tiny).unsqueeze(-1)
    return mean.masked_fill((total <= 0).unsqueeze(-1), 0)


def indexed_roi(semantic, local, times, valid, start_ms, end_ms, decision_ms):
    if not start_ms < end_ms <= decision_ms:
        raise ValueError("Invalid causal ROI coordinates")
    parts = ((start_ms - 60_000, start_ms, 4), (start_ms, end_ms, 16),
             (end_ms, end_ms + 30_000, 4),
             (start_ms - 60_000, start_ms + 60_000, 8),
             (end_ms - 60_000, end_ms + 60_000, 8))
    combined = torch.cat((semantic, local), -1)
    tokens = []
    phase = int(times[0]) % 3000 if len(times) else 0
    for left, right, bins in parts:
        edges = torch.linspace(left, right, bins + 1, dtype=torch.float64, device=times.device)
        limits = torch.searchsorted(times, edges.clamp_max(decision_ms), right=True).tolist()
        for index in range(bins):
            first, stop = limits[index:index + 2]
            values, weights = combined[first:stop], valid[first:stop].to(combined.dtype)
            if len(values) and bool((weights > 0).any()):
                mean = fractional_mean(values, weights)
                maximum = values.masked_fill(weights[:, None] <= 0, -torch.inf).amax(0)
                fraction = weights.mean()
            else:
                mean = combined.new_zeros(combined.shape[-1])
                maximum, fraction = mean, combined.new_zeros(())
            left_edge, right_edge = float(edges[index]), float(edges[index + 1])
            expected = math.floor((right_edge - phase) / 3000) - math.floor((left_edge - phase) / 3000)
            coverage = min(1.0, (stop - first) / expected) if expected > 0 else 0.0
            relative = ((left_edge + right_edge) / 2 - start_ms) / max(end_ms - start_ms, 3000)
            tokens.append(torch.cat((mean, maximum, fraction[None],
                                     combined.new_tensor([coverage, relative]))))
    return torch.stack(tokens)


class JointCandidateHead(_RegionInitialization):
    def encode_regions(self, roi):
            valid = (roi[..., getattr(self, "validity_index", -2)] > 0).unsqueeze(-1)
            values = self.input(roi).masked_fill(~valid, 0)
            encoded = []
            first = 0
            for size in (4, 16, 4, 8, 8):
                mask = valid[:, first:first + size].transpose(1, 2)
                region = self.temporal.input_projection(values[:, first:first + size].transpose(1, 2))
                region = region.masked_fill(~mask, 0)
                for block in self.temporal.blocks:
                    update = block.depthwise(F.pad(region, (2 * block.dilation, 0))).masked_fill(~mask, 0)
                    update = block.pointwise(update).masked_fill(~mask, 0)
                    update = block.dropout(F.silu(block.normalization(update))).masked_fill(~mask, 0)
                    region = (region + update).masked_fill(~mask, 0)
                region = F.silu(self.temporal.output_norm(region)).masked_fill(~mask, 0)
                encoded.append(region.transpose(1, 2))
                first += size
            return (torch.cat(encoded, 1) + self.regions(self.region_ids)[None]).masked_fill(~valid, 0)

    def __init__(self, dropout):
        super().__init__(64, dropout)
        self.input = nn.Linear(259, 64)
        self.validity_index = 256
        self.comparison = nn.Sequential(nn.Linear(640, 64), nn.SiLU())

    def forward(self, roi, metadata):
        if roi.shape[1:] != (40, 259):
            raise ValueError("v414 requires forty 259-dimensional ROI tokens")
        values = self.encode_regions(roi)
        invalid = (roi[..., 256] <= 0) | (roi[..., 257] <= 0)
        values = values.masked_fill(invalid[..., None], 0)
        all_invalid = invalid.all(1)
        safe_invalid = invalid.clone()
        safe_invalid[all_invalid, 0] = False
        pooled, _ = self.pool(self.query.expand(len(roi), -1, -1), values, values,
                              key_padding_mask=safe_invalid, need_weights=False)
        pooled = pooled[:, 0].masked_fill(all_invalid[:, None], 0)
        summaries, present, first = [], [], 0
        for size in (4, 16, 4, 8, 8):
            weights = (~invalid[:, first:first + size]).to(values.dtype)
            summaries.append(fractional_mean(values[:, first:first + size], weights, dim=1))
            present.append(weights.sum(1) > 0)
            first += size
        differences = [(summaries[inner] - summaries[outer]).masked_fill(
            (~(present[inner] & present[outer]))[:, None], 0)
            for inner, outer in ((1, 0), (1, 2), (3, 0), (4, 2))]
        pooled = pooled + self.comparison(torch.cat([pooled, *summaries, *differences], -1)).masked_fill(
            all_invalid[:, None], 0)
        embedding = self.classifier(torch.cat((pooled, self.metadata(metadata)), -1))
        return {"event_logit": self.event(embedding).squeeze(-1),
                "quality_logit": self.quality(embedding).squeeze(-1),
                "candidate_embedding": embedding}


class LocalizationBlock(nn.Module):
    def __init__(self, dilation, dropout):
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv1d(64, 64, 3, dilation=dilation, groups=64)
        self.pointwise = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask):
        values = values.masked_fill(~mask[..., None], 0)
        update = self.depthwise(F.pad(values.transpose(1, 2), (2 * self.dilation, 0))).transpose(1, 2)
        update = update.masked_fill(~mask[..., None], 0)
        update = self.dropout(F.silu(self.norm(self.pointwise(update))))
        return (values + update).masked_fill(~mask[..., None], 0)


class StatsFusionJointEvent(_JointInitialization):
    def __init__(self, config):
        super().__init__(config)
        self.capabilities = capabilities(config)
        self.encoder.mask_invalid_temporal = True
        self.encoder.long_pool.fractional_mean = True
        for scale in self.scales.values():
            scale.fractional_mean = True
        dropout = config["model"]["dropout"]
        self.localization_fusion = nn.Sequential(nn.Linear(64 * 3 + 48 + 32 + 4, 64),
                                                  nn.LayerNorm(64), nn.SiLU())
        dilations = config["model"]["localization_dilations"]
        self.localization_blocks = nn.ModuleList(LocalizationBlock(value, dropout) for value in dilations)
        step_seconds = config["sequence"]["step_seconds"]
        temporal_history = sum((block.depthwise.kernel_size[0] - 1) * block.dilation
                               for block in self.localization_blocks) * step_seconds
        self.localization_dependency_seconds = {
            "motion_blocks": self.encoder.motion_block_seconds,
            "statistics_window": 15,
            "ppg_block": self.encoder.ppg_block_seconds,
            "ppg_maximum_hold": self.encoder.ppg_block_seconds - step_seconds,
            "temporal_history": temporal_history,
        }
        self.localization_receptive_field_seconds = max(
            self.localization_dependency_seconds["motion_blocks"],
            self.localization_dependency_seconds["statistics_window"],
            self.localization_dependency_seconds["ppg_block"]
            + self.localization_dependency_seconds["ppg_maximum_hold"]) + temporal_history
        self.candidate_head = JointCandidateHead(dropout)
        self.boundary_head = JointBoundaryHead()
        nn.init.constant_(self.onset.bias, math.log(0.01 / 0.99))
        nn.init.constant_(self.offset.bias, math.log(0.01 / 0.99))

    def forward(self, batch):
        outputs = self.encoder.forward_features(batch, expose_blocks=True)
        valid = torch.maximum(outputs["motion_valid_fraction"], outputs["ppg_valid_fraction"])
        outputs["sensor_valid"] = valid
        localization = self.localization_fusion(outputs.pop("block_features"))
        for block in self.localization_blocks:
            localization = block(localization, valid > 0)
        outputs["localization_features"] = localization
        outputs["onset_logit"] = self.onset(localization).squeeze(-1)
        outputs["offset_logit"] = self.offset(localization).squeeze(-1)
        outputs["proposal_scales"] = {scale: module(outputs["semantic_features"], valid,
                                                    batch["timeline_position"])
                                      for scale, module in self.scales.items()}
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


def build_joint_model(config):
    if config["model"]["architecture"] != "statsfusion_joint_event_v414":
        raise ValueError("Only statsfusion_joint_event_v414 is supported")
    return StatsFusionJointEvent(config)
