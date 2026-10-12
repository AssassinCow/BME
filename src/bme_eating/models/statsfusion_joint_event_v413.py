from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from bme_eating.joint_capabilities import capabilities
from bme_eating.models.statsfusion_joint_event import StatsFusionJointEvent
from bme_eating.models.statsfusion_joint_event_v411 import BoundaryHeadV411
from bme_eating.models.statsfusion_joint_event_v412 import CandidateHeadV412


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


class CandidateHeadV413(CandidateHeadV412):
    def __init__(self, dropout):
        super().__init__(64, dropout)
        self.input = nn.Linear(259, 64)
        self.validity_index = 256
        self.comparison = nn.Sequential(nn.Linear(640, 64), nn.SiLU())

    def forward(self, roi, metadata):
        if roi.shape[1:] != (40, 259):
            raise ValueError("v413 requires forty 259-dimensional ROI tokens")
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


class StatsFusionJointEventV413(StatsFusionJointEvent):
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
        self.candidate_head = CandidateHeadV413(dropout)
        self.boundary_head = BoundaryHeadV411()
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
