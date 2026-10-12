from __future__ import annotations

import itertools

import torch
from torch import nn

from bme_eating.models.statsfusion_joint_event import (
    JointCandidateHead,
    StatsFusionJointEvent,
)


def indexed_roi(semantic, local, times, valid, start_ms, end_ms, decision_ms):
    if not start_ms < end_ms <= decision_ms:
        raise ValueError("Invalid causal ROI coordinates")
    parts = ((start_ms - 60_000, start_ms, 4), (start_ms, end_ms, 16),
             (end_ms, min(end_ms + 30_000, decision_ms), 4),
             (start_ms - 60_000, start_ms + 60_000, 8),
             (end_ms - 60_000, end_ms + 60_000, 8))
    combined = torch.cat((semantic, local), -1)
    tokens = []
    for left, right, bins in parts:
        edges = torch.linspace(left, right, bins + 1, dtype=torch.float64, device=times.device)
        limits = torch.searchsorted(times, edges.clamp_max(decision_ms), right=True).tolist()
        for index, (first, stop) in enumerate(itertools.pairwise(limits)):
            values, weights = combined[first:stop], valid[first:stop].to(combined.dtype)
            if len(values) and bool((weights > 0).any()):
                mean = (values * weights[:, None]).sum(0) / weights.sum().clamp_min(1)
                maximum = values.masked_fill(weights[:, None] <= 0, -torch.inf).amax(0)
                fraction = weights.mean()
            else:
                mean = combined.new_zeros(combined.shape[-1])
                maximum, fraction = mean, combined.new_zeros(())
            relative = ((float(edges[index]) + float(edges[index + 1])) / 2 - start_ms)
            tokens.append(torch.cat((mean, maximum, fraction[None],
                                     combined.new_tensor([relative / max(end_ms - start_ms, 3000)]))))
    return torch.stack(tokens)


class CandidateHeadV411(JointCandidateHead):
    def __init__(self, channels=64, dropout=0.1):
        super().__init__(channels, dropout)
        self.regions = nn.Embedding(5, channels)
        self.register_buffer("region_ids", torch.tensor([0] * 4 + [1] * 16 + [2] * 4
                                                        + [3] * 8 + [4] * 8))

    def forward(self, roi, metadata):
        if roi.shape[1:] != (40, 258):
            raise ValueError("v411 requires forty aligned ROI tokens")
        values = self.temporal(self.input(roi) + self.regions(self.region_ids)[None])
        invalid = roi[..., -2] <= 0
        all_invalid = invalid.all(1)
        invalid = invalid.clone()
        invalid[all_invalid, 0] = False
        pooled, _ = self.pool(self.query.expand(len(roi), -1, -1), values, values,
                              key_padding_mask=invalid, need_weights=False)
        pooled = pooled[:, 0].masked_fill(all_invalid[:, None], 0)
        embedding = self.classifier(torch.cat((pooled, self.metadata(metadata)), -1))
        return {"event_logit": self.event(embedding).squeeze(-1),
                "quality_logit": self.quality(embedding).squeeze(-1),
                "candidate_embedding": embedding}


class BoundaryHeadV411(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(132, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, local, relative_seconds, embedding, validity, onset, offset):
        if embedding.ndim == 1:
            embedding = embedding[None].expand(len(local), -1)
        scalars = torch.stack((relative_seconds / 900, validity, onset, offset), -1)
        return self.head(torch.cat((local, embedding, scalars), -1))


class StatsFusionJointEventV411(StatsFusionJointEvent):
    is_v411 = True

    def __init__(self, config):
        super().__init__(config)
        self.candidate_head = CandidateHeadV411(64, config["model"]["dropout"])
        self.boundary_head = BoundaryHeadV411()


def build_joint_model(config):
    architecture = config["model"]["architecture"]
    if architecture == "statsfusion_joint_event_v414":
        from bme_eating.models.statsfusion_joint_event_v414 import StatsFusionJointEventV414
        return StatsFusionJointEventV414(config)
    if architecture == "statsfusion_joint_event_v413":
        from bme_eating.models.statsfusion_joint_event_v413 import StatsFusionJointEventV413
        return StatsFusionJointEventV413(config)
    if architecture == "statsfusion_joint_event_v412":
        from bme_eating.models.statsfusion_joint_event_v412 import StatsFusionJointEventV412
        return StatsFusionJointEventV412(config)
    if architecture == "statsfusion_joint_event_v411":
        return StatsFusionJointEventV411(config)
    if architecture == "statsfusion_joint_event":
        return StatsFusionJointEvent(config)
    raise ValueError("Unregistered joint model architecture")
