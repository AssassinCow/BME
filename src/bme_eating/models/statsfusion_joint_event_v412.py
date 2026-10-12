from __future__ import annotations

import torch
from torch.nn import functional as F

from bme_eating.models.statsfusion_joint_event_v411 import (
    CandidateHeadV411,
    StatsFusionJointEventV411,
)


class CandidateHeadV412(CandidateHeadV411):
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

    def forward(self, roi, metadata):
        if roi.shape[1:] != (40, 258):
            raise ValueError("v412 requires forty aligned ROI tokens")
        values = self.encode_regions(roi)
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


class StatsFusionJointEventV412(StatsFusionJointEventV411):
    is_v412 = True

    def __init__(self, config):
        super().__init__(config)
        self.candidate_head = CandidateHeadV412(64, config["model"]["dropout"])
