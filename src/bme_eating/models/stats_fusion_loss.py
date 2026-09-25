from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def one_sided_transition_targets(
    timestamps_ms: torch.Tensor,
    starts_ms: torch.Tensor,
    ends_ms: torch.Tensor,
    onset_seconds: float = 30.0,
    offset_seconds: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    timestamps = timestamps_ms.to(torch.float64).reshape(-1, 1)
    starts = starts_ms.to(torch.float64).reshape(1, -1)
    ends = ends_ms.to(torch.float64).reshape(1, -1)
    onset_delta = (timestamps - starts) / 1000.0
    offset_delta = (timestamps - ends) / 1000.0
    onset = torch.where(
        (onset_delta >= 0.0) & (onset_delta <= onset_seconds),
        1.0 - onset_delta / onset_seconds,
        0.0,
    )
    offset = torch.where(
        (offset_delta >= 0.0) & (offset_delta <= offset_seconds),
        1.0 - offset_delta / offset_seconds,
        0.0,
    )
    return onset.amax(dim=1).to(torch.float32), offset.amax(dim=1).to(torch.float32)


class StatsFusionStateLoss(nn.Module):
    def __init__(
        self,
        *,
        smooth_weight: float,
        smooth_tau: float,
        boundary_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.smooth_weight = float(smooth_weight)
        self.smooth_tau = float(smooth_tau)
        self.boundary_weight = float(boundary_weight)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        supervision = batch["supervision_mask"].to(output["state_logit"].dtype)
        importance = batch.get("importance_weight", torch.ones_like(supervision))
        if importance.ndim == 1:
            importance = importance.unsqueeze(-1)
        state_weights = supervision * importance * batch.get(
            "state_loss_mask", torch.ones_like(supervision)
        )
        state_element = F.binary_cross_entropy_with_logits(
            output["state_logit"], batch["state_target"], reduction="none"
        )
        state = self._weighted_mean(state_element, state_weights)

        onset_element = F.binary_cross_entropy_with_logits(
            output["onset_logit"], batch["onset_target"], reduction="none"
        )
        offset_element = F.binary_cross_entropy_with_logits(
            output["offset_logit"], batch["offset_target"], reduction="none"
        )
        onset = self._weighted_mean(
            onset_element,
            supervision * batch.get("onset_loss_mask", torch.ones_like(supervision)),
        )
        offset = self._weighted_mean(
            offset_element,
            supervision * batch.get("offset_loss_mask", torch.ones_like(supervision)),
        )

        difference = torch.diff(output["state_logit"], dim=1)
        smooth_element = difference.square().clamp_max(self.smooth_tau)
        smooth_mask = supervision[:, 1:] * supervision[:, :-1]
        smooth_mask = smooth_mask * batch.get(
            "smooth_mask", torch.ones_like(supervision)
        )[:, 1:]
        smooth = self._weighted_mean(smooth_element, smooth_mask)
        total = state + self.smooth_weight * smooth + self.boundary_weight * (onset + offset)
        return total, {
            "state": state,
            "smooth": smooth,
            "onset": onset,
            "offset": offset,
        }
