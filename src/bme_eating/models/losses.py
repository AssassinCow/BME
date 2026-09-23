from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_alpha: float,
    gamma: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability_target = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_target = positive_alpha * target + (1.0 - positive_alpha) * (1.0 - target)
    element = alpha_target * (1.0 - probability_target).pow(gamma) * binary_cross_entropy
    if mask is None:
        return element.mean()
    return (element * mask).sum() / mask.sum().clamp_min(1.0)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    if mask is not None:
        probability = probability * mask
        target = target * mask
    numerator = 2.0 * torch.sum(probability * target) + 1.0
    denominator = torch.sum(probability) + torch.sum(target) + 1.0
    return 1.0 - numerator / denominator


class DTPLoss(nn.Module):
    def __init__(
        self,
        positive_alpha: float,
        focal_gamma: float,
        dice_weight: float,
        boundary_weight: float,
        sqi_weight: float,
        boundary_positive_weight: float,
    ) -> None:
        super().__init__()
        self.positive_alpha = positive_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.sqi_weight = sqi_weight
        self.register_buffer(
            "boundary_positive_weight", torch.tensor(float(boundary_positive_weight))
        )

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state_target = batch["state_target"]
        focal = soft_focal_loss(
            output["state_logit"],
            state_target,
            self.positive_alpha,
            self.focal_gamma,
            batch.get("state_loss_mask"),
        )
        dice = soft_dice_loss(
            output["state_logit"], state_target, batch.get("state_loss_mask")
        )
        start_element = F.binary_cross_entropy_with_logits(
            output["start_logit"],
            batch["start_target"],
            pos_weight=self.boundary_positive_weight,
            reduction="none",
        )
        end_element = F.binary_cross_entropy_with_logits(
            output["end_logit"],
            batch["end_target"],
            pos_weight=self.boundary_positive_weight,
            reduction="none",
        )
        start_loss = (start_element * batch["start_loss_mask"]).sum() / batch[
            "start_loss_mask"
        ].sum().clamp_min(1.0)
        end_loss = (end_element * batch["end_loss_mask"]).sum() / batch[
            "end_loss_mask"
        ].sum().clamp_min(1.0)
        boundary = 0.5 * (start_loss + end_loss)
        sqi_mask = batch["ppg_valid"] > 0
        if sqi_mask.any():
            sqi = F.mse_loss(
                output["ppg_gate"][sqi_mask], batch["ppg_quality_target"][sqi_mask]
            )
        else:
            sqi = output["ppg_gate"].sum() * 0.0
        total = (
            focal
            + self.dice_weight * dice
            + self.boundary_weight * boundary
            + self.sqi_weight * sqi
        )
        return total, {"focal": focal, "dice": dice, "boundary": boundary, "sqi": sqi}


class HierarchicalStateLoss(DTPLoss):
    def __init__(self, *args, smooth_weight: float, smooth_tau: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.smooth_weight = float(smooth_weight)
        self.smooth_tau = float(smooth_tau)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, components = super().forward(output, batch)
        history = output["state_history_logit"]
        difference = torch.diff(F.logsigmoid(history), dim=1).square()
        per_sample = difference.clamp_max(self.smooth_tau).mean(dim=1)
        boundary_neighborhood_weight = 1.0 - torch.maximum(
            batch["start_target"], batch["end_target"]
        ).clamp(0.0, 1.0)
        boundary_neighborhood_weight = boundary_neighborhood_weight.clamp_min(0.1)
        state_mask = batch.get("state_loss_mask", torch.ones_like(per_sample))
        smooth_weight = boundary_neighborhood_weight * state_mask
        smooth = (per_sample * smooth_weight).sum() / smooth_weight.sum().clamp_min(1.0)
        total = total + self.smooth_weight * smooth
        return total, {**components, "smooth": smooth}
