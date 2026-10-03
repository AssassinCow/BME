from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _temporal_contrastive_loss(
    hidden: torch.Tensor,
    valid: torch.Tensor,
    timestamps_ms: torch.Tensor,
    subject_keys: object | None,
    session_ids: object | None,
    *,
    radius_steps: int,
    temperature: float,
    maximum_pairs: int,
) -> tuple[torch.Tensor, int, int]:
    if radius_steps <= 0:
        raise ValueError("Temporal contrastive radius_steps must be positive")
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("Temporal contrastive temperature must be positive and finite")
    if maximum_pairs <= 0:
        raise ValueError("Temporal contrastive maximum_pairs must be positive")
    if timestamps_ms.shape != valid.shape:
        raise ValueError("Temporal contrastive timestamps must align with valid positions")
    if not (
        isinstance(subject_keys, (list, tuple))
        and isinstance(session_ids, (list, tuple))
        and len(subject_keys) == len(session_ids) == hidden.shape[0]
    ):
        raise ValueError("Temporal contrastive pairs require subject and session identities")
    if radius_steps >= hidden.shape[1]:
        return hidden.sum() * 0.0, 0, 0
    hidden = F.normalize(hidden.float(), dim=-1, eps=1e-6)
    pair_mask = valid[:, :-radius_steps] & valid[:, radius_steps:]
    pair_positions = torch.nonzero(pair_mask, as_tuple=False)
    if len(pair_positions) == 0:
        return hidden.sum() * 0.0, 0, 0
    if len(pair_positions) > maximum_pairs:
        selected = torch.linspace(
            0,
            len(pair_positions) - 1,
            steps=maximum_pairs,
            device=pair_positions.device,
        ).round().long()
        pair_positions = pair_positions[selected]
    batch_index = pair_positions[:, 0]
    left_index = pair_positions[:, 1]
    right_index = left_index + radius_steps
    anchors = hidden[batch_index, left_index]
    positives = hidden[batch_index, right_index]
    logits = anchors @ positives.transpose(0, 1)
    logits = logits / float(temperature)
    identifiers = list(zip(map(str, subject_keys), map(str, session_ids)))
    selected_sessions = [
        identifiers[int(value)] for value in batch_index.detach().cpu().tolist()
    ]
    same_session = torch.tensor(
        [
            [left_session == right_session for right_session in selected_sessions]
            for left_session in selected_sessions
        ],
        dtype=torch.bool,
        device=hidden.device,
    )
    anchor_time = timestamps_ms[batch_index, left_index].to(hidden.device)
    positive_time = timestamps_ms[batch_index, right_index].to(hidden.device)
    pair_span = (positive_time - anchor_time).abs()
    nearby = (anchor_time[:, None] - positive_time[None, :]).abs() <= 2 * torch.maximum(
        pair_span[:, None], pair_span[None, :]
    )
    diagonal = torch.eye(len(pair_positions), dtype=torch.bool, device=hidden.device)
    forbidden = same_session & nearby & ~diagonal
    logits = logits.masked_fill(forbidden, torch.finfo(logits.dtype).min)
    labels = torch.arange(len(pair_positions), device=hidden.device)
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("Temporal contrastive loss became non-finite")
    negative_pairs = int((~forbidden).sum().item()) - len(pair_positions)
    return loss, len(pair_positions), negative_pairs


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
        smooth_beta: float,
        boundary_weight: float = 0.1,
        temporal_contrastive_weight: float = 0.0,
        temporal_contrastive_temperature: float = 0.1,
        temporal_contrastive_radius_steps: int = 1,
        temporal_contrastive_maximum_pairs: int = 512,
        proposal_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.smooth_weight = float(smooth_weight)
        self.smooth_beta = float(smooth_beta)
        if self.smooth_beta <= 0:
            raise ValueError("Smooth Huber beta must be positive")
        self.boundary_weight = float(boundary_weight)
        self.temporal_contrastive_weight = float(temporal_contrastive_weight)
        self.proposal_weight = float(proposal_weight)
        self.proposal_scale = 1.0
        if self.proposal_weight < 0:
            raise ValueError("Proposal loss weight must be non-negative")
        self.temporal_contrastive_temperature = float(temporal_contrastive_temperature)
        self.temporal_contrastive_radius_steps = int(temporal_contrastive_radius_steps)
        self.temporal_contrastive_maximum_pairs = int(temporal_contrastive_maximum_pairs)
        if self.temporal_contrastive_weight < 0:
            raise ValueError("Temporal contrastive weight must be non-negative")

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
        global_mask = batch.get("state_loss_mask", torch.ones_like(supervision)).to(
            supervision.dtype
        )
        state_weights = supervision * importance * global_mask
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
            supervision
            * importance
            * global_mask
            * batch.get("onset_loss_mask", torch.ones_like(supervision)),
        )
        offset = self._weighted_mean(
            offset_element,
            supervision
            * importance
            * global_mask
            * batch.get("offset_loss_mask", torch.ones_like(supervision)),
        )

        difference = torch.diff(output["state_logit"], dim=1)
        smooth_element = F.smooth_l1_loss(
            difference,
            torch.zeros_like(difference),
            beta=self.smooth_beta,
            reduction="none",
        )
        smooth_mask = supervision[:, 1:] * supervision[:, :-1]
        boundary_smooth_mask = batch.get("smooth_mask", torch.ones_like(supervision)).to(
            supervision.dtype
        )
        smooth_mask = smooth_mask * torch.minimum(
            boundary_smooth_mask[:, 1:], boundary_smooth_mask[:, :-1]
        )
        smooth_loss_mask = batch.get(
            "smooth_loss_mask", torch.ones_like(supervision)
        ).to(supervision.dtype)
        smooth_mask = smooth_mask * torch.minimum(
            smooth_loss_mask[:, 1:], smooth_loss_mask[:, :-1]
        )
        smooth_mask = smooth_mask * torch.minimum(global_mask[:, 1:], global_mask[:, :-1])
        smooth_mask = smooth_mask * torch.minimum(importance[:, 1:], importance[:, :-1])
        smooth = self._weighted_mean(smooth_element, smooth_mask)
        smooth_active = smooth_mask.sum()
        smooth_large_jump = self._weighted_mean(
            (difference.abs() > self.smooth_beta).to(difference.dtype), smooth_mask
        )
        hidden_output = output.get("state_hidden")
        if hidden_output is None:
            if self.temporal_contrastive_weight > 0:
                raise ValueError("Temporal contrastive loss requires state_hidden output")
            hidden_output = output["state_logit"]
        contrastive = hidden_output.sum() * 0.0
        contrastive_pairs = 0
        contrastive_negative_pairs = 0
        if self.temporal_contrastive_weight > 0:
            contrastive, contrastive_pairs, contrastive_negative_pairs = _temporal_contrastive_loss(
                hidden_output,
                (supervision * global_mask) > 0,
                batch["timestamp_ms"],
                batch.get("subject_key"),
                batch.get("session_id"),
                radius_steps=self.temporal_contrastive_radius_steps,
                temperature=self.temporal_contrastive_temperature,
                maximum_pairs=self.temporal_contrastive_maximum_pairs,
            )
        proposal = state * 0.0
        if self.proposal_weight:
            if "proposal_logit" not in output:
                raise ValueError("Enabled proposal loss requires proposal_logit")
            proposal_element = F.binary_cross_entropy_with_logits(
                output["proposal_logit"], batch["proposal_target"], reduction="none"
            )
            proposal = self._weighted_mean(
                proposal_element,
                supervision * importance * global_mask * batch["proposal_weight"],
            )
        total = (
            state
            + self.smooth_weight * smooth
            + self.boundary_weight * (onset + offset)
            + self.temporal_contrastive_weight * contrastive
            + self.proposal_weight * self.proposal_scale * proposal
        )
        if not torch.isfinite(total):
            raise FloatingPointError("StatsFusion state loss became non-finite")
        return total, {
            "state": state,
            "smooth": smooth,
            "onset": onset,
            "offset": offset,
            "smooth_active_count": smooth_active,
            "smooth_large_jump_fraction": smooth_large_jump,
            "contrastive": contrastive,
            "proposal": proposal,
            "contrastive_active_pairs": torch.as_tensor(
                float(contrastive_pairs), device=total.device
            ),
            "contrastive_negative_pairs": torch.as_tensor(
                float(contrastive_negative_pairs), device=total.device
            ),
        }
