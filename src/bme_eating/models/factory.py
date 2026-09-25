from __future__ import annotations

from typing import Any

from torch import nn

from bme_eating.models.dtp_sqf import DTPSQF
from bme_eating.models.hierarchical_state import HierarchicalStateModel


def build_state_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture", "dtp_sqf"))
    if architecture == "dtp_sqf":
        return DTPSQF(config)
    if architecture == "hierarchical_state":
        return HierarchicalStateModel(config)
    if architecture == "stats_fusion_state":
        from bme_eating.models.stats_fusion_state import StatsFusionStateModel

        return StatsFusionStateModel(config)
    raise ValueError(f"Unknown state model architecture: {architecture}")
