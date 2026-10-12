from __future__ import annotations

import pandas as pd
import torch

from bme_eating.candidates import JointCandidate


def candidate(start=30_000, end=90_000, identity="one", parent="parent", family="family"):
    return JointCandidate("subject", "session", start, end, end + 30_000, "proposal", 3, .8,
        30_000, 90_000, family, identity, parent, "parent_hash", "model",
        {"v414": True, "parent_source": "proposal", "parent_scale": 3}, end + 30_000,
        score_formula="calibrated_event_probability_v414")


def feature_fixture(steps=50, gradient=False):
    torch.manual_seed(2026)
    values = torch.randn(steps, 201)
    values[:, 128:137] = .8
    return torch.arange(1, steps + 1) * 3000, values.requires_grad_(gradient)


def truth_fixture():
    return pd.DataFrame({"subject_key": ["subject"], "session_id": ["session"],
                         "event_id": ["truth"], "start_ms": [30_000], "end_ms": [90_000],
                         "hand_relation": ["unknown"]})
