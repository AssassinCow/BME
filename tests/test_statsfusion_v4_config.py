from __future__ import annotations

from pathlib import Path

from bme_eating.config import load_config
from bme_eating.stats_features import STATS_FEATURE_COLUMNS


def test_v4_config_is_strict_and_has_fixed_features() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "hierarchical_v4_statsfusion.yaml")
    assert config["project"]["artifact_schema_version"] == "v4"
    assert config["project"]["strict_resume_identity"] is True
    assert tuple(config["model"]["stable_feature_columns"]) == STATS_FEATURE_COLUMNS
    assert config["hierarchical"]["maximum_event_latency_seconds"] == 60
    assert config["decoder"]["fixed_lag_seconds"] == 60


def test_v4_ablation_configs_change_only_registered_components() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    variants = {
        "S0": (True, False, False, False, False),
        "S1": (True, False, True, False, False),
        "S2": (True, False, True, True, False),
        "S3": (True, True, True, True, False),
        "S4": (True, True, True, True, True),
        "PPG_ONLY": (False, True, False, False, False),
    }
    paths = {
        "S0": "hierarchical_v4_s0.yaml",
        "S1": "hierarchical_v4_s1.yaml",
        "S2": "hierarchical_v4_s2.yaml",
        "S3": "hierarchical_v4_s3.yaml",
        "S4": "hierarchical_v4_statsfusion.yaml",
        "PPG_ONLY": "hierarchical_v4_ppg_only.yaml",
    }
    for name, expected in variants.items():
        config = load_config(root / paths[name])
        actual = (
            config["model"]["use_motion"],
            config["model"]["use_ppg"],
            config["model"]["use_statistics"],
            config["model"]["use_long_context"],
            config["decoder"]["use_semi_markov"],
        )
        assert config["experiment"]["ablation_id"] == name
        assert actual == expected
