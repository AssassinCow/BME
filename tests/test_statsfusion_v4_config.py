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
    assert config["training"]["batch_size"] == 16
    assert config["training"]["gradient_accumulation"] == 2
    assert config["training"]["steps_per_epoch"] == 1250
    assert config["training"]["inference_batch_size"] == 16
    assert config["training"]["num_workers"] == 12
    assert config["training"]["inference_num_workers"] == 12
    assert config["training"]["selector_fraction"] == 0.35
    assert config["training"]["selector_rolling_epochs"] == 3
    assert config["training"]["validation_every_epochs"] == 2
    assert config["training"]["early_stopping_min_epochs"] == 12
    assert config["training"]["early_stopping_patience_checks"] == 3
    assert config["training"]["early_stopping_min_delta"] == 0.003
    assert config["training"]["learning_rate"] == 0.0003


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


def test_training_monitoring_ablation_configs_are_reproducible() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    base = load_config(root / "hierarchical_v4_s2.yaml")
    validation = load_config(root / "hierarchical_v4_s2_ablation_A.yaml")
    lower_lr = load_config(root / "hierarchical_v4_s2_ablation_B.yaml")
    fewer_updates = load_config(root / "hierarchical_v4_s2_ablation_C.yaml")

    assert validation["experiment"]["variant"] == "validation_every_epoch"
    assert validation["training"]["validation_every_epochs"] == 1
    assert validation["training"]["learning_rate"] == base["training"]["learning_rate"]

    assert lower_lr["experiment"]["variant"] == "lower_learning_rate"
    assert lower_lr["training"]["learning_rate"] == 0.00015
    assert lower_lr["training"]["warmup_fraction"] == 0.10
    assert lower_lr["training"]["validation_every_epochs"] == base["training"]["validation_every_epochs"]

    assert fewer_updates["experiment"]["variant"] == "reduced_optimizer_updates"
    assert fewer_updates["training"]["steps_per_epoch"] == 504
    assert fewer_updates["training"]["gradient_accumulation"] == base["training"]["gradient_accumulation"]


def test_training_monitoring_abc_combined_config() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_s2_ablation_ABC.yaml")

    assert config["experiment"]["ablation_id"] == "S2"
    assert config["experiment"]["variant"] == "validation_every_epoch_lower_lr_reduced_updates"
    assert config["training"]["validation_every_epochs"] == 1
    assert config["training"]["learning_rate"] == 0.00015
    assert config["training"]["warmup_fraction"] == 0.10
    assert config["training"]["steps_per_epoch"] == 252
    assert config["training"]["gradient_accumulation"] == 1
