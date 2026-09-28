from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bme_eating.config import load_config
from bme_eating.data.stats_fusion_sequence import sequence_geometry_from_config
from bme_eating.stats_features import STATS_FEATURE_COLUMNS
from bme_eating.training.hierarchical_v4_trainer import (
    _decoder_configurations,
    _enforce_gradient_clipping_gate,
    _gradient_clipping_gate_passed,
    _selector_decoder_search_config,
)
from bme_eating.v4_protocol import validate_r3_config


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, item in value.items():
        if str(key).startswith("_"):
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(_flatten(item, path))
    return output


def _changed_paths(first: dict[str, Any], second: dict[str, Any]) -> set[str]:
    left = _flatten(first)
    right = _flatten(second)
    return {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}


def test_r3_config_is_strict_and_has_fixed_features() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "hierarchical_v4_statsfusion_r3.yaml")
    validate_r3_config(config)
    assert config["experiment"]["code_version"] == "v4.4.2"
    assert config["experiment"]["protocol_version"] == "statsfusion-r3.2"
    assert config["project"]["artifact_schema_version"] == "v4"
    assert config["project"]["strict_resume_identity"] is True
    assert tuple(config["model"]["stable_feature_columns"]) == STATS_FEATURE_COLUMNS
    assert config["hierarchical"]["maximum_event_latency_seconds"] == 60
    assert config["decoder"]["candidate_minimum_seconds"] == 3
    assert config["decoder"]["candidate_maximum_seconds"] == 14_400
    assert config["loss"]["smooth_beta"] == 0.5
    assert "postprocess_search" not in config


def test_r3_config_rejects_blocked_s4_ablation_name() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "hierarchical_v4_statsfusion_r3.yaml")
    config["experiment"]["ablation_id"] = "S4"
    with pytest.raises(ValueError, match="unsupported ablation_id"):
        validate_r3_config(config)


def test_r32_rejects_r31_configuration() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r31_direct_best.yaml")
    with pytest.raises(ValueError, match="statsfusion-r3.2"):
        validate_r3_config(config)


def test_r3_architecture_configs_change_registered_components() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    expected = {
        "hierarchical_v4_r3_s0.yaml": ("R3-S0", False, False, False, False, 0.0, 0.0),
        "hierarchical_v4_r3_s1.yaml": ("R3-S1", False, True, False, False, 0.0, 0.0),
        "hierarchical_v4_r3_s2.yaml": ("R3-S2", False, True, True, False, 0.0, 0.0),
        "hierarchical_v4_r3_d1.yaml": ("R3-D1", False, True, True, True, 0.2, 0.0),
        "hierarchical_v4_r3_d2a.yaml": ("R3-D2a", False, True, True, True, 0.2, 0.0),
        "hierarchical_v4_r3_d2b.yaml": ("R3-D2b", False, True, True, True, 0.2, 0.5),
        "hierarchical_v4_r3_d2c.yaml": ("R3-D2c", False, True, True, True, 0.2, 0.5),
        "hierarchical_v4_r3_p1.yaml": ("R3-P1", True, True, True, True, 0.2, 0.0),
        "hierarchical_v4_r3_m1.yaml": ("R3-M1", True, True, True, True, 0.2, 0.0),
    }
    for filename, values in expected.items():
        config = load_config(root / filename)
        validate_r3_config(config)
        ablation, ppg, statistics, long_context, separate, gyro_dropout, rotation = values
        assert config["experiment"]["ablation_id"] == ablation
        assert config["model"]["use_ppg"] is ppg
        assert config["model"]["use_statistics"] is statistics
        assert config["model"]["use_long_context"] is long_context
        assert config["model"]["separate_motion_branches"] is separate
        assert config["training"]["gyro_modality_dropout"] == gyro_dropout
        assert config["training"]["rotation_augmentation_probability"] == rotation
        assert config["decoder"]["use_semi_markov"] is (ablation == "R3-M1")


def test_r32_time_constrained_config_uses_single_holdout_and_pooled_logistic() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    validate_r3_config(config)
    assert config["experiment"]["ablation_id"] == "R3-DIRECT"
    assert config["experiment"]["variant"] == "time_constrained_outer_single_holdout_logistic"
    assert config["experiment"]["time_constrained_direct"] is True
    assert config["experiment"]["time_constrained_single_holdout"] is True
    assert config["model"]["use_statistics"] is True
    assert config["model"]["use_long_context"] is True
    assert config["model"]["separate_motion_branches"] is True
    assert config["model"]["use_invariant_motion_branch"] is True
    assert config["model"]["use_ppg"] is True
    assert config["training"]["gyro_modality_dropout"] == 0.0
    assert config["training"]["ppg_modality_dropout"] == 0.0
    assert config["training"]["rotation_augmentation_probability"] == 0.0
    assert config["training"]["batch_size"] == 8
    assert config["training"]["gradient_accumulation"] == 4
    assert config["training"]["clips_per_subject_per_epoch"] == 1000
    assert config["training"]["subject_balanced_sampling"] is True
    assert config["training"]["fixed_state_epochs"] == 5
    assert config["training"]["inference_batch_size"] == 8
    assert config["training"]["validation_every_epochs"] == 1
    assert config["training"]["selector_decoder_search"] == "fixed"
    assert config["training"]["learning_rate"] == 0.00015
    assert config["training"]["warmup_fraction"] == 0.10
    assert config["training"]["gradient_clip_norm"] == 5.0
    assert _enforce_gradient_clipping_gate(config) is False
    assert _gradient_clipping_gate_passed(config, 0.90) is True
    assert config["decoder"]["use_semi_markov"] is False
    assert config["decoder"]["use_transition_candidates"] is False
    assert config["hierarchical"]["state_crossfit_mode"] == "single_holdout"
    assert config["hierarchical"]["downstream_mode"] == "pooled_logistic"
    assert config["final_training"]["state_seeds"] == [2026]
    assert config["verifier"]["seeds"] == [2026]
    assert len(_decoder_configurations(config)) == 1
    assert len(_decoder_configurations(_selector_decoder_search_config(config))) == 1
    geometry = sequence_geometry_from_config(config)
    assert geometry.fused_short_receptive_field_steps == 155
    assert geometry.history_steps == 793
    assert geometry.total_steps == 1049


def test_r32_fast_config_uses_one_fixed_subject_holdout() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    validate_r3_config(config)
    assert config["experiment"]["variant"] == "time_constrained_outer_single_holdout_logistic"
    assert config["experiment"]["time_constrained_direct"] is True
    assert config["experiment"]["time_constrained_single_holdout"] is True
    assert config["training"]["fixed_state_epochs"] == 5
    assert config["training"]["single_holdout_fraction"] == 0.35
    assert config["training"]["learning_rate"] == 0.00015
    assert config["training"]["warmup_fraction"] == 0.10
    assert config["hierarchical"]["state_crossfit_mode"] == "single_holdout"
    assert config["hierarchical"]["downstream_mode"] == "pooled_logistic"
    assert config["final_training"]["state_seeds"] == [2026]
    assert len(_decoder_configurations(config)) == 1
    assert _decoder_configurations(config)[0] == {
        **config["decoder"],
        "high_threshold": 0.10,
        "low_threshold": 0.05,
        "ema_half_life_seconds": 12,
        "gap_merge_seconds": 60,
        "transition_threshold": 0.20,
        "semi_markov_duration_weight": 1.0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fixed_state_epochs", 0, "fixed_state_epochs"),
        ("single_holdout_fraction", 0.1, "Single-holdout fraction"),
        ("single_holdout_fraction", 0.5, "Single-holdout fraction"),
    ],
)
def test_r32_fast_config_rejects_invalid_training_scope(
    field: str, value: float, message: str
) -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    config["training"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_r3_config(config)


def test_r32_fast_config_requires_pooled_logistic_downstream() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    config["hierarchical"]["downstream_mode"] = "full"
    with pytest.raises(ValueError, match="pooled-logistic downstream"):
        validate_r3_config(config)


def test_r32_fast_variant_requires_explicit_protocol_flag() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    config["experiment"].pop("time_constrained_single_holdout")
    with pytest.raises(ValueError, match="variant and protocol flag"):
        validate_r3_config(config)


def test_r3_direct_config_cannot_silently_masquerade_as_a_registered_ablation() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    config["experiment"].pop("time_constrained_direct")
    with pytest.raises(ValueError, match="time_constrained_direct"):
        validate_r3_config(config)


def test_r3_direct_config_requires_fixed_selector_decoder_search() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    config["training"]["selector_decoder_search"] = "full"
    with pytest.raises(ValueError, match="fixed decoder search"):
        validate_r3_config(config)


def test_r3_registered_ablations_keep_gradient_clipping_gate() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r3_s2.yaml")
    assert _enforce_gradient_clipping_gate(config) is True
    assert _gradient_clipping_gate_passed(config, 0.20) is True
    assert _gradient_clipping_gate_passed(config, 0.21) is False


def test_r3_training_ablations_change_only_declared_factors() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    base = load_config(root / "hierarchical_v4_r3_s2.yaml")
    cases = {
        "hierarchical_v4_r3_ablation_A.yaml": {
            "experiment.variant",
            "training.validation_every_epochs",
        },
        "hierarchical_v4_r3_ablation_B.yaml": {
            "experiment.variant",
            "training.learning_rate",
            "training.warmup_fraction",
        },
        "hierarchical_v4_r3_ablation_C1.yaml": {
            "experiment.variant",
            "training.steps_per_epoch",
        },
        "hierarchical_v4_r3_ablation_C2.yaml": {
            "experiment.variant",
            "training.gradient_accumulation",
        },
    }
    for filename, allowed in cases.items():
        variant = load_config(root / filename)
        validate_r3_config(variant)
        assert _changed_paths(base, variant) == allowed
