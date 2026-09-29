from __future__ import annotations

from pathlib import Path

import pytest

from bme_eating.config import load_config
from bme_eating.data.stats_fusion_sequence import sequence_geometry_from_config
from bme_eating.stats_features import STATS_FEATURE_COLUMNS
from bme_eating.training.hierarchical_v4_trainer import (
    _decoder_configurations,
    _enforce_gradient_clipping_gate,
    _final_resume_comparison_identity,
    _gradient_clipping_gate_passed,
    _selector_decoder_search_config,
)
from bme_eating.v4_protocol import validate_r3_config, validate_serialized_verifier_seeds


def test_r3_config_is_strict_and_has_fixed_features() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "hierarchical_v4_statsfusion_r3.yaml")
    validate_r3_config(config)
    assert config["experiment"]["code_version"] == "v4.7.1"
    assert config["experiment"]["protocol_version"] == "statsfusion-r3.2"
    assert config["project"]["artifact_schema_version"] == "v4"
    assert config["project"]["strict_resume_identity"] is True
    assert config["project"]["enforce_git_identity_on_resume"] is False
    assert config["project"]["enforce_runtime_source_identity_on_resume"] is False
    assert tuple(config["model"]["stable_feature_columns"]) == STATS_FEATURE_COLUMNS
    assert config["hierarchical"]["maximum_event_latency_seconds"] == 60
    assert config["decoder"]["candidate_minimum_seconds"] == 3
    assert config["decoder"]["candidate_maximum_seconds"] == 14_400
    assert config["loss"]["smooth_beta"] == 0.5
    assert "postprocess_search" not in config


def test_final_resume_ignores_only_source_identity_metadata() -> None:
    config = {
        "project": {
            "enforce_git_identity_on_resume": False,
            "enforce_runtime_source_identity_on_resume": False,
        }
    }
    first = {
        "git": {"commit": "old"},
        "runtime_source_identity": {"sha256": "old"},
        "input_hashes": {"anchors": "same"},
        "parent_artifact_hashes": {
            "fold_0/run_manifest.json": "old-manifest",
            "fold_0/outer/window_logits.parquet": "same-logits",
        },
    }
    second = {
        "git": {"commit": "new"},
        "runtime_source_identity": {"sha256": "new"},
        "input_hashes": {"anchors": "same"},
        "parent_artifact_hashes": {
            "fold_0/run_manifest.json": "new-manifest",
            "fold_0/outer/window_logits.parquet": "same-logits",
        },
    }
    assert _final_resume_comparison_identity(first, config) == (
        _final_resume_comparison_identity(second, config)
    )
    second["parent_artifact_hashes"]["fold_0/outer/window_logits.parquet"] = "changed"
    assert _final_resume_comparison_identity(first, config) != (
        _final_resume_comparison_identity(second, config)
    )


def test_r3_config_rejects_blocked_s4_ablation_name() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "hierarchical_v4_statsfusion_r3.yaml")
    config["experiment"]["ablation_id"] = "S4"
    with pytest.raises(ValueError, match="unsupported ablation_id"):
        validate_r3_config(config)


def test_r32_rejects_r31_configuration() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_pooled_heads_early_select.yaml")
    config["experiment"]["protocol_version"] = "statsfusion-r3.1"
    with pytest.raises(ValueError, match="statsfusion-r3.2"):
        validate_r3_config(config)


def test_r3_architecture_configs_change_registered_components() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    expected = {
        "hierarchical_v4_r3_s2.yaml": ("R3-S2", False, True, True, False, 0.0, 0.0),
        "hierarchical_v4_r3_d1.yaml": ("R3-D1", False, True, True, True, 0.2, 0.0),
        "hierarchical_v4_r3_d2a.yaml": ("R3-D2a", False, True, True, True, 0.2, 0.0),
        "hierarchical_v4_r3_d2c.yaml": ("R3-D2c", False, True, True, True, 0.2, 0.5),
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
        assert config["decoder"]["use_semi_markov"] is False


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
    assert config["training"]["gyro_modality_dropout"] == 0.20
    assert config["training"]["ppg_modality_dropout"] == 0.20
    assert config["training"]["rotation_augmentation_probability"] == 0.50
    assert config["training"]["batch_size"] == 8
    assert config["training"]["gradient_accumulation"] == 4
    assert config["training"]["clips_per_subject_per_epoch"] == 1000
    assert config["training"]["subject_balanced_sampling"] is True
    assert "fixed_state_epochs" not in config["training"]
    assert config["training"]["max_epochs"] == 32
    assert config["training"]["early_stopping_min_epochs"] == 5
    assert config["training"]["early_stopping_patience_checks"] == 3
    assert config["training"]["inference_batch_size"] == 8
    assert config["training"]["validation_every_epochs"] == 1
    assert config["training"]["selector_decoder_search"] == "fixed"
    assert config["training"]["learning_rate"] == 0.00015
    assert config["training"]["warmup_fraction"] == 0.10
    assert config["training"]["gradient_clip_norm"] == 5.0
    assert _enforce_gradient_clipping_gate(config) is True
    assert _gradient_clipping_gate_passed(config, 0.20) is True
    assert _gradient_clipping_gate_passed(config, 0.21) is False
    assert config["training"]["gradient_clipping_abort_fraction"] == 1.0
    assert config["decoder"]["use_semi_markov"] is False
    assert config["decoder"]["use_transition_candidates"] is False
    assert config["hierarchical"]["state_crossfit_mode"] == "single_holdout"
    assert config["hierarchical"]["downstream_mode"] == "pooled_logistic"
    assert config["final_training"]["state_seeds"] == [2026]
    assert config["verifier"]["seeds"] == [2026]
    assert config["promotion_gate"]["minimum_candidate_recall"] == 0.70
    assert config["promotion_gate"]["target_candidate_recall"] == 0.75
    assert len(_decoder_configurations(config)) == 1
    assert len(_decoder_configurations(_selector_decoder_search_config(config))) == 1
    geometry = sequence_geometry_from_config(config)
    assert geometry.fused_short_receptive_field_steps == 155
    assert geometry.history_steps == 793
    assert geometry.total_steps == 1049


def test_r32_pooled_heads_config_keeps_fast_state_and_single_seed_heads() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_pooled_heads.yaml")
    validate_r3_config(config)
    assert config["experiment"]["variant"] == (
        "time_constrained_outer_single_holdout_pooled_heads"
    )
    assert config["hierarchical"]["state_crossfit_mode"] == "single_holdout"
    assert config["hierarchical"]["downstream_mode"] == "pooled_heads"
    assert config["hierarchical"]["pooled_head_protocol"] == (
        "outer_fold_crossfit_joint_tuning_v1"
    )
    assert config["final_training"]["state_seeds"] == [2026]
    assert config["verifier"]["seeds"] == [2026]
    assert config["boundary"]["seeds"] == [2026]
    assert config["verifier"]["minimum_epochs"] == 5
    assert config["boundary"]["minimum_epochs"] == 5
    assert config["verifier"]["patience"] == 8
    assert config["boundary"]["patience"] == 8


def test_r32_early_select_uses_complete_robust_window() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_pooled_heads_early_select.yaml")
    validate_r3_config(config)
    assert config["training"]["max_epochs"] == 32
    assert config["training"]["early_stopping_min_epochs"] == 3
    assert config["training"]["early_stopping_patience_checks"] == 3
    assert config["training"]["checkpoint_selection_min_epoch"] == 3
    assert config["hierarchical"]["downstream_mode"] == "pooled_heads"


def test_verifier_seed_profiles_allow_single_or_three_seed_deployment() -> None:
    assert validate_serialized_verifier_seeds([2026]) == [2026]
    assert validate_serialized_verifier_seeds([2026, 2027, 2028]) == [2026, 2027, 2028]
    with pytest.raises(RuntimeError, match="verifier seeds"):
        validate_serialized_verifier_seeds([2027])


def test_r32_fast_config_uses_one_nested_selector_holdout() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r32_fast_logistic.yaml")
    validate_r3_config(config)
    assert config["experiment"]["variant"] == "time_constrained_outer_single_holdout_logistic"
    assert config["experiment"]["time_constrained_direct"] is True
    assert config["experiment"]["time_constrained_single_holdout"] is True
    assert "fixed_state_epochs" not in config["training"]
    assert config["training"]["max_epochs"] == 32
    assert config["training"]["early_stopping_min_epochs"] == 5
    assert config["training"]["early_stopping_patience_checks"] == 3
    assert config["training"]["single_holdout_fraction"] == 0.35
    assert config["training"]["learning_rate"] == 0.00015
    assert config["training"]["warmup_fraction"] == 0.10
    assert config["hierarchical"]["state_crossfit_mode"] == "single_holdout"
    assert config["hierarchical"]["downstream_mode"] == "pooled_logistic"
    assert config["final_training"]["state_seeds"] == [2026]
    assert config["promotion_gate"]["minimum_candidate_recall"] == 0.70
    assert config["promotion_gate"]["target_candidate_recall"] == 0.75
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
        ("max_epochs", 31, "max_epochs"),
        ("early_stopping_min_epochs", 0, "early-stopping minimum"),
        ("checkpoint_selection_min_epoch", 2, "robust selector window"),
        ("early_stopping_patience_checks", 0, "early-stopping patience"),
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
    with pytest.raises(ValueError, match="downstream mode"):
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


def test_r3_parent_config_keeps_gradient_clipping_gate() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(root / "hierarchical_v4_r3_s2.yaml")
    assert _enforce_gradient_clipping_gate(config) is True
    assert _gradient_clipping_gate_passed(config, 0.20) is True
    assert _gradient_clipping_gate_passed(config, 0.21) is False


def test_repository_keeps_only_current_config_chain() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    expected = {
        "base.yaml",
        "hierarchical_v4_statsfusion_r3.yaml",
        "hierarchical_v4_r3_s2.yaml",
        "hierarchical_v4_r3_d1.yaml",
        "hierarchical_v4_r3_d2a.yaml",
        "hierarchical_v4_r3_d2c.yaml",
        "hierarchical_v4_r32_fast_logistic.yaml",
        "hierarchical_v4_r32_pooled_heads.yaml",
        "hierarchical_v4_r32_pooled_heads_early_select.yaml",
    }
    assert {path.name for path in root.glob("*.yaml")} == expected
