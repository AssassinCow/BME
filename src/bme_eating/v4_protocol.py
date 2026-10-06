from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CODE_VERSION = "v4.8"
V49_CODE_VERSION = "v4.9"
LEGACY_CODE_VERSION = "v4.7.1"
PROTOCOL_VERSION = "statsfusion-r3.2"
POOLED_HEAD_PROTOCOL = "outer_fold_crossfit_joint_tuning_v1"
POOLED_HEAD_TRAINING_PROTOCOL = "fully_excluded_nested_state_oof_v1"
INPUT_SNAPSHOT_FILENAME = "input_snapshot_r3_2.json"
BLOCKED_PREDECESSORS = (
    "statsfusion-r0-blocked",
    "statsfusion-r1-blocked",
    "statsfusion-r2-blocked",
    "statsfusion-r3-blocked",
    "statsfusion-r3.1-blocked",
)
TARGET_SEMANTICS = "soft_interval_occupancy"
CALIBRATION_PROTOCOL = "soft_platt_v1"
DECODER_PROTOCOL = "coherent_fixed_lag_segment_v1"
RAW_INPUT_SCHEMA = "statsfusion-raw-v2"
IGNORE_PROTOCOL = "prediction-overlap-fraction-0.5-causal-support-v2"
OBSERVATION_GAP_PROTOCOL = "active_sensor_valid_runs_v1"
RUNTIME_SOURCE_BINDING = "exact_runtime_sha_v1"
FULL_STATE_SEEDS = (2026, 2027, 2028)
DIRECT_STATE_SEEDS = (2026,)
RUNTIME_SOURCE_FILES = (
    "__init__.py",
    "calibration_v4.py",
    "hierarchical_v4_pipeline.py",
    "metrics.py",
    "proposals_v4.py",
    "stats_features.py",
    "structured_decoder.py",
    "timeline.py",
    "types.py",
    "v4_protocol.py",
    "data/__init__.py",
    "data/deep_dataset.py",
    "data/session.py",
    "data/stats_fusion_preprocess.py",
    "data/stats_fusion_sequence.py",
    "features/__init__.py",
    "features/baseline.py",
    "features/signal.py",
    "models/__init__.py",
    "models/dtp_sqf.py",
    "models/endpoint_refiner.py",
    "models/event_verifier_v4.py",
    "models/stats_fusion_loss.py",
    "models/factory.py",
    "models/hierarchical_state.py",
    "models/stats_fusion_state.py",
)
R3_ABLATION_IDS = frozenset(
    {
        "R3-S2",
        "R3-D1",
        "R3-D2a",
        "R3-D2c",
        "R3-DIRECT",
    }
)

R3_ROOT_CONFIG_KEYS = frozenset(
    {
        "project",
        "data",
        "experiment",
        "model",
        "sequence",
        "training",
        "loss",
        "calibration",
        "decoder",
        "decoder_search",
        "verifier",
        "boundary",
        "hierarchical",
        "promotion_gate",
        "feature_provenance",
        "final_training",
        "v49",
    }
)


def configured_state_seeds(config: dict[str, Any]) -> list[int]:
    actual = [int(value) for value in config.get("final_training", {}).get("state_seeds", [])]
    expected = (
        list(DIRECT_STATE_SEEDS)
        if str(config.get("experiment", {}).get("ablation_id", "")) == "R3-DIRECT"
        else list(FULL_STATE_SEEDS)
    )
    if actual != expected:
        raise ValueError(
            f"{config.get('experiment', {}).get('ablation_id', 'StatsFusion-r3')} "
            f"requires state seeds {expected}"
        )
    return actual


def validate_serialized_state_seeds(values: Any) -> list[int]:
    seeds = [int(value) for value in values]
    if tuple(seeds) not in {DIRECT_STATE_SEEDS, FULL_STATE_SEEDS}:
        raise RuntimeError(
            "V4 state seeds must match a registered single-seed or three-seed profile"
        )
    return seeds


def validate_serialized_verifier_seeds(values: Any) -> list[int]:
    seeds = [int(value) for value in values]
    if tuple(seeds) not in {DIRECT_STATE_SEEDS, FULL_STATE_SEEDS}:
        raise RuntimeError(
            "V4 verifier seeds must match a registered single-seed or three-seed profile"
        )
    return seeds


def runtime_source_identity(package_root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in RUNTIME_SOURCE_FILES:
        path = Path(package_root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"StatsFusion runtime source is missing: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[relative] = digest.hexdigest()
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"files": files, "sha256": hashlib.sha256(payload).hexdigest()}


def validate_r3_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    protocol = experiment.get("protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise ValueError(f"Formal {CODE_VERSION} runs require protocol_version: {PROTOCOL_VERSION}")
    if experiment.get("code_version") not in {CODE_VERSION, V49_CODE_VERSION, LEGACY_CODE_VERSION}:
        raise ValueError(f"Formal {PROTOCOL_VERSION} runs require code_version: {CODE_VERSION}")
    candidate_protocol = config.get("decoder", {}).get("candidate_protocol")
    if candidate_protocol in {"v4.8", "v4.9"}:
        expected_code = V49_CODE_VERSION if candidate_protocol == "v4.9" else CODE_VERSION
        if experiment.get("code_version") != expected_code:
            raise ValueError(f"{candidate_protocol} candidates require {expected_code} code_version")
        expected_variant = (
            "time_constrained_outer_single_holdout_deep_only_v49"
            if candidate_protocol == "v4.9"
            else "time_constrained_outer_single_holdout_deep_only_v48"
        )
        if experiment.get("variant") != expected_variant:
            raise ValueError(f"{candidate_protocol} candidates require the registered deep-only variant")
        if not bool(config.get("verifier", {}).get("include_v48_source_flags", False)):
            raise ValueError(f"{candidate_protocol} verifier requires all candidate-source flags")
        if int(config.get("decoder", {}).get("maximum_variants_per_event", 0)) != 17:
            raise ValueError("v4.8 requires the 17-way symmetric jitter set")
        if float(config["decoder"].get("low_threshold", -1)) != 0.04:
            raise ValueError("v4.8 requires low_threshold: 0.04")
        if float(config["decoder"].get("high_threshold", -1)) not in {0.06, 0.08}:
            raise ValueError("v4.8 high threshold must be 0.06 or 0.08")
        if bool(config.get("verifier", {}).get("use_raw_imu_branch", False)) and (
            config.get("hierarchical", {}).get("downstream_mode") != "pooled_deep_only"
        ):
            raise ValueError("v4.8 raw IMU verifier requires pooled Deep mode")
        if bool(config.get("model", {}).get("use_proposal_head", False)) != bool(
            config.get("decoder", {}).get("use_proposal_head", False)
        ):
            raise ValueError("v4.8 proposal head model and decoder flags must agree")
        if bool(config.get("model", {}).get("use_proposal_head", False)) != (
            float(config.get("loss", {}).get("proposal_weight", 0.0)) == 0.1
        ):
            raise ValueError("v4.8 enabled proposal head requires loss weight 0.1")
    ablation_id = str(experiment.get("ablation_id", ""))
    if ablation_id not in R3_ABLATION_IDS:
        raise ValueError(f"StatsFusion-r3 uses an unsupported ablation_id: {ablation_id!r}")
    if ablation_id == "R3-DIRECT":
        pooled_variants = {
            "time_constrained_outer_single_holdout_pooled_heads",
            "time_constrained_outer_single_holdout_pooled_heads_transition",
            "time_constrained_outer_single_holdout_deep_only_transition",
            "time_constrained_outer_single_holdout_deep_only_v48",
            "time_constrained_outer_single_holdout_deep_only_v49",
        }
        direct_variants = {
            "time_constrained_outer_single_holdout_logistic",
            *pooled_variants,
        }
        if experiment.get("variant") not in direct_variants:
            raise ValueError("R3-DIRECT requires the registered time-constrained variant")
        if not bool(experiment.get("time_constrained_direct", False)):
            raise ValueError("R3-DIRECT must explicitly declare time_constrained_direct: true")
        model = config.get("model", {})
        required_model_flags = (
            "use_ppg",
            "use_statistics",
            "use_long_context",
            "separate_motion_branches",
            "use_invariant_motion_branch",
        )
        if not all(bool(model.get(key, False)) for key in required_model_flags):
            raise ValueError("R3-DIRECT requires the registered D2c+PPG state architecture")
        decoder = config.get("decoder", {})
        if bool(decoder.get("use_semi_markov", False)):
            raise ValueError("R3-DIRECT disables Semi-Markov for the time-constrained route")
        transition_variant = experiment.get("variant") in {
            "time_constrained_outer_single_holdout_pooled_heads_transition",
            "time_constrained_outer_single_holdout_deep_only_transition",
        }
        if bool(decoder.get("use_transition_candidates", False)) != transition_variant:
            raise ValueError(
                "R3-DIRECT transition candidates must match the registered transition variant"
            )
        if config.get("training", {}).get("selector_decoder_search") != "fixed":
            raise ValueError("R3-DIRECT requires fixed decoder search during state selection")
        single_holdout = bool(experiment.get("time_constrained_single_holdout", False))
        variant_is_single_holdout = experiment.get("variant") in direct_variants
        if single_holdout != variant_is_single_holdout:
            raise ValueError("Single-holdout variant and protocol flag must agree")
        state_crossfit_mode = str(config.get("hierarchical", {}).get("state_crossfit_mode", "full"))
        downstream_mode = str(config.get("hierarchical", {}).get("downstream_mode", "full"))
        if single_holdout:
            if state_crossfit_mode != "single_holdout":
                raise ValueError("Single-holdout direct training requires state_crossfit_mode")
            if experiment.get("variant") in {
                "time_constrained_outer_single_holdout_deep_only_transition",
                "time_constrained_outer_single_holdout_deep_only_v48",
                "time_constrained_outer_single_holdout_deep_only_v49",
            }:
                expected_downstream = "pooled_deep_only"
            else:
                expected_downstream = (
                    "pooled_heads"
                    if experiment.get("variant") in pooled_variants
                    else "pooled_logistic"
                )
            if downstream_mode != expected_downstream:
                raise ValueError(
                    "Single-holdout direct training downstream mode does not match its variant"
                )
            training = config.get("training", {})
            if "fixed_state_epochs" in training:
                raise ValueError("Single-holdout training must select epochs with early stopping")
            maximum_epochs = int(training.get("max_epochs", 0))
            minimum_epochs = int(training.get("early_stopping_min_epochs", 0))
            checkpoint_selection_minimum_epoch = int(
                training.get("checkpoint_selection_min_epoch", minimum_epochs)
            )
            patience_checks = int(training.get("early_stopping_patience_checks", 0))
            holdout_fraction = float(config.get("training", {}).get("single_holdout_fraction", 0.0))
            if maximum_epochs != 32:
                raise ValueError("Single-holdout max_epochs must be 32")
            if not 1 <= minimum_epochs < maximum_epochs:
                raise ValueError("Single-holdout early-stopping minimum must be in [1, 31]")
            if not 1 <= checkpoint_selection_minimum_epoch <= minimum_epochs:
                raise ValueError(
                    "Single-holdout checkpoint-selection minimum must be positive and not "
                    "exceed the early-stopping minimum"
                )
            if not 1 <= patience_checks <= 8:
                raise ValueError("Single-holdout early-stopping patience must be in [1, 8]")
            if not 0.2 <= holdout_fraction < 0.5:
                raise ValueError("Single-holdout fraction must be in [0.2, 0.5)")
            if not bool(training.get("subject_balanced_sampling", False)):
                raise ValueError("R3-DIRECT requires subject-balanced sampling")
            if int(training.get("clips_per_subject_per_epoch", 0)) <= 0:
                raise ValueError("R3-DIRECT requires positive clips_per_subject_per_epoch")
            if float(training.get("gyro_modality_dropout", 0.0)) != 0.20:
                raise ValueError("R3-DIRECT requires gyro_modality_dropout: 0.20")
            if float(training.get("ppg_modality_dropout", 0.0)) != 0.20:
                raise ValueError("R3-DIRECT requires ppg_modality_dropout: 0.20")
            if float(training.get("rotation_augmentation_probability", 0.0)) != 0.50:
                raise ValueError("R3-DIRECT requires rotation_augmentation_probability: 0.50")
            promotion_fraction = float(
                training.get("gradient_clipping_promotion_fraction", 0.20)
            )
            abort_fraction = float(training.get("gradient_clipping_abort_fraction", 0.50))
            if promotion_fraction != 0.20:
                raise ValueError("R3-DIRECT clipping promotion fraction must be 0.20")
            if not 0.50 <= abort_fraction <= 1.0:
                raise ValueError("R3-DIRECT clipping abort fraction must be in [0.50, 1.0]")
        elif state_crossfit_mode != "full" or downstream_mode != "full":
            raise ValueError("Registered full direct training requires full crossfit/downstream")
    if tuple(experiment.get("blocked_protocols", ())) != BLOCKED_PREDECESSORS:
        raise ValueError("StatsFusion-r3 must declare all blocked predecessor protocols")
    public_keys = {key for key in config if not key.startswith("_")}
    unknown = public_keys - R3_ROOT_CONFIG_KEYS
    if unknown:
        raise ValueError(f"StatsFusion-r3 config contains unsupported root keys: {sorted(unknown)}")
    project = config.get("project", {})
    if project.get("input_artifact_schema_version") != "v2":
        raise ValueError("StatsFusion-r3 requires immutable v2 input artifacts")
    if project.get("artifact_schema_version") != "v4":
        raise ValueError("StatsFusion-r3 outputs must use artifact schema v4")
    if not bool(project.get("strict_resume_identity", False)):
        raise ValueError("StatsFusion-r3 requires strict_resume_identity: true")
    for key in (
        "enforce_git_identity_on_resume",
        "enforce_runtime_source_identity_on_resume",
    ):
        if key in project and not isinstance(project[key], bool):
            raise ValueError(f"project.{key} must be a boolean")
    decoder = config.get("decoder", {})
    minimum = float(decoder.get("candidate_minimum_seconds", 0))
    maximum = float(decoder.get("candidate_maximum_seconds", 0))
    if minimum != 3.0 or maximum != 14_400.0:
        raise ValueError("StatsFusion-r3 candidate duration bounds must be 3 and 14400 seconds")
    if int(decoder.get("fixed_lag_seconds", -1)) > 60:
        raise ValueError("StatsFusion-r3 decoder latency cannot exceed 60 seconds")
    if "smooth_tau" in config.get("loss", {}):
        raise ValueError("StatsFusion-r3 uses smooth_beta, not the blocked smooth_tau loss")
    if float(config.get("loss", {}).get("smooth_beta", 0)) != 0.5:
        raise ValueError("StatsFusion-r3 Huber smooth_beta must be 0.5")
    verifier_config = config.get("verifier", {})
    if bool(verifier_config.get("use_learned_query_pooling", False)):
        channels = int(verifier_config.get("hidden_channels", 64))
        heads = int(verifier_config.get("attention_heads", 4))
        if heads <= 0 or channels % heads:
            raise ValueError("Learned-query verifier attention_heads must divide hidden_channels")
    boundary_config = config.get("boundary", {})
    if bool(boundary_config.get("use_proposal_conditioning", False)) and int(
        boundary_config.get("proposal_condition_dim", 0)
    ) != 4:
        raise ValueError("Proposal-conditioned Boundary requires proposal_condition_dim: 4")
    loss_config = config.get("loss", {})
    contrastive_weight = float(loss_config.get("temporal_contrastive_weight", 0.0))
    if contrastive_weight < 0:
        raise ValueError("temporal_contrastive_weight must be non-negative")
    if contrastive_weight > 0:
        if float(loss_config.get("temporal_contrastive_temperature", 0.0)) <= 0:
            raise ValueError("Enabled temporal contrastive loss requires positive temperature")
        if int(loss_config.get("temporal_contrastive_radius_steps", 0)) <= 0:
            raise ValueError("Enabled temporal contrastive loss requires positive radius")
    selector_decoder_search = str(config.get("training", {}).get("selector_decoder_search", "full"))
    if selector_decoder_search not in {"full", "fixed"}:
        raise ValueError("training.selector_decoder_search must be 'full' or 'fixed'")
    configured_state_seeds(config)
    verifier_seeds = [int(value) for value in config.get("verifier", {}).get("seeds", [])]
    expected_verifier_seeds = [2026] if ablation_id == "R3-DIRECT" else [2026, 2027, 2028]
    if verifier_seeds != expected_verifier_seeds:
        raise ValueError(
            f"{ablation_id or 'StatsFusion-r3'} requires verifier seeds {expected_verifier_seeds}"
        )
    if str(config.get("hierarchical", {}).get("downstream_mode", "full")) in {
        "pooled_heads",
        "pooled_deep_only",
    }:
        if config.get("hierarchical", {}).get("pooled_head_protocol") != POOLED_HEAD_PROTOCOL:
            raise ValueError(
                f"Pooled heads require pooled_head_protocol: {POOLED_HEAD_PROTOCOL}"
            )
        for section in ("verifier", "boundary"):
            minimum_epochs = int(config.get(section, {}).get("minimum_epochs", 0))
            maximum_epochs = int(config.get(section, {}).get("max_epochs", 0))
            patience = int(config.get(section, {}).get("patience", 0))
            if not 1 <= minimum_epochs <= maximum_epochs:
                raise ValueError(f"{section} minimum_epochs must lie within its epoch budget")
            if patience != 8:
                raise ValueError(f"Pooled {section} patience must be 8")
