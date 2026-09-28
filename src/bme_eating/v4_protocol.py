from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CODE_VERSION = "v4.4.2"
PROTOCOL_VERSION = "statsfusion-r3.2"
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
    "models/factory.py",
    "models/hierarchical_state.py",
    "models/stats_fusion_state.py",
)
R3_ABLATION_IDS = frozenset(
    {
        "R3-S0",
        "R3-S1",
        "R3-S2",
        "R3-D1",
        "R3-D2a",
        "R3-D2b",
        "R3-D2c",
        "R3-P1",
        "R3-M1",
        "R3-DIRECT",
    }
)

R3_ROOT_CONFIG_KEYS = frozenset(
    {
        "project",
        "data",
        "experiment",
        "features",
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
    if experiment.get("code_version") != CODE_VERSION:
        raise ValueError(f"Formal {PROTOCOL_VERSION} runs require code_version: {CODE_VERSION}")
    ablation_id = str(experiment.get("ablation_id", ""))
    if ablation_id not in R3_ABLATION_IDS:
        raise ValueError(f"StatsFusion-r3 uses an unsupported ablation_id: {ablation_id!r}")
    if ablation_id == "R3-DIRECT":
        direct_variants = {"time_constrained_outer_single_holdout_logistic"}
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
        if bool(decoder.get("use_transition_candidates", True)):
            raise ValueError("R3-DIRECT disables transition candidates")
        if config.get("training", {}).get("selector_decoder_search") != "fixed":
            raise ValueError("R3-DIRECT requires fixed decoder search during state selection")
        single_holdout = bool(experiment.get("time_constrained_single_holdout", False))
        variant_is_single_holdout = (
            experiment.get("variant") == "time_constrained_outer_single_holdout_logistic"
        )
        if single_holdout != variant_is_single_holdout:
            raise ValueError("Single-holdout variant and protocol flag must agree")
        state_crossfit_mode = str(config.get("hierarchical", {}).get("state_crossfit_mode", "full"))
        downstream_mode = str(config.get("hierarchical", {}).get("downstream_mode", "full"))
        if single_holdout:
            if state_crossfit_mode != "single_holdout":
                raise ValueError("Single-holdout direct training requires state_crossfit_mode")
            if downstream_mode != "pooled_logistic":
                raise ValueError(
                    "Single-holdout direct training must use pooled-logistic downstream"
                )
            fixed_epochs = int(config.get("training", {}).get("fixed_state_epochs", 0))
            holdout_fraction = float(config.get("training", {}).get("single_holdout_fraction", 0.0))
            if not 1 <= fixed_epochs <= 12:
                raise ValueError("Single-holdout fixed_state_epochs must be in [1, 12]")
            if not 0.2 <= holdout_fraction < 0.5:
                raise ValueError("Single-holdout fraction must be in [0.2, 0.5)")
            training = config.get("training", {})
            if not bool(training.get("subject_balanced_sampling", False)):
                raise ValueError("R3-DIRECT requires subject-balanced sampling")
            if int(training.get("clips_per_subject_per_epoch", 0)) <= 0:
                raise ValueError("R3-DIRECT requires positive clips_per_subject_per_epoch")
            for key in (
                "ppg_modality_dropout",
                "gyro_modality_dropout",
                "rotation_augmentation_probability",
            ):
                if float(training.get(key, 0.0)) != 0.0:
                    raise ValueError(f"R3-DIRECT requires {key}: 0")
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
