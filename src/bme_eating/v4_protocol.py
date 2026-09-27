from __future__ import annotations

from typing import Any

CODE_VERSION = "v4.3"
PROTOCOL_VERSION = "statsfusion-r3"
BLOCKED_PREDECESSORS = (
    "statsfusion-r0-blocked",
    "statsfusion-r1-blocked",
    "statsfusion-r2-blocked",
)
TARGET_SEMANTICS = "soft_interval_occupancy"
CALIBRATION_PROTOCOL = "soft_platt_v1"
DECODER_PROTOCOL = "coherent_fixed_lag_segment_v1"
RAW_INPUT_SCHEMA = "statsfusion-raw-v2"
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


def validate_r3_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    protocol = experiment.get("protocol_version")
    if protocol != PROTOCOL_VERSION:
        raise ValueError(f"Formal v4.3 runs require protocol_version: {PROTOCOL_VERSION}")
    if experiment.get("code_version") != CODE_VERSION:
        raise ValueError(f"Formal StatsFusion-r3 runs require code_version: {CODE_VERSION}")
    ablation_id = str(experiment.get("ablation_id", ""))
    if ablation_id not in R3_ABLATION_IDS:
        raise ValueError(f"StatsFusion-r3 uses an unsupported ablation_id: {ablation_id!r}")
    if ablation_id == "R3-DIRECT":
        if experiment.get("variant") != "time_constrained_direct_d2c_ppg_semimarkov":
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
        if not bool(config.get("decoder", {}).get("use_semi_markov", False)):
            raise ValueError("R3-DIRECT requires the coherent Semi-Markov decoder")
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
    state_seeds = [int(value) for value in config.get("final_training", {}).get("state_seeds", [])]
    verifier_seeds = [int(value) for value in config.get("verifier", {}).get("seeds", [])]
    if state_seeds != [2026, 2027, 2028] or verifier_seeds != [2026, 2027, 2028]:
        raise ValueError("StatsFusion-r3 requires state and verifier seeds 2026/2027/2028")
