from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_evaluation import choose_operating_point
from bme_eating.joint_protocol import (
    checkpoint_qualified,
    config_identity,
    configure_numerics,
    digest,
    require_idle_gpu,
    source_identity,
    validate_config,
)
from bme_eating.metrics import partition_evaluation_events
from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
from bme_eating.training.statsfusion_joint_trainer import fit_source_mapping, predict_joint_model


def require_benchmark_checkpoint(payload, config, subjects, data_identity):
    validate_config(config)
    if config["model"]["architecture"] != "statsfusion_joint_event_v412":
        raise RuntimeError("Benchmark requires the registered v412 architecture")
    training = set(payload.get("training_subjects", []))
    excluded = set(payload.get("excluded_subjects", []))
    identity = payload.get("identity")
    if (not isinstance(identity, str) or not identity or not training
            or not checkpoint_qualified(payload, identity, training, config["training"]["checkpoint_min_epoch"])):
        raise RuntimeError("Benchmark requires a complete finite and clipping-qualified checkpoint")
    if not subjects or training & excluded or subjects & training or not subjects.issubset(excluded):
        raise RuntimeError("Benchmark subjects must be explicitly excluded from training")
    source = source_identity(Path(config["_config_path"]).resolve().parents[1])
    if (payload.get("config_sha256") != config_identity(config)
            or payload.get("source_sha256") != digest(source)
            or payload.get("data_sha256") != digest(data_identity)):
        raise RuntimeError("Benchmark checkpoint configuration/source/data identity differs")
    if set(payload.get("preprocessing", {}).get("training_subjects", [])) != training:
        raise RuntimeError("Benchmark preprocessing training lineage differs from checkpoint")
    return training


def main():
    parser = argparse.ArgumentParser(description="Compare complete v412 validation without reducing registered search")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/statsfusion_v412_integrated_repair.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subjects", required=True, help="JSON list of explicitly excluded selector or internal subjects")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    validate_config(config)
    configure_numerics(config)
    _load_project_dotenv(config)
    require_idle_gpu(config["training"]["device"])
    payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    subjects = set(json.loads(Path(arguments.subjects).read_text(encoding="utf-8")))
    base = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(base / config["project"]["input_artifact_schema_version"],
                               base / config["project"]["canonical_schema_version"])
    training = require_benchmark_checkpoint(payload, config, subjects, inputs.identity)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=False)
    model = build_joint_model(config).to(config["training"]["device"]).eval()
    model.load_state_dict(payload["model"])
    _, _, events, _ = inputs.frames(subjects, labels=True)
    truth, ignore = partition_evaluation_events(events, subjects)
    reports, registered = {}, None
    for name, reference in (("reference", True), ("optimized", False)):
        current = copy.deepcopy(config)
        current["_reference_execution"] = reference
        started = time.monotonic()
        mapping = fit_source_mapping(model, inputs, training, current, payload["preprocessing"],
                                     output / name / "source_fit", payload["identity"])
        cdf_seconds = time.monotonic() - started
        predictions = predict_joint_model(model, inputs, subjects, current, payload["preprocessing"],
            output / name / "prediction", payload["identity"],
            {**payload["boundary"], "search_eligible": False}, mapping=mapping)
        point, records = choose_operating_point(predictions["coarse"], predictions["windows"], truth,
                                               ignore, {payload["identity"]: mapping}, current)
        report = {"elapsed_seconds": time.monotonic() - started, "cdf_seconds": cdf_seconds,
                  "point": point, "registered_operating_points": len(records), "prediction": predictions["performance"]}
        scores = [{key: record[key] for key in ("setting", "budget", "nms", "acceptance", "true_positive",
                  "false_positive", "false_negative", "promotion_eligible")} for record in records]
        if registered is not None and registered != scores:
            raise RuntimeError("Optimization changed matching, operating points or gate qualification")
        if len(records) != 8019:
            raise RuntimeError("Benchmark must retain all 8019 coarse operating points")
        registered = scores
        reports[name] = report
    speedup = reports["reference"]["elapsed_seconds"] / reports["optimized"]["elapsed_seconds"]
    write_json_atomic(output / "benchmark.json", {"checkpoint_sha256": sha256_file(arguments.checkpoint),
        "checkpoint_qualified": True, "config_sha256": payload["config_sha256"],
        "source_sha256": payload["source_sha256"], "data_sha256": payload["data_sha256"],
        "training_subjects": sorted(training), "prediction_subjects": sorted(subjects),
        "reports": reports, "full_validation_speedup": speedup, "target_2x_met": speedup >= 2,
        "search_reduced": False, "scope": "CDF_encoder_ROI_head_coarse_search", "boundary_mode": "off"})


if __name__ == "__main__":
    main()
