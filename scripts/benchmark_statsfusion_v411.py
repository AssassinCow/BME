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
from bme_eating.joint_protocol import config_identity, digest, require_idle_gpu
from bme_eating.metrics import partition_evaluation_events
from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
from bme_eating.training.statsfusion_joint_trainer import fit_source_mapping, predict_joint_model


def main():
    parser = argparse.ArgumentParser(description="Compare identical v411 checkpoint and complete search")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/statsfusion_v411_learned_boundary.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subjects", required=True, help="JSON list of excluded selector or OOF subjects")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    _load_project_dotenv(config)
    require_idle_gpu(config["training"]["device"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    if payload["config_sha256"] != config_identity(config):
        raise RuntimeError("Benchmark configuration differs from checkpoint")
    subjects = set(json.loads(Path(arguments.subjects).read_text(encoding="utf-8")))
    training = set(payload["training_subjects"])
    if not subjects or subjects & training or not subjects.issubset(set(payload["excluded_subjects"])):
        raise RuntimeError("Benchmark subjects must be explicitly excluded from training")
    base = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(base / config["project"]["input_artifact_schema_version"],
                               base / config["project"]["canonical_schema_version"])
    if payload["data_sha256"] != digest(inputs.identity):
        raise RuntimeError("Benchmark data differs from checkpoint")
    model = build_joint_model(config).to(config["training"]["device"]).eval()
    model.load_state_dict(payload["model"])
    mapping = fit_source_mapping(model, inputs, training, config, payload["preprocessing"],
                                 output / "source_fit", payload["identity"])
    _, _, events, _ = inputs.frames(subjects, labels=True)
    truth, ignore = partition_evaluation_events(events, subjects)
    reports, registered = {}, None
    for name, reference in (("reference", True), ("optimized", False)):
        current = copy.deepcopy(config)
        current["_reference_execution"] = reference
        started = time.monotonic()
        predictions = predict_joint_model(model, inputs, subjects, current, payload["preprocessing"],
            output / name, payload["identity"], payload["boundary"], mapping=mapping)
        point, records = choose_operating_point(predictions["coarse"], predictions["windows"], truth,
                                               ignore, {payload["identity"]: mapping}, current)
        report = {"elapsed_seconds": time.monotonic() - started, "point": point,
                  "registered_operating_points": len(records), "prediction": predictions["performance"]}
        scores = [{key: record[key] for key in ("setting", "budget", "nms", "acceptance", "true_positive",
                  "false_positive", "false_negative", "promotion_eligible")} for record in records]
        if registered is not None and registered != scores:
            raise RuntimeError("Optimization changed matching, operating points or gate qualification")
        registered = scores
        reports[name] = report
    speedup = reports["reference"]["elapsed_seconds"] / reports["optimized"]["elapsed_seconds"]
    write_json_atomic(output / "benchmark.json", {"checkpoint_sha256": sha256_file(arguments.checkpoint),
        "reports": reports, "full_validation_speedup": speedup, "target_2x_met": speedup >= 2,
        "search_reduced": False, "device": config["training"]["device"]})


if __name__ == "__main__":
    main()
