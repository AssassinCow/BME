from __future__ import annotations

import argparse
import copy
import tempfile
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import configure_numerics
from bme_eating.joint_stream import FrozenSourceCDF
from bme_eating.joint_v411_search import choose_point
from bme_eating.training.statsfusion_joint_trainer import (
    predict_joint_model,
    seed_model,
)


def benchmark_search(candidate_sets, windows, truth, ignore, mappings, config):
    started = time.monotonic()
    optimized, records = choose_point(copy.deepcopy(candidate_sets), windows, truth, ignore, mappings, config)
    optimized_seconds = time.monotonic() - started
    from bme_eating.joint_evaluation import choose_operating_point
    started = time.monotonic()
    reference_config = {**config, "_reference_execution": True}
    reference, reference_records = choose_operating_point(copy.deepcopy(candidate_sets), windows, truth,
                                                           ignore, mappings, reference_config)
    reference_seconds = time.monotonic() - started
    if len(records) != 16038 or len(reference_records) != 16038:
        raise RuntimeError("Performance comparison did not execute all 16038 points")
    for left, right in zip(records, reference_records):
        for name in ("setting", "budget", "nms", "acceptance", "true_positive", "false_positive",
                     "false_negative", "promotion_eligible"):
            if left[name] != right[name]:
                raise RuntimeError(f"Optimized search differs from reference: {name}")
    for name in ("setting", "budget", "nms", "acceptance"):
        if optimized[name] != reference[name]:
            raise RuntimeError("Operating point tie break differs from reference")
    return {"same_candidates": True, "all_operating_points": 16038, "metrics_and_selection_identical": True,
            "optimized_seconds": optimized_seconds, "reference_seconds": reference_seconds,
            "speedup": reference_seconds / max(optimized_seconds, 1e-9)}


def benchmark_validation(model, inputs, subjects, config, preprocessing, directory, identity, mapping):
    timings, predictions = {}, {}
    warmups = {}
    for name, reference in (("optimized", False), ("reference", True)):
        warmups[name] = predict_joint_model(model, inputs, subjects,
            {**config, "_reference_execution": reference}, preprocessing,
            directory / ("warmup_" + name), identity, {"search_eligible": False}, mapping=mapping)
    for name, reference in (("optimized", False), ("reference", True)):
        started = time.monotonic()
        predictions[name] = predict_joint_model(model, inputs, subjects,
            {**config, "_reference_execution": reference}, preprocessing, directory / name, identity,
            {"search_eligible": False}, mapping=mapping)
        timings[name] = time.monotonic() - started
    optimized = predictions["optimized"]
    from bme_eating.metrics import partition_evaluation_events
    _, _, events, _ = inputs.frames(subjects, labels=True)
    truth, ignore = partition_evaluation_events(events, subjects)
    search = benchmark_search(optimized["coarse"], optimized["windows"], truth, ignore,
                              {identity: mapping}, config)
    from bme_eating.joint_evaluation import choose_operating_point
    started = time.monotonic()
    reference_point, reference_records = choose_operating_point(predictions["reference"]["coarse"],
        predictions["reference"]["windows"], truth, ignore, {identity: mapping},
        {**config, "_reference_execution": True})
    reference_search = time.monotonic() - started
    optimized_point, optimized_records = choose_point(optimized["coarse"], optimized["windows"],
                                                     truth, ignore, {identity: mapping}, config)
    for left, right in zip(optimized_records, reference_records):
        for name in ("setting", "budget", "nms", "acceptance", "true_positive", "false_positive", "false_negative"):
            if left[name] != right[name]:
                raise RuntimeError("Batch inference or admitted-candidate union changed operating point metrics")
    if any(optimized_point[name] != reference_point[name] for name in ("setting", "budget", "nms", "acceptance")):
        raise RuntimeError("Full validation changed chosen operating point")
    fast = timings["optimized"] + search["optimized_seconds"]
    slow = timings["reference"] + reference_search
    return {"search": search, "inference_seconds": timings, "full_validation_seconds": {
        "optimized": fast, "reference": slow}, "speedup": slow / max(fast, 1e-9),
        "two_times_target_met": slow >= 2 * fast, "same_model": True,
        "boundary_search_enabled": False, "all_operating_points": 16038,
        "time_features_warmed_for_both_paths": True,
        "warmup_performance": {name: result["performance"] for name, result in warmups.items()},
        "performance": {name: output["performance"] for name, output in predictions.items()}}


def real_audit_benchmark(config, output):
    import json

    import torch

    from bme_eating.data.joint_sessions import JointInputs
    _load_project_dotenv(config)
    base = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(base / "v2", base / "v4")
    audit = json.loads((PROJECT_ROOT / ".modeling/v413_real_training_audit.json").read_text(encoding="utf-8"))
    disposable = Path(audit["disposable_root"])
    if sha256_file(disposable / "audit_checkpoint.pt") != audit["checkpoint_sha256"]:
        raise RuntimeError("Disposable benchmark audit checkpoint changed")
    checkpoint = torch.load(disposable / "audit_checkpoint.pt",
                            map_location="cpu", weights_only=False)
    model = seed_model(config).eval()
    training = set(checkpoint["training_subjects"])
    legal = {subject for subject, fold in inputs.subject_folds.items() if fold != 0} - training
    anchors, _, events, _ = inputs.frames(legal, labels=True)
    from bme_eating.metrics import partition_evaluation_events
    truth, _ = partition_evaluation_events(events, legal)
    lengths = anchors.groupby(["subject_key", "session_id"]).size().to_dict()
    positive_keys = set(truth[["subject_key", "session_id"]].astype(str).itertuples(index=False, name=None))
    positive = min(positive_keys, key=lambda key: (lengths[key], key))
    negatives = set(lengths) - positive_keys
    negative = min(negatives, key=lambda key: (abs(lengths[key] - lengths[positive]), key))
    keys = {positive, negative}
    subjects = {key[0] for key in keys}
    class SubsetInputs:
        def __init__(self):
            self.identity = {**inputs.identity, "benchmark_sessions": sorted(keys)}

        def frames(self, requested, *, labels):
            frames = inputs.frames(requested, labels=labels)
            return tuple(frame[[((str(subject), str(session)) in keys) for subject, session in
                frame[["subject_key", "session_id"]].itertuples(index=False, name=None)]]
                if len(frame) else frame for frame in frames)

        def assert_session_unchanged(self, key):
            inputs.assert_session_unchanged(key)
    replay = torch.load(next((disposable / "epoch_3").glob("*.pt")),
        map_location="cpu", weights_only=False)
    mapping = FrozenSourceCDF.fit(replay["candidates"], training)
    mapping.fallback = True
    config = copy.deepcopy(config)
    config["training"].update({"device": "cpu", "num_workers": 10})
    from bme_eating.joint_v412_cache import feature_binding
    binding = feature_binding(model, inputs, config, checkpoint["preprocessing"], "disposable_benchmark")
    if binding["weights_sha256"] != replay["binding"]["weights_sha256"]:
        raise RuntimeError("Benchmark model and natural source CDF use different weights")
    report = benchmark_validation(model, SubsetInputs(), subjects, config, checkpoint["preprocessing"],
        Path(tempfile.gettempdir()) / "bme_v413_validation" / "benchmark", "disposable_benchmark", mapping)
    report.update({"disposable_model": True, "accuracy_evidence": False, "sessions": sorted(keys),
        "source_replay_seconds": audit["natural_replay"]["seconds"], "gpu_benchmark": False,
        "audit_checkpoint_sha256": audit["checkpoint_sha256"],
        "model_weights_sha256": binding["weights_sha256"],
        "model_is_pre_update_replay_model": True,
        "config_sha256": binding["config_sha256"], "data_sha256": binding["data_sha256"],
        "feature_source_sha256": binding["feature_source_sha256"]})
    return report


def main():
    parser = argparse.ArgumentParser(description="v413 full-search reference performance comparison")
    parser.add_argument("--config", default="configs/statsfusion_v413_integrated_repair.yaml")
    parser.add_argument("--evidence", help="Torch payload containing candidate_sets/windows/truth/ignore/mappings")
    parser.add_argument("--real-audit", action="store_true")
    parser.add_argument("--output", default=".modeling/v413_search_benchmark.json")
    arguments = parser.parse_args()
    import torch
    config = load_config(PROJECT_ROOT / arguments.config)
    configure_numerics(config)
    if arguments.real_audit:
        result = real_audit_benchmark(config, arguments.output)
        write_json_atomic(Path(arguments.output), result)
        print(result, flush=True)
        return
    if not arguments.evidence:
        parser.error("--evidence or --real-audit is required")
    payload = torch.load(Path(arguments.evidence), map_location="cpu", weights_only=False)
    result = benchmark_search(payload["candidate_sets"], payload["windows"], payload["truth"], payload["ignore"],
                              {key: FrozenSourceCDF(**value) for key, value in payload["mappings"].items()}, config)
    result["complete_validation_benchmarked"] = False
    write_json_atomic(Path(arguments.output), result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
