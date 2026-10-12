from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from _bootstrap import PROJECT_ROOT

from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs, SessionChunkSampler, make_joint_dataset
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_evaluation import NonnegativePlatt
from bme_eating.joint_pipeline import JointChunkDetector
from bme_eating.joint_progress import progress
from bme_eating.joint_protocol import configure_numerics, require_idle_gpu
from bme_eating.joint_stream import FrozenSourceCDF
from bme_eating.metrics import partition_evaluation_events
from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
from bme_eating.training.statsfusion_joint_trainer import enforce_resources, pack_features


def compare_events(reference, actual):
    fields = ("proposal_id", "start_ms", "end_ms", "decision_time_ms", "max_input_time_ms")
    if [[item[name] for name in fields] for item in reference] != [
            [item[name] for name in fields] for item in actual]:
        raise RuntimeError("Real session chunking changes committed geometry or identity")
    error = max((abs(left[name] - right[name]) for left, right in zip(reference, actual)
        for name in ("rank_score", "predicted_quality")), default=0.0)
    if error > 1e-6:
        raise RuntimeError(f"Real session replay score error exceeds 1e-6: {error}")
    return error


def replay(model, inputs, config, preprocessing, key, chunk_steps):
    config = copy.deepcopy(config)
    config["sequence"]["supervised_steps"] = chunk_steps
    dataset, _ = make_joint_dataset(inputs, {key[0]}, config, preprocessing, training=False)
    sampler = SessionChunkSampler(dataset, 0, training=False)
    chunks = [item for item in sampler.chunks if item[3] == key]
    mapping = FrozenSourceCDF({name: [0, 1] for name in
        ("state", "proposal", "jitter", "backtrack", "start_expansion")},
        preprocessing["training_subjects"], fallback=True)
    point = {"setting": "high_0.1_low_0.9_gap_15", "budget": 30, "acceptance": .01,
             "nms": .3, "calibration": NonnegativePlatt(1, 0, True).to_json(),
             "boundary_enabled": False}
    detector = JointChunkDetector(model, config, point, mapping, "disposable_real_replay",
                                  {"qualified": False}, *key)
    tiny = JointChunkDetector(model, config, point, mapping, "disposable_real_replay",
                              {"qualified": False}, *key)
    timeline, features, events, tiny_events = [], [], [], []
    started = time.monotonic()
    device = next(model.parameters()).device
    usage = {}
    with torch.no_grad():
        for row, _, mask, _, last, count in progress(chunks, total=len(chunks),
                description=f"Real replay {chunk_steps} {key[0]}/{key[1]}"):
            batch = dataset[(row, 0, mask)]
            batch = {name: value[None].to(device) if isinstance(value, torch.Tensor) else value
                     for name, value in batch.items()}
            outputs = model(batch)
            selected = torch.zeros_like(batch["supervision_mask"][0], dtype=torch.bool)
            selected[-count:] = True
            times = batch["timestamp_ms"][0, selected].cpu()
            values = pack_features(outputs, selected).cpu()
            if len(times) != count or values.requires_grad:
                raise RuntimeError("Real replay loses timeline bins or retains a computation graph")
            timeline.append(times)
            features.append(values)
            events.extend(detector.push_features(times, values, final=last))
            for first in range(0, len(times), 17):
                stop = min(first + 17, len(times))
                tiny_events.extend(tiny.push_features(times[first:stop], values[first:stop],
                                                       final=last and stop == len(times)))
            inputs.assert_session_unchanged(key)
            usage = enforce_resources(config, device)
    error = compare_events(events, tiny_events)
    return {"times": torch.cat(timeline), "features": torch.cat(features), "events": events,
            "report": {"chunk_steps": chunk_steps, "chunks": len(chunks),
                "seconds": time.monotonic() - started, "events": len(events),
                "incremental_feature_chunk_steps": 17, "incremental_score_max_error": error,
                "detached_history": True, "resources": usage}}


def main():
    parser = argparse.ArgumentParser(description="Disposable v413 real long/short session replay")
    parser.add_argument("--config", default="configs/statsfusion_v413_integrated_repair.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--audit", default=".modeling/v413_real_training_audit.json")
    parser.add_argument("--output", default=".modeling/v413_real_session_replay.json")
    arguments = parser.parse_args()
    config = load_config(PROJECT_ROOT / arguments.config)
    config["training"].update({"device": arguments.device, "num_workers": 0})
    require_idle_gpu(arguments.device)
    configure_numerics(config)
    _load_project_dotenv(config)
    output_base = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output_base / "v2", output_base / "v4")
    audit = json.loads((PROJECT_ROOT / arguments.audit).read_text(encoding="utf-8"))
    checkpoint_path = Path(audit["disposable_root"]) / "audit_checkpoint.pt"
    if sha256_file(checkpoint_path) != audit["checkpoint_sha256"]:
        raise RuntimeError("Disposable replay audit checkpoint changed")
    checkpoint = torch.load(checkpoint_path,
                            map_location="cpu", weights_only=False)
    if checkpoint.get("disposable_audit") is not True or checkpoint.get("weights_enter_formal_run") is not False:
        raise RuntimeError("Real replay requires a disposable audit checkpoint")
    model = build_joint_model(config).to(arguments.device).eval()
    model.load_state_dict(checkpoint["model"])
    legal = {subject for subject, fold in inputs.subject_folds.items() if fold != 0}
    legal -= set(checkpoint["training_subjects"])
    anchors, _, events, _ = inputs.frames(legal, labels=True)
    truth, _ = partition_evaluation_events(events, legal)
    truth = truth.assign(duration=truth.end_ms - truth.start_ms)
    longest = truth.sort_values(["duration", "subject_key", "session_id"], ascending=[False, True, True]).iloc[0]
    long_key = (str(longest.subject_key), str(longest.session_id))
    lengths = anchors.groupby(["subject_key", "session_id"]).size().to_dict()
    short_key = min((key for key in lengths if key != long_key), key=lambda key: (lengths[key], key))
    reports = []
    for key in (short_key, long_key):
        reference = replay(model, inputs, config, checkpoint["preprocessing"], key, 256)
        alternate = replay(model, inputs, config, checkpoint["preprocessing"], key, 128)
        if not torch.equal(reference["times"], alternate["times"]):
            raise RuntimeError("Real session encoder chunking changes timestamps")
        error = float((reference["features"] - alternate["features"]).abs().max())
        if error > 1e-5:
            raise RuntimeError(f"Real session encoder feature error exceeds 1e-5: {error}")
        score_error = compare_events(reference["events"], alternate["events"])
        reports.append({"subject": key[0], "session": key[1], "bins": len(reference["times"]),
            "duration_seconds": len(reference["times"]) * 3,
            "longest_truth_seconds": int(longest.duration) / 1000 if key == long_key else None,
            "feature_max_error": error, "score_max_error": score_error,
            "runs": [reference["report"], alternate["report"]]})
    write_json_atomic(Path(arguments.output), {"disposable_model": True, "accuracy_evidence": False,
        "outer_excluded": True, "device": arguments.device, "sessions": reports,
        "audit_checkpoint_sha256": audit["checkpoint_sha256"],
        "event_identity_geometry_and_decision_consistent": True,
        "feature_tolerance": 1e-5, "score_tolerance": 1e-6})
    print("Real long/short session replay passed", flush=True)


if __name__ == "__main__":
    main()
