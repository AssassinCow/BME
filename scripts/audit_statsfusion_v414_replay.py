from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT
from audit_statsfusion_v414_training import SelectedSessions

from bme_eating.artifacts import sha256_file, write_json_atomic
from bme_eating.candidate_scoring import score_candidates
from bme_eating.candidates import FrozenSourceCDF
from bme_eating.config import _load_project_dotenv, load_config, require_environment_path
from bme_eating.data.joint_sessions import JointInputs, SessionChunkSampler, make_joint_dataset
from bme_eating.data.session import SessionWindowReader
from bme_eating.data.stats_fusion_preprocess import RawSessionInput
from bme_eating.evaluation import (
    NonnegativePlatt,
    apply_online_budget,
    commit_candidates,
    metrics_with_exposure,
)
from bme_eating.metrics import partition_evaluation_events
from bme_eating.operating_points import CachedSearch
from bme_eating.progress import progress
from bme_eating.proposal_stream import make_stream
from bme_eating.protocol import (
    CHECKPOINT_STATE_VERSION,
    config_identity,
    configure_numerics,
    digest,
    protocol_for,
    require_idle_gpu,
    source_identity,
)
from bme_eating.streaming import JointDetector
from bme_eating.training.natural_replay import replay_session
from bme_eating.training.statsfusion_joint_trainer import (
    decoder_settings,
    enforce_resources,
    generate_chunk,
    seed_model,
)


def benchmark_search(model, inputs, config, preprocessing, audit):
    keys = sorted(audit["sessions"], key=lambda key: (audit["session_bins"]["|".join(key)], key))[:4]
    selected = SelectedSessions(inputs, set(map(tuple, keys)))
    subjects = {key[0] for key in keys}
    dataset, _ = make_joint_dataset(selected, subjects, config, preprocessing, training=True)
    sampler = SessionChunkSampler(dataset, 9, training=False)
    directory = Path(tempfile.mkdtemp(prefix="bme_v414_r2_search_"))
    settings = decoder_settings(config)
    pools, windows = {name: [] for name in settings}, []
    for key in map(tuple, keys):
        payload, _ = replay_session(model, selected, config, preprocessing, dataset, sampler,
            key, directory, "disposable_search", 9, 0)
        times, features = payload["times"], payload["features"]
        windows.extend({"subject_key": key[0], "session_id": key[1],
            "timestamp_ms": int(timestamp), "valid": float(valid)}
            for timestamp, valid in zip(times, features[:, 128]))
        for name, decoder in settings.items():
            stream = make_stream(*key, "disposable_search", decoder)
            pool = generate_chunk(stream, times, features)
            pool.extend(stream.finish())
            score_candidates(model, pool, times, features, next(model.parameters()).device,
                             batch_size=config["training"]["validation_head_batch_size"])
            pools[name].extend(pool)
    import pandas as pd
    windows = pd.DataFrame(windows)
    _, _, events, _ = selected.frames(subjects, labels=True)
    truth, ignore = partition_evaluation_events(events, subjects)
    mapping = FrozenSourceCDF.fit([item for pool in pools.values() for item in pool], subjects)
    mapping.fallback = True
    calibrator = NonnegativePlatt(1., 0., True)
    thresholds = np.arange(1, 100) / 100
    groups = [(name, budget) for name in settings for budget in config["search"]["budgets"]]
    reference_seconds, cached_seconds, points, errors = 0., 0., 0, 0
    for name, budget in progress(groups, total=len(groups), description="Full search equivalence"):
        admitted, _ = apply_online_budget(copy.deepcopy(pools[name]), windows,
            {"disposable_search": mapping}, budget, {})
        started = time.monotonic()
        cached = CachedSearch(admitted, truth, ignore, windows, calibrator)
        values = [cached.evaluate(float(threshold), nms)
                  for nms in config["search"]["nms_ious"] for threshold in thresholds]
        cached_seconds += time.monotonic() - started
        started = time.monotonic()
        reference = [metrics_with_exposure(truth, commit_candidates(admitted, calibrator,
            float(threshold), nms), ignore, windows)[0]
            for nms in config["search"]["nms_ious"] for threshold in thresholds]
        reference_seconds += time.monotonic() - started
        for actual, expected in zip(values, reference, strict=True):
            for metric in ("true_positive", "false_positive", "false_negative", "f1", "fp_per_hour"):
                errors += int(not math.isclose(actual[metric], expected[metric], rel_tol=0, abs_tol=1e-12))
        points += len(values)
    return {"passed": points == 16038 and errors == 0, "registered_points": points,
        "metric_mismatches": errors, "reference_seconds": reference_seconds,
        "cached_seconds": cached_seconds, "search_speedup": reference_seconds / max(cached_seconds, 1e-9),
        "target_twofold_met": reference_seconds >= cached_seconds * 2,
        "candidate_count": sum(map(len, pools.values())), "sessions": keys,
        "scope": "search_only_on_real_audit_subset", "calibration": "fixed_for_equivalence_only",
        "same_weights_and_candidates": True, "directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description="Disposable v414-r2 native replay and full search audit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--benchmark-search", action="store_true")
    arguments = parser.parse_args()
    config = load_config(PROJECT_ROOT / "configs/statsfusion_v414_full_data.yaml")
    config["training"]["device"] = arguments.device
    require_idle_gpu(arguments.device)
    configure_numerics(config)
    _load_project_dotenv(config)
    output = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output / "v2", output / "v4")
    audit = json.loads((PROJECT_ROOT / f"manifests/v414_r2_production_loop_{arguments.device}.json").read_text(encoding="utf-8"))
    checkpoint = Path(audit["directory"]) / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (not audit["passed"] or payload["protocol"] != protocol_for(config)
            or payload["state_version"] != CHECKPOINT_STATE_VERSION
            or audit["source_sha256"] != digest(source_identity(PROJECT_ROOT))):
        raise RuntimeError("Native replay requires a qualified disposable r2 audit checkpoint")
    detector = JointDetector()
    detector.model = seed_model(config).to(arguments.device).eval()
    detector.model.load_state_dict(payload["model"])
    detector.config, detector.preprocessing = config, payload["preprocessing"]
    detector.identity = "disposable_real_replay"
    detector.boundary = {"qualified": False, "mode": "off"}
    detector.mapping = FrozenSourceCDF({"state:3": [0., 1.], "proposal:3": [0., 1.],
        "proposal:15": [0., 1.], "proposal:60": [0., 1.]}, audit["training_subjects"], fallback=True)
    detector.point = {"setting": "high_0.1_low_0.9_gap_15", "budget": 30, "acceptance": 0.,
        "nms": .3, "calibration": {"slope": 1., "intercept": 0., "qualified": True}, "boundary_enabled": False}
    if arguments.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sessions = [("64124e0fe3248a92d711", "cc14017e0f79af253445", 29.),
                ("64124e0fe3248a92d711", "f5ea5ae8a1746ea19bba", 13022.)]
    results, started = [], time.monotonic()
    for subject, session, duration in sessions:
        _, _, _, segments = inputs.frames({subject}, labels=False)
        segments = segments[segments.session_id.astype(str).eq(session)]
        reader = SessionWindowReader(segments, cache_size=1)
        arrays = reader.read(session, int(segments.start_ms.min()), int(segments.end_ms.max()), subject_key=subject)
        raw = RawSessionInput(subject_key=subject, session_id=session,
            **{name: arrays[name] for name in ("motion_timestamp_ms", "motion_values", "motion_mask",
                "ppg_timestamp_ms", "ppg_values", "ppg_mask")}).validated()
        outputs, seconds = [], []
        for chunk in (128, 256):
            begin = time.monotonic()
            print(f"real replay {session} chunk={chunk}", flush=True)
            outputs.append(detector.predict_session(raw, new_steps=chunk))
            seconds.append(time.monotonic() - begin)
        identities = [[item.event_id for item in events] for events in outputs]
        error = max((abs(left.score - right.score) for left, right in zip(*outputs)), default=0.)
        if identities[0] != identities[1] or error > 1e-6:
            raise RuntimeError("Real native replay chunk or score alignment failed")
        results.append({"subject": subject, "session": session, "selection_duration_seconds": duration,
            "motion_samples": len(raw.motion_timestamp_ms), "ppg_samples": len(raw.ppg_timestamp_ms),
            "chunks": [128, 256], "elapsed_seconds": seconds, "event_count": len(outputs[0]),
            "score_max_error": error, "identity_alignment": True})
    report = {"passed": True, "disposable_audit_weights": True, "formal_run_started": False,
        "synthetic_cdf_and_calibration_for_consistency_only": True, "label_use": "audit_selection_and_search_only",
        "checkpoint_sha256": sha256_file(checkpoint), "source_sha256": digest(source_identity(PROJECT_ROOT)),
        "data_sha256": digest(inputs.identity), "config_sha256": config_identity(config),
        "sessions": results, "elapsed_seconds": time.monotonic() - started}
    if arguments.benchmark_search:
        report["search_benchmark"] = benchmark_search(detector.model, inputs, config, payload["preprocessing"], audit)
        report["passed"] &= report["search_benchmark"]["passed"]
    report["resources"] = enforce_resources(config, torch.device(arguments.device))
    write_json_atomic(PROJECT_ROOT / f"manifests/v414_r2_real_replay_{arguments.device}.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise RuntimeError("Native replay/search audit failed")


if __name__ == "__main__":
    main()
