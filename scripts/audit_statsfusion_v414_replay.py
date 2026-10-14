from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile
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
    PROFILE,
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


def candidate_reference(archive):
    with zipfile.ZipFile(archive) as snapshot:
        source = snapshot.read("src/bme_eating/proposal_stream.py")
    if hashlib.sha256(source).hexdigest() != "1ea5384e37b244c34b7ca2755bd3c05b8073486a4ff98db1e281afe7e31e8bbc":
        raise RuntimeError("Candidate comparison requires the immutable c7d6be3 candidate source")
    module = types.ModuleType("disposable_r3_candidate_reference")
    exec(compile(source, str(archive) + "!proposal_stream.py", "exec"), module.__dict__)
    return module.make_stream


def coverage(pool, truth):
    from bme_eating.candidates import candidate_targets
    covered = {candidate_targets(item, truth)[2] for item in pool if candidate_targets(item, truth)[0]}
    return {"covered_events": len(covered), "support_events": len(truth),
            "covered_event_ids": sorted(covered), "candidates": len(pool)}


@torch.inference_mode()
def benchmark_search(model, inputs, config, preprocessing, audit, baseline_archive=None):
    all_keys = sorted(map(tuple, audit["sessions"]), key=lambda key: (audit["session_bins"]["|".join(key)], key))
    keys = all_keys[:4]
    comparison_keys = all_keys if baseline_archive else keys
    selected = SelectedSessions(inputs, set(map(tuple, comparison_keys)))
    subjects = {key[0] for key in comparison_keys}
    dataset, _ = make_joint_dataset(selected, subjects, config, preprocessing, training=True)
    sampler = SessionChunkSampler(dataset, 9, training=False)
    directory = Path(tempfile.mkdtemp(prefix=f"bme_{PROFILE.repair_revision.replace('-', '_')}_search_"))
    settings = decoder_settings(config)
    pools, windows = {name: [] for name in settings}, []
    comparison_pools = {name: {"r3": [], "r4": []} for name in settings}
    parent_candidates, comparison_windows, scoring = [], [], {"r3": {}, "r4": {}}
    reference_stream = candidate_reference(baseline_archive) if baseline_archive else None
    for key in map(tuple, comparison_keys):
        payload, _ = replay_session(model, selected, config, preprocessing, dataset, sampler,
            key, directory, "disposable_search", 9, 0)
        times, features = payload["times"], payload["features"]
        session_windows = [{"subject_key": key[0], "session_id": key[1],
            "timestamp_ms": int(timestamp), "valid": float(valid)}
            for timestamp, valid in zip(times, features[:, 128])]
        comparison_windows.extend(session_windows)
        if key in keys:
            windows.extend(session_windows)
        for name, decoder in settings.items():
            parent_stream = make_stream(*key, "disposable_search", {**decoder, "parent_proposals_only": True})
            parent_candidates.extend(generate_chunk(parent_stream, times, features))
            parent_candidates.extend(parent_stream.finish())
            stream = make_stream(*key, "disposable_search", decoder)
            pool = generate_chunk(stream, times, features)
            pool.extend(stream.finish())
            score_candidates(model, pool, times, features, next(model.parameters()).device,
                batch_size=config["training"]["validation_head_batch_size"], performance=scoring["r4"])
            comparison_pools[name]["r4"].extend(pool)
            if key in keys:
                pools[name].extend(pool)
            if reference_stream:
                previous = reference_stream(*key, "disposable_search", decoder)
                previous_pool = generate_chunk(previous, times, features)
                previous_pool.extend(previous.finish())
                score_candidates(model, previous_pool, times, features, next(model.parameters()).device,
                    batch_size=config["training"]["validation_head_batch_size"], performance=scoring["r3"])
                comparison_pools[name]["r3"].extend(previous_pool)
    import pandas as pd
    windows = pd.DataFrame(windows)
    _, _, comparison_events, _ = selected.frames(subjects, labels=True)
    comparison_truth, comparison_ignore = partition_evaluation_events(comparison_events, subjects)
    search_inputs = SelectedSessions(inputs, set(map(tuple, keys)))
    _, _, events, _ = search_inputs.frames(subjects, labels=True)
    truth, ignore = partition_evaluation_events(events, subjects)
    mapping = FrozenSourceCDF.fit(parent_candidates, subjects)
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
    result = {"passed": points == 16038 and errors == 0, "registered_points": points,
        "metric_mismatches": errors, "reference_seconds": reference_seconds,
        "cached_seconds": cached_seconds, "search_speedup": reference_seconds / max(cached_seconds, 1e-9),
        "target_twofold_met": reference_seconds >= cached_seconds * 2,
        "candidate_count": sum(map(len, pools.values())), "sessions": keys,
        "scope": "search_only_on_real_audit_subset", "calibration": "fixed_for_equivalence_only",
        "same_weights_and_candidates": True, "directory": str(directory)}
    if reference_stream:
        comparison_windows = pd.DataFrame(comparison_windows)
        comparison = []
        for name in settings:
            raw = {revision: coverage(pool, comparison_truth) for revision, pool in comparison_pools[name].items()}
            for budget in config["search"]["budgets"]:
                admitted = {revision: apply_online_budget(copy.deepcopy(pool), comparison_windows,
                    {"disposable_search": mapping}, budget, {})[0]
                    for revision, pool in comparison_pools[name].items()}
                row = {"decoder": name, "budget": budget, "raw": raw,
                    "admitted": {revision: coverage(pool, comparison_truth) for revision, pool in admitted.items()}}
                row["lost_after_budget"] = sorted(set(row["admitted"]["r3"]["covered_event_ids"])
                    - set(row["admitted"]["r4"]["covered_event_ids"]))
                row["gained_after_budget"] = sorted(set(row["admitted"]["r4"]["covered_event_ids"])
                    - set(row["admitted"]["r3"]["covered_event_ids"]))
                row["fixed_diagnostic_metrics"] = {revision: metrics_with_exposure(comparison_truth,
                    commit_candidates(pool, calibrator, .50, .30), comparison_ignore, comparison_windows)[0]
                    for revision, pool in admitted.items()}
                comparison.append(row)
        result["candidate_comparison"] = {"baseline_candidate_commit": "c7d6be3",
            "same_model_weights": True, "same_parent_cdf": True,
            "cdf_from_natural_training_parents": True, "sessions": comparison_keys,
            "scope": "five_real_disposable_training_sessions", "scoring": scoring,
            "records": comparison, "uses_comparison_to_select_epoch_or_threshold": False,
            "formal_effect_evidence": False}
    return result


def audit_bundle_replay(detector, checkpoint, raw, expected):
    directory = Path(tempfile.mkdtemp(prefix="bme_disposable_bundle_"))
    bundle = directory / "bundle"
    bundle.mkdir()
    shutil.copyfile(checkpoint, bundle / "checkpoint.pt")
    write_json_atomic(bundle / "resolved_config.json", detector.config)
    write_json_atomic(bundle / "inference.json", {"point": detector.point,
        "mapping": detector.mapping.to_json(), "identity": detector.identity,
        "boundary": detector.boundary, "subject": raw.subject_key, "session": raw.session_id,
        "artifact_kind": "disposable_audit_bundle", "official_test_candidate": False,
        "promotable": False, "calibration_is_synthetic_for_consistency": True})
    for relative in source_identity(PROJECT_ROOT):
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, target)
    hashes = {path.relative_to(bundle).as_posix(): sha256_file(path)
              for path in bundle.rglob("*") if path.is_file()}
    write_json_atomic(bundle / "SHA256SUMS.json", hashes)
    raw_path = directory / "private_real_input.npz"
    np.savez(raw_path, **{name: getattr(raw, name) for name in (
        "motion_timestamp_ms", "motion_values", "motion_mask",
        "ppg_timestamp_ms", "ppg_values", "ppg_mask")})
    code = """import builtins,json,sys
from pathlib import Path
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0]=='xgboost':
        raise ImportError('xgboost forbidden')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import numpy as np
import torch
from bme_eating.artifacts import sha256_file
from bme_eating.candidates import FrozenSourceCDF
from bme_eating.data.stats_fusion_preprocess import RawSessionInput
from bme_eating.protocol import CHECKPOINT_STATE_VERSION,config_identity,protocol_for,validate_config,verify_artifact_hashes
from bme_eating.streaming import JointDetector
from bme_eating.training.statsfusion_joint_trainer import seed_model
root=Path.cwd()
hashes=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
verify_artifact_hashes(root,hashes)
config=json.loads((root/'resolved_config.json').read_text(encoding='utf-8'))
validate_config(config)
point=json.loads((root/'inference.json').read_text(encoding='utf-8'))
payload=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
assert payload['protocol']==protocol_for(config) and payload['state_version']==CHECKPOINT_STATE_VERSION
assert payload['config_sha256']==config_identity(config)
detector=JointDetector()
detector.model=seed_model(config).eval()
detector.model.load_state_dict(payload['model'])
detector.config,detector.preprocessing=config,payload['preprocessing']
detector.point,detector.boundary,detector.identity=point['point'],point['boundary'],point['identity']
detector.mapping=FrozenSourceCDF(**point['mapping'])
with np.load(sys.argv[1],allow_pickle=False) as archive:
    raw=RawSessionInput(subject_key=point['subject'],session_id=point['session'],**dict(archive)).validated()
events=detector.predict_session(raw,new_steps=256)
print(json.dumps({'events':[{'id':item.event_id,'start_ms':item.start_ms,'end_ms':item.end_ms,'score':item.score} for item in events],'checkpoint_sha256':sha256_file(root/'checkpoint.pt')}))
"""
    environment = {**os.environ, "PYTHONPATH": str(bundle / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code, str(raw_path)], cwd=bundle,
        env=environment, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("Disposable real bundle replay failed: " + result.stderr)
    output = json.loads(result.stdout.strip().splitlines()[-1])
    expected_geometry = [(item.event_id, item.start_ms, item.end_ms) for item in expected]
    actual_geometry = [(item["id"], item["start_ms"], item["end_ms"]) for item in output["events"]]
    error = max((abs(left.score - right["score"]) for left, right in zip(expected, output["events"])), default=0)
    if expected_geometry != actual_geometry or error > 1e-6:
        raise RuntimeError("Disposable bundle differs from live real-session inference")
    return {"passed": True, "bundle": str(bundle), "checkpoint_sha256": output["checkpoint_sha256"],
        "score_max_error": error, "event_count": len(expected), "identity_alignment": True,
        "no_xgboost_load_and_replay": True, "real_input": True,
        "artifact_kind": "disposable_audit_bundle", "official_test_candidate": False,
        "promotable": False, "final39_contract_fixture_tested_separately": True}


def main():
    parser = argparse.ArgumentParser(description=f"Disposable {PROFILE.repair_revision} native replay and full search audit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--benchmark-search", action="store_true")
    parser.add_argument("--candidate-baseline-archive", type=Path)
    arguments = parser.parse_args()
    config = load_config(PROJECT_ROOT / "configs/statsfusion_v414_full_data.yaml")
    config["training"]["device"] = arguments.device
    require_idle_gpu(arguments.device)
    configure_numerics(config)
    _load_project_dotenv(config)
    output = require_environment_path(config, "output_root_env")
    inputs = JointInputs.audit(output / "v2", output / "v4")
    repair_tag = PROFILE.repair_revision.replace("-", "_")
    audit = json.loads((PROJECT_ROOT / f"manifests/{repair_tag}_production_loop_{arguments.device}.json").read_text(encoding="utf-8"))
    checkpoint = Path(audit["directory"]) / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (not audit["passed"] or payload["protocol"] != protocol_for(config)
            or payload["state_version"] != CHECKPOINT_STATE_VERSION
            or audit["source_sha256"] != digest(source_identity(PROJECT_ROOT))):
        raise RuntimeError("Native replay requires a qualified disposable audit checkpoint for the current revision")
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
    results, started, bundle_report = [], time.monotonic(), None
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
        if duration == 29.:
            bundle_report = audit_bundle_replay(detector, checkpoint, raw, outputs[-1])
    report = {"passed": True, "disposable_audit_weights": True, "formal_run_started": False,
        "synthetic_cdf_and_calibration_for_consistency_only": True, "label_use": "audit_selection_and_search_only",
        "checkpoint_sha256": sha256_file(checkpoint), "source_sha256": digest(source_identity(PROJECT_ROOT)),
        "data_sha256": digest(inputs.identity), "config_sha256": config_identity(config),
        "sessions": results, "elapsed_seconds": time.monotonic() - started,
        "disposable_real_bundle_replay": bundle_report}
    if arguments.benchmark_search:
        report["search_benchmark"] = benchmark_search(detector.model, inputs, config,
            payload["preprocessing"], audit, arguments.candidate_baseline_archive)
        report["passed"] &= report["search_benchmark"]["passed"]
    report["resources"] = enforce_resources(config, torch.device(arguments.device))
    write_json_atomic(PROJECT_ROOT / f"manifests/{repair_tag}_real_replay_{arguments.device}.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise RuntimeError("Native replay/search audit failed")


if __name__ == "__main__":
    main()
