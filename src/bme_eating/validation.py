from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from bme_eating.candidates import FrozenSourceCDF, TimelineCache
from bme_eating.data.deep_dataset import Normalization
from bme_eating.data.stats_fusion_preprocess import RawSessionInput, causal_completed_block_layout
from bme_eating.evaluation import NonnegativePlatt
from bme_eating.models.statsfusion_joint_event import quality_focal_loss
from bme_eating.protocol import (
    capabilities,
    config_identity,
    configure_numerics,
    digest,
    protocol_for,
    runtime_identity,
    source_identity,
)
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler
from bme_eating.streaming import JointChunkDetector, JointDetector


def synthetic_batch(config, steps=20, device="cpu"):
    generator = torch.Generator().manual_seed(2026)
    times = np.arange(1, steps + 1, dtype=np.int64) * 3000
    ppg_times, ends, mapping = causal_completed_block_layout(
        times, session_origin_ms=0, step_ms=3000, factor=5)
    ppg_count = len(ppg_times)
    values = {
        "motion_blocks": torch.randn(1, steps, 12, 300, generator=generator),
        "motion_invariant_blocks": torch.randn(1, steps, 6, 300, generator=generator),
        "motion_valid": torch.ones(1, steps),
        "ppg_blocks": torch.randn(1, ppg_count, 2, 750, generator=generator),
        "ppg_quality": torch.zeros(1, ppg_count, 8), "ppg_valid": torch.ones(1, ppg_count),
        "statistics": torch.cat((torch.randn(1, steps, 12, generator=generator),
                                  torch.zeros(1, steps, 12)), -1),
        "timestamp_ms": torch.from_numpy(times)[None],
        "timeline_position": torch.arange(steps)[None],
        "ppg_to_motion_index": torch.from_numpy(mapping)[None],
        "long_block_end_indices": torch.from_numpy(ends)[None],
        "supervision_mask": torch.ones(1, steps),
    }
    return {name: value.to(device) for name, value in values.items()}


def long_event_feature_replay(model, config, steps=4341):
    point = {"setting": ("high_0.1_low_0.9_gap_15"), "budget": 30, "acceptance": 0.0, "nms": .5,
             "calibration": NonnegativePlatt(0, 20, True).to_json(),
             "boundary_enabled": False, "boundary_entropy": None}
    mapping = FrozenSourceCDF({source: [0, 1] for source in
        ("state", "proposal", "jitter", "backtrack", "start_expansion")}, ["training"])
    times = torch.arange(1, steps + 22) * 3000
    features = torch.zeros(len(times), capabilities(config).feature_width)
    features[:, 128] = 1
    features[:, 135] = 1
    features[:steps, 129:133] = .95
    outputs = []
    for chunk in (17, 256, len(times)):
        detector = JointChunkDetector(model, config, point, mapping, "synthetic", {"qualified": False},
                                      "subject", "session")
        emitted = []
        for first in range(0, len(times), chunk):
            stop = min(first + chunk, len(times))
            emitted.extend(detector.push_features(times[first:stop], features[first:stop], final=stop == len(times)))
        outputs.append(emitted)
    identities = [[event["proposal_id"] for event in output] for output in outputs]
    if identities[0] != identities[1] or identities[1] != identities[2] or not identities[0]:
        raise RuntimeError("Long event replay chunk invariance failed")
    for left, right in zip(outputs[0], outputs[1]):
        for name in ("event_probability", "predicted_quality", "rank_score"):
            if abs(left[name] - right[name]) > 1e-6:
                raise RuntimeError("Long replay determinism failed")
    return {"event_duration_seconds": steps * 3, "chunks": [17, 256, len(times)],
            "event_count": len(outputs[0]), "identities": identities[0],
            "max_latency_seconds": max((event["decision_time_ms"] - event["end_ms"]) / 1000 for event in outputs[0])}


def validate_joint_implementation(config, *, device="cpu"):
    started = time.monotonic()
    configure_numerics(config)
    torch.manual_seed(2026)
    from bme_eating.models.statsfusion_joint_event import build_joint_model
    model = build_joint_model(config).to(device).eval()
    from bme_eating.data.joint_sessions import joint_geometry
    from bme_eating.models.statsfusion_joint_event import indexed_roi as roi_function
    batch = synthetic_batch(config, steps=joint_geometry(config).total_steps if device.startswith("cuda") else 20,
                            device=device)
    outputs = model(batch)
    if not all(torch.isfinite(outputs[name]).all() for name in ("semantic_features", "local_features")):
        raise RuntimeError("Shared encoder finite gate failed")
    times = batch["timestamp_ms"][0]
    roi = roi_function(outputs["semantic_features"][0], outputs.get("localization_features", outputs["local_features"])[0], times,
                         torch.ones_like(times).float(), 9000, 45_000, 60_000)
    scores = model.candidate_head(roi[None], torch.zeros(1, 12, device=device))
    for name, loss in (("event", torch.nn.functional.binary_cross_entropy_with_logits(
        scores["event_logit"], torch.ones(1, device=device))),
        ("quality", quality_focal_loss(scores["quality_logit"], torch.full((1,), .8, device=device)).mean())):
        gradient = torch.autograd.grad(loss, model.encoder.short_fusion[0].weight, retain_graph=True)[0]
        if not torch.isfinite(gradient).all() or not bool((gradient.abs() > 0).any()):
            raise RuntimeError(f"{name} loss does not update the shared encoder")
    (scores["event_logit"].sum() + scores["quality_logit"].sum()).backward()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        repeated = model(batch)
    difference = float((outputs["semantic_features"] - repeated["semantic_features"]).abs().max())
    if difference > 1e-6:
        raise RuntimeError("Checkpoint repeat inference differs")
    with tempfile.TemporaryDirectory(prefix="bme_joint_cache_") as temporary:
        binding = {"epoch": 1, "subject": "synthetic", "session": "long", "model": "fixed"}
        cache = TimelineCache(Path(temporary), binding, capabilities(config).feature_width)
        values = np.zeros((4341, capabilities(config).feature_width), dtype=np.float32)
        for first in range(0, 4341, 256):
            stop = min(4341, first + 256)
            cache.append(np.arange(first, stop, dtype=np.int64) * 3000, values[first:stop])
        cached_times, cached_values = cache.tensors()
        if cached_values.requires_grad or len(cached_times) != 4341:
            raise RuntimeError("Detached history cache failed")
        del cached_times, cached_values
    report = {"shared_event_quality_gradients": True, "repeat_max_error": difference,
              "long_event": long_event_feature_replay(model, config), "device": device,
              "raw_session": raw_session_replay(model, config),
              "gpu_validation_performed": device.startswith("cuda")}
    report["localization_receptive_field_seconds"] = model.localization_receptive_field_seconds
    report["localization_dependency_seconds"] = model.localization_dependency_seconds
    with torch.no_grad():
        batched_roi = roi.detach()[None].expand(128, -1, -1).contiguous()
        metadata = torch.zeros(128, 12, device=device)
        single = model.candidate_head(batched_roi[:1], metadata[:1])
        parallel = model.candidate_head(batched_roi, metadata)
        head_error = max(float((single[name] - parallel[name][:1]).abs().max())
                         for name in ("event_logit", "quality_logit", "candidate_embedding"))
    if head_error > 1e-6:
        raise RuntimeError("v411 batch128 equivalence gate failed; optimization cannot be enabled")
    report["candidate_batch128_max_error"] = head_error
    report["coordinate_selection_discrete"] = True
    report["history_gradient_detached"] = True
    if device.startswith("cuda"):
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mixed = model(batch)
            mixed_roi = roi_function(mixed["semantic_features"][0], mixed.get("localization_features", mixed["local_features"])[0], times,
                                      torch.ones_like(times).float(), 9000, 45_000, 60_000)
            mixed_scores = model.candidate_head(mixed_roi[None], torch.zeros(1, 12, device=device))
            mixed_losses = {
                "event": torch.nn.functional.binary_cross_entropy_with_logits(
                    mixed_scores["event_logit"].float(), torch.ones(1, device=device)),
                "quality": quality_focal_loss(mixed_scores["quality_logit"].float(),
                                               torch.full((1,), .8, device=device)).mean()}
        norms = {}
        for name, loss in mixed_losses.items():
            gradient = torch.autograd.grad(loss, model.encoder.short_fusion[0].weight, retain_graph=True)[0]
            if not torch.isfinite(gradient).all() or not (gradient.abs() > 0).any():
                raise RuntimeError(f"bfloat16 {name} loss does not update the shared encoder")
            norms[name] = float(gradient.float().norm())
        sum(mixed_losses.values()).backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise RuntimeError("bfloat16 joint backward produced nonfinite gradients")
        model.zero_grad(set_to_none=True)
        model.eval()
        report["bfloat16_shared_gradient_norms"] = norms
        report["bfloat16_backward_passed"] = True
        from bme_eating.training.statsfusion_joint_trainer import enforce_resources
        report.update(enforce_resources(config, torch.device(device)))
    report["elapsed_seconds"] = time.monotonic() - started
    report.update({"protocol": protocol_for(config), "config_sha256": config_identity(config),
        "source_sha256": digest(source_identity(Path(config["_config_path"]).resolve().parents[1])),
        "runtime": runtime_identity(), "synthetic_validation": True, "formal_five_fold_training": False})
    return report


def raw_session_replay(model, config):
    from bme_eating.streaming import JointRawStream, causal_raw_batches
    from bme_eating.training.statsfusion_joint_trainer import pack_features
    rng = np.random.default_rng(2026)
    motion_time = np.arange(0, 60_001, 10, dtype=np.int64)
    ppg_time = np.arange(0, 60_001, 20, dtype=np.int64)
    raw = RawSessionInput(subject_key="synthetic", session_id="raw", motion_timestamp_ms=motion_time,
        motion_values=rng.normal(size=(len(motion_time), 6)).astype(np.float32),
        motion_mask=np.ones((len(motion_time), 6), bool), ppg_timestamp_ms=ppg_time,
        ppg_values=rng.normal(size=len(ppg_time)).astype(np.float32), ppg_mask=np.ones(len(ppg_time), bool))
    scaler = FoldRobustScaler(STATS_FEATURE_COLUMNS, np.zeros(12), np.ones(12), ("fit",))
    normalization = Normalization(np.zeros(6), np.ones(6), 0, 1)
    preprocessing = {"scaler": scaler.to_json(), "normalization": normalization.to_json()}
    results = []
    event_results = []
    model.eval()
    for chunk in (10, 20):
        times, hidden = [], []
        point = {"setting": ("high_0.1_low_0.9_gap_15"), "budget": 30, "acceptance": 0, "nms": .5,
                 "calibration": NonnegativePlatt(0, 20, True).to_json(), "boundary_enabled": False}
        mapping = FrozenSourceCDF({name: [0, 1] for name in
            ("state", "proposal", "jitter", "backtrack", "start_expansion")}, ["fit"])
        detector = JointChunkDetector(model, config, point, mapping, "raw_smoke", {"qualified": False},
                                      "synthetic", "raw")
        events = []
        for batch, final in causal_raw_batches(raw, preprocessing, config, chunk):
            batch = {name: value.to(next(model.parameters()).device) for name, value in batch.items()}
            with torch.no_grad():
                outputs = model(batch)
            mask = batch["supervision_mask"][0] > 0
            times.extend(batch["timestamp_ms"][0, mask].cpu().tolist())
            hidden.append(outputs["semantic_features"][0, mask].detach().cpu())
            events.extend(detector.push_features(batch["timestamp_ms"][0, mask], pack_features(outputs, mask),
                                                 final=final))
        results.append((times, torch.cat(hidden)))
        event_results.append(events)
    difference = float((results[0][1] - results[1][1]).abs().max())
    if results[0][0] != results[1][0] or difference > 1e-5:
        raise RuntimeError(f"Native raw replay chunk alignment/determinism failed: {difference}")
    if [event["proposal_id"] for event in event_results[0]] != [event["proposal_id"] for event in event_results[1]]:
        raise RuntimeError("Native raw replay changes committed event identities")
    score_difference = max((abs(left["rank_score"] - right["rank_score"]) for left, right in
                            zip(*event_results)), default=0)
    if score_difference > 1e-6:
        raise RuntimeError("Native replay event score reproducibility failed")
    incremental = JointRawStream(JointChunkDetector(model, config, point, mapping, "raw_smoke",
                                {"qualified": False}, "synthetic", "raw"), preprocessing, config)
    previous = -1
    incremental_events = []
    for cutoff in (30_000, 60_000):
        arrays = {}
        for modality in ("motion", "ppg"):
            timestamps = getattr(raw, modality + "_timestamp_ms")
            selected = (timestamps > previous) & (timestamps <= cutoff)
            for suffix in ("timestamp_ms", "values", "mask"):
                name = modality + "_" + suffix
                arrays[name] = getattr(raw, name)[selected]
        piece = RawSessionInput(subject_key="synthetic", session_id="raw", **arrays)
        incremental_events.extend(incremental.push(piece, final=cutoff == 60_000))
        previous = cutoff
    if [event.event_id for event in incremental_events] != [event["proposal_id"] for event in event_results[0]]:
        raise RuntimeError("Native incremental raw replay event identity differs")
    return {"native_motion_samples": len(motion_time), "native_ppg_samples": len(ppg_time),
            "new_chunk_steps": [10, 20], "feature_max_error": difference,
            "event_score_max_error": score_difference, "event_identity_alignment": True,
            "timestamp_alignment": True, "incremental_native_chunks": True,
            "public_event_schema": True}


def replay_joint_bundle(bundle: Path, config, *, device="cpu"):
    from bme_eating.bundle import StatsFusionJointFinalDataDetector
    detector_type = StatsFusionJointFinalDataDetector
    detector = detector_type(bundle, device=device)
    report = long_event_feature_replay(detector.model, config)
    locked = replay_locked_bundle(detector)
    code = """import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'xgboost':
        raise ImportError('xgboost forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from bme_eating.bundle import StatsFusionJointFinalDataDetector as StatsFusionJointDetector
from bme_eating.validation import replay_locked_bundle
replay_locked_bundle(StatsFusionJointDetector(sys.argv[1]))
print('PASS')
"""
    command = [sys.executable, "-c", code, str(bundle)]
    import os
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(bundle / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=environment,
                            cwd=bundle)
    if result.returncode:
        raise RuntimeError("No-XGBoost bundle load failed: " + result.stderr)
    return {**report, "device": device, "gpu_validation_performed": device == "cuda",
            "locked_native_replay": locked, "no_xgboost_load": True,
            "no_xgboost_replay": True, "bundle_sha256": json.loads(
        (bundle / "SHA256SUMS.json").read_text(encoding="utf-8"))}


def replay_locked_bundle(detector: JointDetector):
    rng = np.random.default_rng(2026)
    motion_time = np.arange(0, 30_001, 10, dtype=np.int64)
    ppg_time = np.arange(0, 30_001, 20, dtype=np.int64)
    raw = RawSessionInput(subject_key="bundle_replay", session_id="session",
        motion_timestamp_ms=motion_time, motion_values=rng.normal(size=(len(motion_time), 6)).astype(np.float32),
        motion_mask=np.ones((len(motion_time), 6), bool), ppg_timestamp_ms=ppg_time,
        ppg_values=rng.normal(size=len(ppg_time)).astype(np.float32), ppg_mask=np.ones(len(ppg_time), bool))
    full = detector.predict_session(raw, new_steps=10)
    repeated = detector.predict_session(raw, new_steps=10)
    small = detector.predict_session(raw, new_steps=5)
    def identity(events):
        return [event.event_id for event in events]
    if identity(full) != identity(repeated) or identity(full) != identity(small):
        raise RuntimeError("Locked bundle raw replay event identities changed")
    difference = max((abs(left.score - right.score) for left, right in zip(full, repeated)), default=0)
    if difference > 1e-6:
        raise RuntimeError("Locked bundle repeat inference exceeds 1e-6")
    incremental = detector.new_raw_session("bundle_replay", "session")
    previous, streamed = -1, []
    for cutoff in (15_000, 30_000):
        arrays = {}
        for modality in ("motion", "ppg"):
            times = getattr(raw, modality + "_timestamp_ms")
            selected = (times > previous) & (times <= cutoff)
            for suffix in ("timestamp_ms", "values", "mask"):
                name = modality + "_" + suffix
                arrays[name] = getattr(raw, name)[selected]
        streamed.extend(incremental.push(RawSessionInput(subject_key="bundle_replay", session_id="session",
                                                         **arrays), final=cutoff == 30_000))
        previous = cutoff
    if identity(streamed) != identity(full):
        raise RuntimeError("Locked native incremental bundle replay differs")
    return {"uses_locked_operating_point": True, "public_event_schema": True,
            "incremental_identity_alignment": True, "repeat_max_error": difference,
            "event_count": len(full), "synthetic_input": True}
