from __future__ import annotations

import copy
import json
import math
import random
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from bme_eating.data.joint_sessions import (
    JointInputs,
    SessionChunkSampler,
    make_joint_dataset,
    session_loader,
)
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_evaluation import (
    NonnegativePlatt,
    apply_online_budget,
    calibration_ece,
    candidate_coverage,
    choose_operating_point,
    commit_candidates,
    metrics_with_exposure,
)
from bme_eating.joint_protocol import (
    PROTOCOL,
    assert_excluded,
    checkpoint_qualified,
    config_identity,
    configure_numerics,
    digest,
    protocol_identities,
    save_torch_atomic,
    source_identity,
    verify_internal_manifest,
    verify_oof_lineage,
)
from bme_eating.joint_stream import (
    FrozenSourceCDF,
    JointCandidate,
    StreamingProposals,
    TimelineCache,
    candidate_targets,
    sample_candidates,
    stable_id,
)
from bme_eating.metrics import partition_evaluation_events, prediction_ignore_mask
from bme_eating.models.statsfusion_joint_event import (
    StatsFusionJointEvent,
    candidate_roi,
    distribution_endpoint,
    event_contrastive_loss,
    gaussian_distribution,
    loss_schedule,
    quality_focal_loss,
)
from bme_eating.training.hierarchical_v4_trainer import _selector_split, stacking_partitions

FEATURE_WIDTH = 137


def seed_model(config):
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    configure_numerics(config)
    return StatsFusionJointEvent(config)


def resource_usage(device):
    import ctypes
    import os
    peak_ram = None
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("page_faults", ctypes.c_ulong),
                        ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                        ("quota_peak_paged", ctypes.c_size_t), ("quota_paged", ctypes.c_size_t),
                        ("quota_peak_nonpaged", ctypes.c_size_t), ("quota_nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
        counters = ProcessMemoryCounters()
        counters.size = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
            peak_ram = counters.peak_working_set / 1024 ** 3
    peak_gpu = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0
    return {"peak_process_ram_gb": peak_ram, "peak_gpu_gb": peak_gpu,
            "peak_gpu_reserved_gb": peak_reserved}


def enforce_resources(config, device):
    usage = resource_usage(device)
    if max(usage["peak_gpu_gb"], usage["peak_gpu_reserved_gb"]) >= config["resources"]["maximum_gpu_gb"]:
        raise RuntimeError("GPU memory gate failed; protocol cannot silently shrink batches")
    return usage


def pack_features(outputs, mask):
    scalar = [outputs["sensor_valid"], outputs["state_logit"].sigmoid(),
              *(outputs["proposal_scales"][str(scale)][0].sigmoid()
                * (outputs["proposal_scales"][str(scale)][1] > 0) for scale in (3, 15, 60)),
              outputs["onset_logit"].sigmoid(), outputs["offset_logit"].sigmoid(),
              outputs["gyro_valid_fraction"], outputs["missing_fraction"]]
    return torch.cat((outputs["semantic_features"][0, mask], outputs["local_features"][0, mask],
                      torch.stack([value[0, mask] for value in scalar], -1)), -1).float()


def generate_chunk(stream: StreamingProposals, times: torch.Tensor, features: torch.Tensor):
    candidates = []
    for timestamp, values in zip(times.tolist(), features.detach().cpu().numpy()):
        probabilities = {"state": float(values[129]),
                         **{f"proposal_{scale}": float(values[130 + index])
                            for index, scale in enumerate((3, 15, 60))}}
        candidates.extend(stream.observe(int(timestamp), probabilities, float(values[128]),
                                          float(values[134])))
    return candidates


def score_candidate(model, candidate, times, features, device):
    causal = times <= candidate.decision_time_ms
    selected_times, selected_features = times[causal], features[causal]
    needed = (selected_times > candidate.start_ms - 60_000) & (
        selected_times <= min(candidate.end_ms + 30_000, candidate.decision_time_ms))
    selected_times, selected_features = selected_times[needed], selected_features[needed]
    if len(selected_features):
        candidate.missing_fraction = float(selected_features[:, 136].mean().detach())
        candidate.gyro_missing = bool((selected_features[:, 135] <= 0).all())
    roi = candidate_roi(selected_features[:, :64], selected_features[:, 64:128], selected_times,
                         selected_features[:, 128], candidate.start_ms, candidate.end_ms,
                         candidate.decision_time_ms)
    outputs = model.candidate_head(roi[None].to(device), candidate.metadata(device)[None])
    candidate.event_logit = float(outputs["event_logit"].detach())
    candidate.predicted_quality = float(outputs["quality_logit"].sigmoid().detach())
    return outputs


def refine_candidate(model, candidate, times, features, device, boundary, entropy):
    refined = copy.deepcopy(candidate)
    refined.unrefined_start_ms, refined.unrefined_end_ms = candidate.start_ms, candidate.end_ms
    refined.coarse_event_logit, refined.coarse_quality = candidate.event_logit, candidate.predicted_quality
    def fallback(reason):
        refined.start_ms, refined.end_ms = candidate.start_ms, candidate.end_ms
        refined.refined = False
        refined.boundary_fallback_reason = reason
        refined.boundary_conflict = reason in {"conflict_or_latency", "insufficient_valid_bins"}
        return refined
    if not boundary.get("qualified") or candidate.source == "truth_jitter":
        return fallback("training_qualification_failed")
    for endpoint, range_seconds, head_index in (("start_ms", boundary["start_range_seconds"], 0),
                                               ("end_ms", boundary.get("end_range_seconds", 30), 1)):
        center = getattr(candidate, endpoint)
        selected = ((times - center).abs() <= range_seconds * 1000) & (times <= candidate.decision_time_ms)
        selected &= features[:, 128] > 0
        setattr(refined, "boundary_valid_" + ("start" if head_index == 0 else "end") + "_bins", int(selected.sum()))
        if int(selected.sum()) < 3:
            return fallback("insufficient_valid_bins")
        local, support = features[selected, 64:128].to(device), times[selected].to(device)
        logits = model.boundary_head(local, (support - center).float() / 1000)[:, head_index]
        proposal, uncertainty = distribution_endpoint(logits, support)
        setattr(refined, "boundary_" + ("start" if head_index == 0 else "end") + "_entropy", uncertainty)
        if uncertainty > entropy:
            return fallback("entropy")
        refined.boundary_clipped |= abs(proposal - center) >= range_seconds * 1000 - 3000
        setattr(refined, endpoint, proposal)
    if (refined.start_ms >= refined.end_ms or refined.end_ms > candidate.decision_time_ms
            or candidate.decision_time_ms - refined.end_ms > 60_000):
        return fallback("conflict_or_latency")
    refined.refined = (refined.start_ms != candidate.start_ms or refined.end_ms != candidate.end_ms)
    score_candidate(model, refined, times, features, device)
    return refined


def boundary_distribution_loss(model, candidate, target, times, features, device, qualification):
    positive, _, _, true_start, true_end = target
    if not positive or not qualification.get("qualified") or true_end > candidate.decision_time_ms:
        return (features.sum() * 0).to(device)
    losses = []
    for center, truth_endpoint, maximum, column in (
        (candidate.start_ms, true_start, qualification["start_range_seconds"], 0),
        (candidate.end_ms, true_end, qualification.get("end_range_seconds", 30), 1)):
        selected = ((times - center).abs() <= maximum * 1000) & (times <= candidate.decision_time_ms)
        selected &= features[:, 128] > 0
        if selected.sum() < 3 or abs(truth_endpoint - center) > maximum * 1000:
            continue
        support, local = times[selected].to(device), features[selected, 64:128].to(device)
        logits = model.boundary_head(local, (support - center).float() / 1000)[:, column]
        target_distribution = gaussian_distribution(support, truth_endpoint,
                                                    qualification.get("sigma_seconds", 9))
        losses.append(-(target_distribution * logits.float().log_softmax(0)).sum())
    return torch.stack(losses).mean() if losses else (features.sum() * 0).to(device)


def truth_jitter_candidates(truth, subject, session, time_ms, previous_time, fraction, identity, rng):
    candidates = []
    if fraction <= 0:
        return candidates
    completed = truth[(truth.subject_key.astype(str) == subject)
                      & (truth.session_id.astype(str) == session)
                      & (truth.end_ms + 30_000 <= time_ms)
                      & (truth.end_ms + 30_000 > previous_time)]
    for event in completed.itertuples(index=False):
        end = int(event.end_ms) + int(rng.choice([-15, 0, 15])) * 1000
        decision = int(event.end_ms) + 30_000
        start = int(event.start_ms) + int(rng.choice([-30, -15, 0, 15, 30])) * 1000
        if start >= end:
            continue
        family = stable_id(subject, session, event.start_ms, event.end_ms, identity)
        candidates.append(JointCandidate(subject, session, start, end, decision, "truth_jitter", 3, 1,
            int(event.start_ms), int(event.end_ms), family, stable_id(family, start, end), family,
            stable_id(event.start_ms, event.end_ms), identity, {"training_only": True}, decision))
    return candidates


def random_background_candidates(truth, ignore, subject, session, times, features, identity, rng,
                                 maximum=2):
    if not len(times):
        return []
    decision = int(times[-1])
    candidates, seen = [], set()
    for _ in range(16):
        end = decision - int(rng.integers(0, 11)) * 3000
        duration = int(rng.choice([15, 30, 60, 120, 300])) * 1000
        start = end - duration
        if start < int(times[0]) - 3000 or (start, end) in seen:
            continue
        seen.add((start, end))
        support = (times > start) & (times <= end)
        valid = features[support, 128] > 0
        if not len(valid) or float(valid.float().mean()) < .5:
            continue
        family = stable_id(identity, subject, session, "random_background", start, end)
        raw_score = float(features[support, 129][valid].mean().detach())
        parent_hash = digest({"subject": subject, "session": session, "model": identity,
                              "sampling": "random_background", "start": start, "end": end})
        candidate = JointCandidate(subject, session, start, end, decision, "state", 3, raw_score,
            start, end, family, stable_id(family, "proposal"), family, parent_hash, identity,
            {"sampling": "random_background", "training_only": True}, decision)
        if candidate_targets(candidate, truth)[1] > 0:
            continue
        if len(ignore):
            from bme_eating.joint_evaluation import event_frame
            if prediction_ignore_mask(event_frame([candidate]), ignore).any():
                continue
        candidate.validate()
        candidates.append(candidate)
        if len(candidates) >= maximum:
            break
    return candidates


def gradient_diagnostics(tasks, model):
    parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    vectors, report = {}, {}
    for name, loss in tasks.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vector = torch.cat([gradient.detach().float().flatten() if gradient is not None
                            else parameter.new_zeros(parameter.numel()).float()
                            for parameter, gradient in zip(parameters, gradients)])
        vectors[name] = vector
        report[name + "_weighted_norm"] = float(vector.norm())
    for left, right in product(vectors, repeat=2):
        if left >= right:
            continue
        denominator = vectors[left].norm() * vectors[right].norm()
        cosine = (vectors[left] @ vectors[right] / denominator.clamp_min(1e-12)).clamp(-1, 1)
        report[left + "_" + right + "_angle_degrees"] = float(torch.acos(cosine) * 180 / math.pi)
    return report


def train_joint_model(inputs: JointInputs, subjects: set[str], config: dict[str, Any],
                      directory: Path, identity: str, *, epochs: int, resume: bool,
                      selector: set[str] | None = None, outer_excluded: set[str] | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    device = torch.device(training["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required by protocol")
        torch.cuda.reset_peak_memory_stats(device)
    model = seed_model(config).to(device)
    source_root = Path(config["_config_path"]).resolve().parents[1]
    bound_source = source_identity(source_root)
    bound_config = config_identity(config)
    def source_signature():
        paths = list((source_root / "src" / "bme_eating").rglob("*.py")) + list((source_root / "scripts").glob("*.py"))
        return {path.relative_to(source_root).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
                for path in paths}
    bound_signature = source_signature()
    input_signature = inputs.file_signature() if hasattr(inputs, "file_signature") else None
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": training["encoder_learning_rate"]},
        {"params": [parameter for name, parameter in model.named_parameters()
                    if not name.startswith("encoder.")], "lr": training["head_learning_rate"]}],
        weight_decay=training["weight_decay"])
    dataset, preprocessing = make_joint_dataset(inputs, subjects, config, training=True)
    write_json_atomic(directory / "preprocessing.json", preprocessing)
    checkpoint_path = directory / "resume.pt"
    state = {"epoch": 1, "next_chunk": 0, "session_count": 0, "update_count": 0,
             "clipped_steps": 0, "optimizer_steps": 0, "epoch_complete": False,
             "boundary": {"qualified": False, "start_range_seconds": 900},
             "cache_rows": 0, "session_key": None, "stream": None,
             "history": [], "best_epoch": None, "best_f1": -1.0, "stale_checks": 0,
             "diagnostic_epoch": None, "diagnostic_f1": -1.0,
             "event_contributions": set(), "family_contributions": set(),
             "epoch_candidates": [], "gradient_diagnostics": [], "sample_audit": [],
             "training_complete": False}
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["identity"] != identity or set(payload["training_subjects"]) != subjects:
            raise RuntimeError("Training checkpoint identity or subject lineage mismatch")
        if (payload.get("requested_epochs") != epochs
                or set(payload.get("selector_subjects", [])) != (selector or set())
                or set(payload.get("excluded_subjects", [])) != (outer_excluded or set()) | (selector or set())):
            raise RuntimeError("Training resume epoch/selector/exclusion protocol changed")
        if (payload.get("config_sha256") != bound_config
                or payload.get("source_sha256") != digest(bound_source)
                or payload.get("data_sha256") != digest(inputs.identity)):
            raise RuntimeError("Training checkpoint config/source/data identity mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for name, parameter in model.named_parameters():
            gradient = payload["gradients"].get(name)
            parameter.grad = gradient.to(device) if gradient is not None else None
        state = payload["state"]
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])
        if state.get("training_complete"):
            if selector and not (state["best_epoch"] or state["diagnostic_epoch"]):
                raise RuntimeError("No finite/clipping-qualified joint epoch; diagnostic checkpoints preserved")
            return (state["best_epoch"] or state["diagnostic_epoch"]) if selector else epochs
    truth, _ = partition_evaluation_events(dataset.events, subjects)
    session_lengths = {key: len(group) for key, group in dataset.session_groups.items()}
    session_events = {key: max(1, int(((truth.subject_key.astype(str) == key[0])
                        & (truth.session_id.astype(str) == key[1])).sum())) for key in session_lengths}
    started = time.monotonic()

    def checkpoint():
        if config_identity(config) != bound_config:
            raise RuntimeError("Resolved training configuration changed before checkpoint")
        if source_signature() != bound_signature and source_identity(source_root) != bound_source:
            raise RuntimeError("Runtime source changed before checkpoint")
        if input_signature is not None and inputs.file_signature() != input_signature:
            raise RuntimeError("Runtime input data changed before checkpoint")
        finite = all(torch.isfinite(value).all() for value in model.state_dict().values())
        if not finite or any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                             for parameter in model.parameters()):
            raise RuntimeError("Finite gate failed before chunk checkpoint")
        save_torch_atomic(checkpoint_path, {"identity": identity, "training_subjects": sorted(subjects),
            "requested_epochs": epochs, "selector_subjects": sorted(selector or set()),
            "excluded_subjects": sorted((outer_excluded or set()) | (selector or set())),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
            "gradients": {name: parameter.grad.detach().cpu() if parameter.grad is not None else None
                          for name, parameter in model.named_parameters()},
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "config_sha256": bound_config, "source_sha256": digest(bound_source),
            "data_sha256": digest(inputs.identity), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})

    while state["epoch"] <= epochs:
        epoch = state["epoch"]
        sampler = SessionChunkSampler(dataset, epoch, state["next_chunk"])
        if state.get("sampler_identity") and state["sampler_identity"] != sampler.identity:
            raise RuntimeError("Resume session order changed")
        state["sampler_identity"] = sampler.identity
        model.train()
        loader = session_loader(dataset, sampler, config)
        for chunk_index, batch in enumerate(loader, start=sampler.start_chunk):
            _, _, _, key, last, new_steps = sampler.chunks[chunk_index]
            if hasattr(inputs, "assert_session_unchanged"):
                inputs.assert_session_unchanged(key)
            if state["session_key"] != key:
                if state["session_key"] is not None:
                    raise RuntimeError("Session changed before completion")
                state["session_key"] = key
                state["cache_rows"] = 0
                state["stream"] = StreamingProposals(*key, identity, config["decoder"])
            cache = TimelineCache(directory / "cache" / f"epoch_{epoch}" / stable_id(*key),
                {"identity": identity, "epoch": epoch, "update": state["update_count"],
                 "subject": key[0], "session": key[1]}, FEATURE_WIDTH, state["cache_rows"])
            history_times, history_features = cache.tensors()
            batch = {name: value.to(device) if isinstance(value, torch.Tensor) else value
                     for name, value in batch.items()}
            mask = torch.zeros_like(batch["supervision_mask"], dtype=torch.bool)
            mask[:, -new_steps:] = True
            current_times = batch["timestamp_ms"][0, mask[0]]
            schedule = loss_schedule(epoch)
            rng = np.random.default_rng(2026 + epoch * 1_000_000 + chunk_index)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                outputs = model(batch)
                features = pack_features(outputs, mask[0])
                times = torch.cat((history_times, current_times.cpu()))
                full_features = torch.cat((history_features, features.cpu()))
                candidates = generate_chunk(state["stream"], current_times, features)
                if last:
                    candidates.extend(state["stream"].finish())
                state["epoch_candidates"].extend(copy.deepcopy(candidates))
                previous = int(history_times[-1]) if len(history_times) else int(current_times[0]) - 3000
                candidates.extend(truth_jitter_candidates(truth, *key, int(current_times[-1]),
                    previous, schedule["truth_jitter"], identity, rng))
                candidates.extend(random_background_candidates(truth, dataset.ignore_events, *key,
                    times, full_features, identity, rng))
                session_segments = dataset.segments[
                    (dataset.segments.subject_key.astype(str) == key[0])
                    & (dataset.segments.session_id.astype(str) == key[1])] if hasattr(dataset, "segments") else pd.DataFrame()
                session_truth = truth[(truth.subject_key.astype(str) == key[0]) & (truth.session_id.astype(str) == key[1])]
                for candidate in candidates:
                    causal_support = (times > candidate.start_ms - 60_000) & (times <= candidate.decision_time_ms)
                    support = full_features[causal_support]
                    if len(support):
                        candidate.gyro_missing = bool((support[:, 135] <= 0).all())
                        candidate.missing_fraction = float(support[:, 136].mean().detach())
                    if "wear_hand" in session_segments and len(session_segments):
                        candidate.wear_hand = str(session_segments.wear_hand.iloc[0])
                    if "hand_relation" in session_truth:
                        relations = set(session_truth.hand_relation.fillna("unknown").astype(str))
                        candidate.hand_relation = next(iter(relations)) if len(relations) == 1 else "unknown"
                ignore = dataset.ignore_events
                if len(candidates) and len(ignore):
                    from bme_eating.joint_evaluation import event_frame
                    ignored = prediction_ignore_mask(event_frame(candidates), ignore)
                    candidates = [candidate for candidate, drop in zip(candidates, ignored)
                                  if not drop or candidate_targets(candidate, truth)[0] > 0]
                candidates, sample_report = sample_candidates(candidates, truth,
                    training["candidates_per_chunk"], training["sample_composition"], rng,
                    truth_jitter_fraction=schedule["truth_jitter"])
                state["sample_audit"].append({"epoch": epoch, "chunk": chunk_index,
                    "subject": key[0], "session": key[1], **sample_report})
                target = batch["state_target"].float()
                valid = mask.float() * batch["state_loss_mask"] * (outputs["sensor_valid"] > 0)
                def masked_bce(logit, label, weight):
                    return (F.binary_cross_entropy_with_logits(logit.float(), label, reduction="none")
                            * weight).sum() / weight.sum().clamp_min(1)
                tasks = {"state": masked_bce(outputs["state_logit"], target, valid)}
                proposals = []
                for scale, (logits, scale_valid) in outputs["proposal_scales"].items():
                    factor = int(scale) // 3
                    scale_target = torch.zeros_like(target)
                    scale_mask = torch.zeros_like(valid)
                    positions = batch["timeline_position"]
                    for endpoint in range(factor - 1, target.shape[1]):
                        if int((positions[0, endpoint] + 1) % factor) != 0:
                            continue
                        members = slice(endpoint - factor + 1, endpoint + 1)
                        labels_valid = batch["state_loss_mask"][:, members]
                        scale_target[:, endpoint] = (target[:, members] * labels_valid).sum(1) / labels_valid.sum(1).clamp_min(1)
                        scale_mask[:, endpoint] = valid[:, endpoint] * (scale_valid[:, endpoint] > 0) * (labels_valid.sum(1) > 0)
                    proposals.append(masked_bce(logits, scale_target, scale_mask))
                tasks["proposal"] = torch.stack(proposals).mean()
                tasks["transitions"] = sum(masked_bce(outputs[name + "_logit"],
                    batch[name + "_target"], mask.float() * batch[name + "_loss_mask"])
                    for name in ("onset", "offset"))
                smooth_mask = valid[:, 1:] * valid[:, :-1] * batch["smooth_mask"][:, 1:] * batch["smooth_loss_mask"][:, 1:]
                tasks["smooth"] = ((outputs["state_logit"][:, 1:].sigmoid()
                    - outputs["state_logit"][:, :-1].sigmoid()).square() * smooth_mask).sum() / smooth_mask.sum().clamp_min(1)
                event_losses, quality_losses, boundary_losses = [], [], []
                negative_event_losses, negative_quality_losses = [], []
                for candidate in candidates:
                    target_data = candidate_targets(candidate, truth)
                    positive, quality, event_id, *_ = target_data
                    contribution = (key, event_id) if positive else (key, candidate.proposal_family_id)
                    bucket = state["event_contributions"] if positive else state["family_contributions"]
                    if contribution in bucket:
                        continue
                    bucket.add(contribution)
                    score = score_candidate(model, candidate, times, full_features, device)
                    event_loss = F.binary_cross_entropy_with_logits(score["event_logit"].float(),
                                         torch.tensor([positive], device=device))
                    quality_loss = quality_focal_loss(score["quality_logit"].float(),
                        torch.tensor([quality], device=device), config["loss"]["quality_beta"]).mean()
                    if schedule["boundary"] > 0 and state["boundary"].get("qualified"):
                        with torch.no_grad():
                            refined_candidate = refine_candidate(model, candidate, times, full_features,
                                device, state["boundary"], 0.65)
                        if refined_candidate.refined:
                            refined_positive, refined_quality, *_ = candidate_targets(refined_candidate, truth)
                            refined_score = score_candidate(model, refined_candidate, times, full_features, device)
                            event_loss = (event_loss + F.binary_cross_entropy_with_logits(
                                refined_score["event_logit"].float(), torch.tensor([refined_positive], device=device))) / 2
                            quality_loss = (quality_loss + quality_focal_loss(refined_score["quality_logit"].float(),
                                torch.tensor([refined_quality], device=device), config["loss"]["quality_beta"]).mean()) / 2
                            state["sample_audit"][-1].setdefault("refined_label_changes", []).append({
                                "proposal_id": candidate.proposal_id, "coarse_target": positive,
                                "refined_target": refined_positive, "coarse_quality": quality,
                                "refined_quality": refined_quality})
                    (event_losses if positive else negative_event_losses).append(event_loss)
                    (quality_losses if positive else negative_quality_losses).append(quality_loss)
                    boundary_losses.append(boundary_distribution_loss(model, candidate, target_data,
                        times, full_features, device, state["boundary"]))
                zero = features.sum() * 0
                tasks["event"] = torch.stack(event_losses).sum() / session_events[key] if event_losses else zero
                tasks["quality"] = torch.stack(quality_losses).sum() / session_events[key] if quality_losses else zero
                if negative_event_losses:
                    tasks["event"] = tasks["event"] + torch.stack(negative_event_losses).mean() * new_steps / session_lengths[key]
                    tasks["quality"] = tasks["quality"] + torch.stack(negative_quality_losses).mean() * new_steps / session_lengths[key]
                tasks["boundary"] = torch.stack(boundary_losses).sum() / session_events[key] if boundary_losses else zero
                state["sample_audit"][-1]["training_weights"] = {
                    "valid_timeline": new_steps / session_lengths[key], "positive_event": 1 / session_events[key],
                    "negative_family": new_steps / session_lengths[key] / max(len(negative_event_losses), 1),
                    "natural_calibration": 1.0}
                event_ids = torch.full((new_steps,), -2, dtype=torch.long, device=device)
                phases = torch.zeros_like(event_ids)
                invalid = batch["state_loss_mask"][0, mask[0]] <= 0
                event_ids[invalid], phases[invalid] = -1, -1
                for event_index, event in enumerate(session_truth.itertuples(index=False)):
                    boundary_near = ((current_times - int(event.start_ms)).abs() <= 30_000) | (
                        (current_times - int(event.end_ms)).abs() <= 30_000)
                    event_ids[boundary_near], phases[boundary_near] = -1, -1
                    eligible = (current_times > int(event.start_ms) + 30_000) & (current_times < int(event.end_ms) - 30_000)
                    eligible &= ~invalid
                    event_ids[eligible] = event_index
                    phases[eligible] = (((current_times[eligible] - int(event.start_ms)) * 3)
                                        // (int(event.end_ms) - int(event.start_ms))).clamp(0, 2)
                tasks["contrastive"], contrastive_pairs = event_contrastive_loss(
                    model.contrastive_projection(outputs["semantic_features"][0, mask[0]]), event_ids,
                    phases, temperature=config["loss"]["contrastive_temperature"],
                    maximum_pairs=config["loss"]["contrastive_max_pairs"])
                state["sample_audit"][-1]["contrastive_pairs"] = contrastive_pairs
                state["sample_audit"][-1]["contrastive_no_negative_or_positive_pairs"] = contrastive_pairs == 0
                weighted = {name: loss * config["loss"][name] * schedule.get(name, 1)
                            * (new_steps / session_lengths[key] if name not in {"event", "quality", "boundary"} else 1)
                            / training["accumulate_sessions"] for name, loss in tasks.items()}
                loss = sum(weighted.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite joint loss")
            if chunk_index % training["gradient_diagnostics_every_chunks"] == 0:
                report = gradient_diagnostics(weighted, model)
                state["gradient_diagnostics"].append({"epoch": epoch, "chunk": chunk_index,
                    "contrastive_pairs": contrastive_pairs, **report})
            loss.backward()
            cache.append(current_times.detach().cpu().numpy(), features.detach().cpu().numpy())
            state["cache_rows"] = cache.rows
            state["next_chunk"] = chunk_index + 1
            if last:
                state["session_count"] += 1
                state["session_key"], state["stream"] = None, None
                state["cache_rows"] = 0
                if state["session_count"] == training["accumulate_sessions"] or chunk_index == len(sampler.chunks) - 1:
                    if state["session_count"] < training["accumulate_sessions"]:
                        factor = training["accumulate_sessions"] / state["session_count"]
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(factor)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"],
                                                         error_if_nonfinite=True)
                    state["clipped_steps"] += int(float(norm) > training["gradient_clip_norm"])
                    state["optimizer_steps"] += 1
                    fraction = state["clipped_steps"] / state["optimizer_steps"]
                    if epoch >= 6 and fraction > training["clipping_abort_fraction"]:
                        raise RuntimeError(f"Joint clipping abort gate failed: {fraction:.4f}")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    state["update_count"] += 1
                    state["session_count"] = 0
            enforce_resources(config, device)
            if hasattr(inputs, "assert_session_unchanged"):
                inputs.assert_session_unchanged(key)
            checkpoint()
            print(f"[joint epoch={epoch}/{epochs} chunk={chunk_index + 1}/{len(sampler.chunks)}] "
                  f"subject={key[0]} loss={float(loss.detach()):.5f}", flush=True)
        fraction = state["clipped_steps"] / max(state["optimizer_steps"], 1)
        coverage = candidate_coverage(state["epoch_candidates"], truth)
        residuals = []
        independent = set()
        for candidate in state["epoch_candidates"]:
            positive, _, event_id, true_start, _ = candidate_targets(candidate, truth)
            if positive:
                independent.add((candidate.subject_key, candidate.session_id, event_id))
                residuals.append(abs(candidate.start_ms - true_start) / 1000)
        settings = config["boundary"]
        start_range = min(settings["maximum_start_range_seconds"], max(3,
            math.ceil(float(np.quantile(residuals, .95)) / 3) * 3)) if residuals else settings["maximum_start_range_seconds"]
        state["boundary"] = {"qualified": coverage["recall"] >= settings["minimum_candidate_recall"]
                             and len(independent) >= settings["minimum_positive_events"],
                             "positive_events": len(independent), "candidate_recall": coverage["recall"],
                             "start_range_seconds": start_range,
                             "end_range_seconds": settings["maximum_end_range_seconds"],
                             "sigma_seconds": settings["sigma_seconds"]}
        state["epoch_complete"] = True
        qualification = epoch >= training["checkpoint_min_epoch"] and fraction <= training["clipping_promotion_fraction"]
        epoch_payload = {"model": model.state_dict(), "identity": identity, "epoch": epoch,
            "epoch_complete": True, "promotion_eligible": qualification, "clipping_fraction": fraction,
            "training_subjects": sorted(subjects),
            "excluded_subjects": sorted((outer_excluded or set()) | (selector or set())),
            "preprocessing": preprocessing, "boundary": state["boundary"],
            "config_sha256": bound_config, "source_sha256": digest(bound_source),
            "data_sha256": digest(inputs.identity)}
        checkpoint()
        if not all(torch.isfinite(value).all() for value in epoch_payload["model"].values()):
            raise RuntimeError("Epoch checkpoint finite gate failed")
        save_torch_atomic(directory / "latest.pt", epoch_payload)
        record = {"epoch": epoch, "clipping_fraction": fraction, "promotion_eligible": qualification,
                  "chunks": len(sampler.chunks), "sessions": len(session_lengths),
                  "optimizer_steps": state["optimizer_steps"], "candidate_coverage": coverage,
                  "elapsed_seconds": time.monotonic() - started, **enforce_resources(config, device)}
        stop = False
        if selector and epoch >= training["checkpoint_min_epoch"] and epoch % training["validation_every_epochs"] == 0:
            assert_excluded(subjects, selector, outer_excluded or set())
            fitted_mapping = fit_source_mapping(model, inputs, subjects, config, preprocessing,
                                                directory / "source_fit" / str(epoch), identity)
            predictions = predict_joint_model(model, inputs, selector, config, preprocessing,
                                               directory / "selector" / str(epoch), identity,
                                               state["boundary"])
            _, _, selector_events, _ = inputs.frames(selector, labels=True)
            selector_truth, selector_ignore = partition_evaluation_events(selector_events, selector)
            point, search_records = choose_operating_point(predictions["coarse"], predictions["windows"],
                selector_truth, selector_ignore, {identity: fitted_mapping}, config)
            record["selector"] = point
            write_json_atomic(directory / "selector" / str(epoch) / "search.json", search_records)
            f1 = point["f1"]
            eligible = qualification and point["promotion_eligible"]
            if qualification and f1 > state["diagnostic_f1"]:
                state["diagnostic_epoch"], state["diagnostic_f1"] = epoch, f1
                save_torch_atomic(directory / "diagnostic_selected.pt", {
                    **epoch_payload, "promotion_eligible": False, "diagnostic_only": True})
            if eligible and (state["best_epoch"] is None or f1 > state["best_f1"] + training["f1_min_delta"]):
                state["best_epoch"], state["best_f1"], state["stale_checks"] = epoch, f1, 0
                save_torch_atomic(directory / "selected.pt", epoch_payload)
            else:
                state["stale_checks"] += 1
            stop = state["stale_checks"] >= training["patience_checks"]
        state["history"].append(record)
        write_json_atomic(directory / "training_history.json", state["history"])
        write_json_atomic(directory / "sampling.json", state["sample_audit"])
        write_json_atomic(directory / "gradients.json", state["gradient_diagnostics"])
        state["epoch"] += 1
        state["next_chunk"] = 0
        state["sampler_identity"] = None
        state["epoch_complete"] = False
        state["clipped_steps"], state["optimizer_steps"] = 0, 0
        state["event_contributions"], state["family_contributions"] = set(), set()
        state["epoch_candidates"] = []
        checkpoint()
        if stop:
            break
    state["training_complete"] = True
    checkpoint()
    if selector:
        if state["best_epoch"] is None and state["diagnostic_epoch"] is None:
            raise RuntimeError("No finite/clipping-qualified joint epoch; diagnostic checkpoints preserved")
        if state["best_epoch"] is None:
            write_json_atomic(directory / "selector_failure_report.json", {
                "promotion_eligible": False, "diagnostic_epoch": state["diagnostic_epoch"],
                "reason": "No feasible selector operating point"})
        return state["best_epoch"] or state["diagnostic_epoch"]
    return epochs


def decoder_settings(config):
    return {f"high_{high:g}_gap_{gap}": {**config["decoder"], "high_threshold": high,
                                         "gap_merge_seconds": gap}
            for high, gap in product(config["search"]["high_thresholds"], config["search"]["gap_seconds"])}


@torch.no_grad()
def predict_joint_model(model, inputs, subjects, config, preprocessing, directory, identity,
                        boundary, *, training_source: bool = False):
    device = next(model.parameters()).device
    dataset, _ = make_joint_dataset(inputs, subjects, config, preprocessing, training=training_source)
    dataset.training = False
    sampler = SessionChunkSampler(dataset, 0, training=False)
    loader = session_loader(dataset, sampler, config)
    model.eval()
    coarse = {name: [] for name in decoder_settings(config)}
    refined = {str(entropy): {name: [] for name in coarse} for entropy in config["boundary"]["entropy_thresholds"]}
    windows, streams, current_key, cache = [], {}, None, None
    for chunk_index, batch in enumerate(loader):
        _, _, _, key, last, count = sampler.chunks[chunk_index]
        if hasattr(inputs, "assert_session_unchanged"):
            inputs.assert_session_unchanged(key)
        if key != current_key:
            current_key = key
            streams = {name: StreamingProposals(*key, identity, decoder)
                       for name, decoder in decoder_settings(config).items()}
            cache = TimelineCache(directory / "cache" / stable_id(*key),
                {"identity": identity, "epoch": "prediction", "subject": key[0], "session": key[1]},
                FEATURE_WIDTH)
        batch = {name: value.to(device) if isinstance(value, torch.Tensor) else value
                 for name, value in batch.items()}
        outputs = model(batch)
        mask = torch.zeros(batch["timestamp_ms"].shape[1], dtype=torch.bool, device=device)
        mask[-count:] = True
        times = batch["timestamp_ms"][0, mask]
        features = pack_features(outputs, mask)
        cache.append(times.cpu().numpy(), features.cpu().numpy())
        all_times, all_features = cache.tensors()
        for timestamp, values in zip(times.tolist(), features.cpu().numpy()):
            windows.append({"subject_key": key[0], "session_id": key[1], "timestamp_ms": timestamp,
                            "valid": float(values[128]), "gyro_valid": float(values[135]),
                            "model_identity": identity})
        for setting, stream in streams.items():
            candidates = generate_chunk(stream, times, features)
            if last:
                candidates.extend(stream.finish())
            for candidate in candidates:
                score_candidate(model, candidate, all_times, all_features, device)
                coarse[setting].append(candidate)
                for entropy in config["boundary"]["entropy_thresholds"]:
                    refined[str(entropy)][setting].append(refine_candidate(model, candidate,
                        all_times, all_features, device, boundary, entropy))
        enforce_resources(config, device)
        if hasattr(inputs, "assert_session_unchanged"):
            inputs.assert_session_unchanged(key)
        if hasattr(inputs, "file_signature") and inputs.file_signature() != inputs.bound_files:
            raise RuntimeError("Runtime canonical input changed during prediction")
    return {"coarse": coarse, "refined": refined, "windows": pd.DataFrame(windows)}


def fit_source_mapping(model, inputs, subjects, config, preprocessing, directory, identity):
    outputs = predict_joint_model(model, inputs, subjects, config, preprocessing, directory, identity,
                                   {"qualified": False}, training_source=True)
    candidates = [candidate for pool in outputs["coarse"].values() for candidate in pool]
    mapping = FrozenSourceCDF.fit(candidates, subjects)
    write_json_atomic(directory / "source_mapping.json", mapping.to_json())
    return mapping


def load_trained(directory, config, identity, subjects, data_identity):
    payload = torch.load(directory / "latest.pt", map_location=config["training"]["device"], weights_only=True)
    if not checkpoint_qualified(payload, identity, subjects, config["training"]["checkpoint_min_epoch"]):
        raise RuntimeError("Unqualified joint checkpoint cannot enter OOF/promotion")
    if payload.get("config_sha256") != config_identity(config) or payload.get("source_sha256") != digest(
            source_identity(Path(config["_config_path"]).resolve().parents[1])):
        raise RuntimeError("Joint checkpoint config/source identity mismatch")
    if payload.get("data_sha256") != digest(data_identity):
        raise RuntimeError("Joint checkpoint data identity mismatch")
    model = seed_model(config).to(config["training"]["device"])
    model.load_state_dict(payload["model"])
    return model, payload


def boundary_pair_gate(coarse, refined, truth, ignore, windows, config, *, require_sixty=True,
                       refined_candidates=None):
    coarse_metrics, coarse_match = metrics_with_exposure(truth, coarse, ignore, windows)
    refined_metrics, refined_match = metrics_with_exposure(truth, refined, ignore, windows)
    event_keys = ["subject_key", "session_id", "event_id"]
    keys = [*event_keys, "prediction_event_id"]
    proposal_keys = ["subject_key", "session_id", "prediction_event_id"]
    before = set(map(tuple, coarse_match[proposal_keys].values)) if len(coarse_match) else set()
    after = set(map(tuple, refined_match[proposal_keys].values)) if len(refined_match) else set()
    paired = coarse_match.merge(refined_match, on=keys, suffixes=("_coarse", "_refined")) if before and after else pd.DataFrame()
    improvements = {}
    for endpoint in ("start", "end"):
        old = float(paired[endpoint + "_absolute_error_ms_coarse"].mean()) if len(paired) else float("nan")
        new = float(paired[endpoint + "_absolute_error_ms_refined"].mean()) if len(paired) else float("nan")
        improvements[endpoint] = (old - new) / old if old > 0 else 0
    tp_to_fp = len(before - after) / max(len(before), 1)
    population = {candidate.proposal_id: candidate for candidate in
                  (refined_candidates if refined_candidates is not None else refined)}
    population.update({candidate.proposal_id: candidate for candidate in refined})
    population = list(population.values())
    clipping = sum(candidate.boundary_clipped for candidate in population) / max(len(population), 1)
    conflicts = sum(candidate.boundary_conflict for candidate in population)
    gates = config["gates"]
    positive_events = len(paired[event_keys].drop_duplicates()) if len(paired) else 0
    passed = (positive_events >= (config["boundary"]["minimum_positive_events"] if require_sixty else 1)
              and all(value >= gates["boundary_minimum_mae_improvement"] for value in improvements.values())
              and coarse_metrics["f1"] - refined_metrics["f1"] <= gates["boundary_maximum_f1_drop"]
              and tp_to_fp <= gates["boundary_maximum_tp_to_fp_rate"]
              and clipping <= gates["boundary_maximum_clipping_rate"])
    return {"passed": passed, "paired_positive_events": positive_events, "improvements": improvements,
            "coarse": coarse_metrics, "refined": refined_metrics, "tp_to_fp_rate": tp_to_fp,
            "clipping_rate": clipping, "conflicts": conflicts,
            "diagnostic_candidate_count": len(population),
            "changed_candidates": sum(candidate.refined for candidate in population),
            "changed_endpoints": sum(int(candidate.start_ms != candidate.unrefined_start_ms)
                + int(candidate.end_ms != candidate.unrefined_end_ms) for candidate in population
                if candidate.unrefined_start_ms is not None and candidate.unrefined_end_ms is not None),
            "valid_start_bins": [candidate.boundary_valid_start_bins for candidate in population],
            "valid_end_bins": [candidate.boundary_valid_end_bins for candidate in population],
            "fallback_reasons": {reason: sum(candidate.boundary_fallback_reason == reason for candidate in population)
                                 for reason in {candidate.boundary_fallback_reason for candidate in population}}}


def train_partition_protocol(inputs, subjects, outer, config, directory, run_identity, resume):
    if resume and (directory / "internal_manifest.json").exists():
        point = verify_internal_manifest(directory, config, inputs.identity, subjects, outer, run_identity)
        return int(np.median([lineage["epoch"] for lineage in point["lineage"]])), point
    partitions = stacking_partitions(subjects, 3, 2026)
    epochs, outputs, mappings, lineages = [], [], {}, []
    for partition in range(3):
        prediction = {subject for subject, group in partitions.items() if group == partition}
        train = subjects - prediction
        assert_excluded(train, prediction, outer)
        anchors, _, events, _ = inputs.frames(train, labels=True)
        fit, selector, split_report = _selector_split(train, config["training"]["selector_fraction"],
                                                     2026, events=events, anchors=anchors)
        partition_root = directory / f"partition_{partition}"
        identity = digest({"run": run_identity, "training": sorted(train), "prediction": sorted(prediction),
                           "outer": sorted(outer), "partition": partition})
        write_json_atomic(partition_root / "selector_split.json", split_report)
        selected_epoch = train_joint_model(inputs, fit, config, partition_root / "selection", identity,
            epochs=config["training"]["max_epochs"], resume=resume, selector=selector,
            outer_excluded=outer | prediction)
        epochs.append(selected_epoch)
        train_joint_model(inputs, train, config, partition_root / "retrain", identity,
                          epochs=selected_epoch, resume=resume, outer_excluded=outer | prediction)
        model, payload = load_trained(partition_root / "retrain", config, identity, train, inputs.identity)
        mapping = fit_source_mapping(model, inputs, train, config, payload["preprocessing"],
                                      partition_root / "source_fit", identity)
        mappings[identity] = mapping
        output = predict_joint_model(model, inputs, prediction, config, payload["preprocessing"],
            partition_root / "prediction", identity, payload["boundary"])
        outputs.append(output)
        history = json.loads((partition_root / "selection" / "training_history.json").read_text(encoding="utf-8"))
        selected_record = next(record for record in history if record["epoch"] == selected_epoch)
        lineages.append({"partition": partition, "model_identity": identity,
                         "training_subjects": sorted(train), "prediction_subjects": sorted(prediction),
                         "outer_excluded_subjects": sorted(outer), "epoch": selected_epoch,
                         "source_training_subjects": mapping.training_subjects,
                         "normalization_subjects": payload["preprocessing"]["training_subjects"],
                         "selector_fit_subjects": sorted(fit), "selector_subjects": sorted(selector),
                         "config_sha256": payload["config_sha256"], "source_sha256": payload["source_sha256"],
                         "data_sha256": payload["data_sha256"], "checkpoint_qualified": True})
        lineages[-1]["selector_promotion_eligible"] = bool(selected_record.get("selector", {}).get("promotion_eligible"))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    windows = pd.concat([output["windows"] for output in outputs], ignore_index=True)
    coarse = {name: [candidate for output in outputs for candidate in output["coarse"][name]]
              for name in outputs[0]["coarse"]}
    _, _, events, _ = inputs.frames(subjects, labels=True)
    truth, ignore = partition_evaluation_events(events, subjects)
    point, search = choose_operating_point(coarse, windows, truth, ignore, mappings, config)
    coarse_point = copy.deepcopy(point)
    natural, _ = apply_online_budget(coarse[point["setting"]], windows, mappings, point["budget"],
                                     config["decoder"]["quotas"])
    calibrator = NonnegativePlatt(**point["calibration"])
    coarse_predictions = commit_candidates(natural, calibrator, point["acceptance"], point["nms"])
    boundary_reports, enabled, best_boundary_f1, best_entropy, winning_point = [], False, -1.0, None, None
    for entropy in config["boundary"]["entropy_thresholds"]:
        candidates = [candidate for output in outputs for candidate in output["refined"][str(entropy)][point["setting"]]]
        natural_refined = {candidate.proposal_id: candidate for candidate in candidates}
        refined_candidates = [natural_refined[candidate.proposal_id] for candidate in natural]
        from bme_eating.joint_evaluation import event_frame
        ignored = prediction_ignore_mask(event_frame(refined_candidates), ignore)
        targets = np.asarray([candidate_targets(candidate, truth)[0] for candidate in refined_candidates])
        keep = ~ignored | (targets > 0)
        logits = np.asarray([candidate.event_logit for candidate in refined_candidates])
        refined_calibrator = NonnegativePlatt.fit(logits[keep], targets[keep])
        ece = calibration_ece(refined_calibrator.transform(logits[keep]), targets[keep], config["calibration"]["bins"])
        refined_point, refined_best = None, None
        report = boundary_pair_gate(coarse_predictions, coarse_predictions, truth, ignore, windows, config)
        if any(candidate.refined for candidate in refined_candidates) and refined_calibrator.qualified and ece <= config["calibration"]["maximum_ece"]:
            acceptances = np.arange(config["search"]["acceptance_min"], config["search"]["acceptance_max"] + 1e-8,
                                    config["search"]["acceptance_step"])
            for nms, acceptance in product(config["search"]["nms_ious"], acceptances):
                refined_predictions, refined_diagnostics = commit_candidates(refined_candidates, refined_calibrator,
                    float(acceptance), nms, return_diagnostics=True)
                paired_coarse = commit_candidates(natural, refined_calibrator, float(acceptance), nms)
                trial = boundary_pair_gate(paired_coarse, refined_predictions, truth, ignore, windows, config,
                                           refined_candidates=refined_diagnostics)
                trial["locked_coarse"] = coarse_point
                metrics = trial["refined"]
                eligible = (trial["passed"] and point["candidate_recall"] >= 0.70
                            and metrics["fp_per_hour"] <= config["gates"]["maximum_fp_per_hour"]
                            and coarse_point["f1"] - metrics["f1"] <= config["gates"]["boundary_maximum_f1_drop"])
                search.append({"boundary_entropy": entropy, "nms": nms, "acceptance": float(acceptance),
                               "calibration_ece": ece, "promotion_eligible": eligible, **metrics})
                key = (int(eligible), metrics["f1"], -metrics["fp_per_hour"])
                if refined_best is None or key > refined_best:
                    refined_best, report = key, trial
                    refined_point = {**point, **metrics, "calibration": refined_calibrator.to_json(),
                                     "calibration_ece": ece, "acceptance": float(acceptance), "nms": nms,
                                     "promotion_eligible": eligible}
        report["entropy"] = entropy
        boundary_reports.append(report)
        if refined_point and refined_point["promotion_eligible"] and report["passed"] and report["refined"]["f1"] > best_boundary_f1:
            enabled, best_boundary_f1, best_entropy = True, report["refined"]["f1"], entropy
            winning_point = refined_point
    if winning_point is not None:
        point = winning_point
    point["coarse_operating_point"] = coarse_point
    point["boundary_enabled"], point["boundary_entropy"] = enabled, best_entropy
    point["promotion_eligible"] &= all(lineage["selector_promotion_eligible"] for lineage in lineages)
    point["lineage"] = lineages
    verify_oof_lineage(lineages, subjects, outer, identities=protocol_identities(config, inputs.identity))
    write_json_atomic(directory / "operating_point.json", point)
    write_json_atomic(directory / "search.json", search)
    write_json_atomic(directory / "boundary_crossfit.json", boundary_reports)
    if not enabled:
        write_json_atomic(directory / "boundary_failure_report.json", {
            "boundary_enabled": False, "coarse_retained": True,
            "reason": "No internal cross-fit entropy passed all Boundary gates", "diagnostics": boundary_reports})
    windows.to_parquet(directory / "internal_oof_windows.parquet", index=False)
    save_torch_atomic(directory / "internal_oof.pt", {"outputs": outputs, "lineages": lineages,
                                                      "mappings": mappings})
    evidence = {path.relative_to(directory).as_posix(): sha256_file(path) for path in directory.rglob("*")
                if path.is_file() and "cache" not in path.relative_to(directory).parts
                and path.suffix in {".json", ".pt", ".parquet"}
                and path.name not in {"internal_manifest.json", "resume.pt"}}
    write_json_atomic(directory / "internal_manifest.json", {"protocol": PROTOCOL,
        "run_identity": run_identity, "training_subjects": sorted(subjects),
        "outer_excluded_subjects": sorted(outer), **protocol_identities(config, inputs.identity),
        "evidence_sha256": evidence})
    return int(np.median(epochs)), point
