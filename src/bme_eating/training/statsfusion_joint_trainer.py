from __future__ import annotations

import copy
import json
import math
import pickle
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
    release_loaders_after,
    reuse_session_loader,
    session_loader,
)
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_capabilities import capabilities, supported
from bme_eating.joint_evaluation import (
    NonnegativePlatt,
    apply_online_budget,
    calibration_ece,
    candidate_coverage,
    choose_operating_point,
    commit_candidates,
    metrics_with_exposure,
)
from bme_eating.joint_progress import print_metrics, progress
from bme_eating.joint_protocol import (
    assert_excluded,
    calibration_scope,
    checkpoint_qualified,
    config_identity,
    configure_numerics,
    digest,
    internal_partitions,
    protocol_for,
    protocol_identities,
    save_torch_atomic,
    source_identity,
    verify_calibration_lineage,
    verify_internal_manifest,
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
    candidate_roi,
    distribution_endpoint,
    event_contrastive_loss,
    gaussian_distribution,
    loss_schedule,
    quality_focal_loss,
)
from bme_eating.training.hierarchical_v4_trainer import _selector_split, stacking_partitions

FEATURE_WIDTH = 137


class PackedCheckpointState:
    def __init__(self, state):
        self.payload = pickle.dumps(state, protocol=5)

    def __reduce__(self):
        return pickle.loads, (self.payload,)


def tensors_finite(values):
    groups = {}
    for value in values:
        groups.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(checks).all()) for checks in groups.values())


def cpu_checkpoint_snapshot(payload):
    groups, replacements = {}, {}

    def collect(value):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            key = (value.device, value.dtype)
            if id(value) not in replacements:
                groups.setdefault(key, []).append(value)
                replacements[id(value)] = None
        elif isinstance(value, dict):
            for member in value.values():
                collect(member)
        elif isinstance(value, (list, tuple)):
            for member in value:
                collect(member)

    sections = {key: payload[key] for key in ("model", "optimizer", "gradients") if key in payload}
    collect(sections)
    for members in groups.values():
        packed = torch.cat([value.detach().reshape(-1) for value in members]).cpu()
        offset = 0
        for value in members:
            replacements[id(value)] = packed[offset:offset + value.numel()].reshape(value.shape)
            offset += value.numel()

    def rebuild(value):
        if isinstance(value, torch.Tensor):
            return replacements.get(id(value), value)
        if isinstance(value, dict):
            result = copy.copy(value)
            for key, member in value.items():
                result[key] = rebuild(member)
            return result
        if isinstance(value, list):
            return [rebuild(member) for member in value]
        if isinstance(value, tuple):
            return tuple(rebuild(member) for member in value)
        return value

    return {**payload, **rebuild(sections)}


def training_checkpoint_snapshot(payload):
    snapshot = cpu_checkpoint_snapshot(payload)
    snapshot["state"] = PackedCheckpointState(payload["state"])
    return snapshot


def proposal_supervision(batch, cpu_batch, valid, scale_valid, factor, fractional=False):
    target = batch["state_target"].float()
    labels_valid = batch["state_loss_mask"]
    scale_target, scale_mask = torch.zeros_like(target), torch.zeros_like(valid)
    if factor > target.shape[1]:
        return scale_target, scale_mask
    completed = ((cpu_batch["timeline_position"][0, factor - 1:] + 1) % factor) == 0
    selected = completed.nonzero().flatten()
    if not len(selected):
        return scale_target, scale_mask
    weights = labels_valid.unfold(1, factor, 1)
    denominator = weights.sum(-1)
    epsilon = torch.finfo(target.dtype).eps if fractional else 1
    targets = (target.unfold(1, factor, 1) * weights).sum(-1) / denominator.clamp_min(epsilon)
    endpoints = (selected + factor - 1).to(target.device, non_blocking=True)
    scale_target[:, endpoints] = targets[:, selected.to(target.device, non_blocking=True)]
    scale_mask[:, endpoints] = (valid[:, endpoints] * (scale_valid[:, endpoints] > 0)
                                * (denominator[:, selected.to(target.device, non_blocking=True)] > 0))
    cpu_target, cpu_weights = cpu_batch["state_target"], cpu_batch["state_loss_mask"]
    nonbinary = ((cpu_target != 0) & (cpu_target != 1)) | ((cpu_weights != 0) & (cpu_weights != 1))
    fallback = nonbinary.unfold(1, factor, 1).any(-1).any(0) & completed
    for position in fallback.nonzero().flatten().tolist():
        endpoint = position + factor - 1
        members = slice(position, endpoint + 1)
        member_weights = labels_valid[:, members]
        scale_target[:, endpoint] = ((target[:, members] * member_weights).sum(1)
                                     / member_weights.sum(1).clamp_min(epsilon))
        scale_mask[:, endpoint] = (valid[:, endpoint] * (scale_valid[:, endpoint] > 0)
                                   * (member_weights.sum(1) > 0))
    return scale_target, scale_mask


def epoch_clipping_report(epoch, clipped_steps, optimizer_steps, expected_optimizer_steps,
                          maximum_fraction):
    if (expected_optimizer_steps < 1 or not 0 <= clipped_steps <= optimizer_steps
            <= expected_optimizer_steps or not 0 <= maximum_fraction <= 1):
        raise RuntimeError("Invalid joint epoch clipping counters or threshold")
    minimum_fraction = clipped_steps / expected_optimizer_steps
    return {"protocol": "full_epoch_optimizer_updates_v1", "epoch": epoch,
            "clipped_steps": clipped_steps, "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_optimizer_steps,
            "clipping_fraction": clipped_steps / max(optimizer_steps, 1),
            "minimum_epoch_clipping_fraction": minimum_fraction,
            "maximum_clipping_fraction": maximum_fraction,
            "abort_failed": epoch >= 6 and minimum_fraction > maximum_fraction}


def seed_model(config):
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    configure_numerics(config)
    from bme_eating.models.statsfusion_joint_event_v411 import build_joint_model
    return build_joint_model(config)


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
    if supported(config, "grouped"):
        from bme_eating.joint_v411_resources import private_process_tree
        usage.update(private_process_tree())
        if usage["process_tree_private_ram_gb"] >= config["resources"]["maximum_private_ram_gb"]:
            raise RuntimeError("Process tree private RAM exceeds registered 26 GB limit")
    if max(usage["peak_gpu_gb"], usage["peak_gpu_reserved_gb"]) >= config["resources"]["maximum_gpu_gb"]:
        raise RuntimeError("GPU memory gate failed; protocol cannot silently shrink batches")
    return usage


def pack_features(outputs, mask):
    scalar = [outputs["sensor_valid"], outputs["state_logit"].sigmoid(),
              *(outputs["proposal_scales"][str(scale)][0].sigmoid()
                * (outputs["proposal_scales"][str(scale)][1] > 0) for scale in (3, 15, 60)),
              outputs["onset_logit"].sigmoid(), outputs["offset_logit"].sigmoid(),
              outputs["gyro_valid_fraction"], outputs["missing_fraction"]]
    parts = [outputs["semantic_features"][0, mask], outputs["local_features"][0, mask],
             torch.stack([value[0, mask] for value in scalar], -1)]
    if "localization_features" in outputs:
        parts.append(outputs["localization_features"][0, mask])
    return torch.cat(parts, -1).float()


def generate_chunk(stream: StreamingProposals, times: torch.Tensor, features: torch.Tensor):
    candidates = []
    for timestamp, values in zip(times.tolist(), features.detach().cpu().numpy()):
        probabilities = {"state": float(values[129]),
                         **{f"proposal_{scale}": float(values[130 + index])
                            for index, scale in enumerate((3, 15, 60))}}
        arguments = (int(timestamp), probabilities, float(values[128]), float(values[134]))
        candidates.extend(stream.observe(*arguments, onset_probability=float(values[133]))
                          if stream.config.get("learned_localization") else stream.observe(*arguments))
    return candidates


def score_candidate(model, candidate, times, features, device):
    if supported(model, "grouped"):
        from bme_eating.joint_v411_scoring import score_candidates
        return score_candidates(model, [candidate], times, features, device)[0]
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
    if supported(model, "grouped"):
        from bme_eating.joint_v411_scoring import refine
        return refine(model, candidate, times, features, device, boundary, entropy)
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
                                 maximum=2, durations=(15, 30, 60, 120, 300)):
    if not len(times):
        return []
    decision = int(times[-1])
    candidates, seen = [], set()
    for _ in range(max(16, maximum * 4)):
        end = decision - int(rng.integers(0, 11)) * 3000
        duration = int(rng.choice(durations)) * 1000
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
    heads = [(name, parameter) for name, parameter in model.named_parameters()
             if not name.startswith("encoder.") and parameter.requires_grad]
    vectors, report = {}, {}
    for name, loss in tasks.items():
        all_gradients = torch.autograd.grad(loss, [*parameters, *[parameter for _, parameter in heads]],
                                           retain_graph=True, allow_unused=True)
        gradients = all_gradients[:len(parameters)]
        vector = torch.cat([gradient.detach().float().flatten() if gradient is not None
                            else parameter.new_zeros(parameter.numel()).float()
                            for parameter, gradient in zip(parameters, gradients)])
        vectors[name] = vector
        report[name + "_weighted_norm"] = float(vector.norm())
        head_norms = {}
        for (head_name, _), gradient in zip(heads, all_gradients[len(parameters):]):
            if gradient is not None:
                module = head_name.split(".")[0]
                head_norms[module] = head_norms.get(module, 0) + float(gradient.detach().float().square().sum())
        for module, squared_norm in head_norms.items():
            report[name + "_head_" + module + "_weighted_norm"] = math.sqrt(squared_norm)
    for left, right in product(vectors, repeat=2):
        if left >= right:
            continue
        denominator = vectors[left].norm() * vectors[right].norm()
        cosine = (vectors[left] @ vectors[right] / denominator.clamp_min(1e-12)).clamp(-1, 1)
        report[left + "_" + right + "_angle_degrees"] = (float(torch.acos(cosine) * 180 / math.pi)
            if float(denominator) > 0 else None)
    return report


@release_loaders_after
def train_joint_model(inputs: JointInputs, subjects: set[str], config: dict[str, Any],
                      directory: Path, identity: str, *, epochs: int, resume: bool,
                      selector: set[str] | None = None, outer_excluded: set[str] | None = None,
                      deployment_epoch: int | None = None):
    if deployment_epoch is not None and (not supported(config, "full_data_workflow")
            or selector or len(subjects) != 39 or epochs != deployment_epoch + 4):
        raise ValueError("Neighbor checkpoint extension requires final39 with four extra epochs")
    directory.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    device = torch.device(training["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required by protocol")
        torch.cuda.reset_peak_memory_stats(device)
    model = seed_model(config).to(device)
    v411 = supported(model, "grouped")
    v412 = supported(model, "consistent_inputs")
    v413 = supported(model, "stabilized")
    from bme_eating.joint_v411_features import FeatureBlocks
    from bme_eating.joint_v411_journal import CheckedJournal
    from bme_eating.joint_v411_stream import make_stream
    from bme_eating.joint_v411_training import PositiveGradientBuffer
    if v412:
        from bme_eating.joint_v412_training import (
            ReplaceablePositiveGradientBuffer,
            apply_learning_rate,
            heatmap_focal_loss,
            subject_weights,
        )
        PositiveGradientBuffer = ReplaceablePositiveGradientBuffer
    if v413:
        from bme_eating.joint_v413_training import (
            ContributionBank,
            PositiveGradientBuffer,
            heatmap_focal_loss,
            selection_update,
        )
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
             "cache_rows": 0, "cache_sha256": None, "session_key": None, "stream": None,
             "history": [], "best_epoch": None, "best_f1": -1.0, "stale_checks": 0,
             "diagnostic_epoch": None, "diagnostic_f1": -1.0,
             "event_contributions": set(), "family_contributions": set(),
             "epoch_candidates": [], "gradient_diagnostics": [], "sample_audit": [],
             "training_complete": False}
    if v411:
        state.update({"positive_gradients": {}, "positive_group_counts": {},
                      "journal_offsets": {}, "contrastive_history": None})
    if v413:
        ContributionBank(state)
        state.update({"natural_replay": None, "unqualified_checks": 0, "lost_qualification_checks": 0})
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["identity"] != identity or set(payload["training_subjects"]) != subjects:
            raise RuntimeError("Training checkpoint identity or subject lineage mismatch")
        if (payload.get("requested_epochs") != epochs
                or payload.get("deployment_epoch") != deployment_epoch
                or set(payload.get("selector_subjects", [])) != (selector or set())
                or set(payload.get("excluded_subjects", [])) != (outer_excluded or set()) | (selector or set())):
            raise RuntimeError("Training resume epoch/selector/exclusion protocol changed")
        if (payload.get("config_sha256") != bound_config
                or payload.get("data_sha256") != digest(inputs.identity)):
            raise RuntimeError("Training checkpoint config/source/data identity mismatch")
        migration = None
        if v411 and payload.get("source_sha256") != digest(bound_source):
            raise RuntimeError("v411 resume source changed; create a new run")
        if payload.get("source_sha256") != digest(bound_source):
            from bme_eating.joint_resume_migration import authorize_checkpoint_migration
            migration = authorize_checkpoint_migration(directory, checkpoint_path, payload, bound_source)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for name, parameter in model.named_parameters():
            gradient = payload["gradients"].get(name)
            parameter.grad = gradient.to(device) if gradient is not None else None
        state = payload["state"]
        if v412:
            PositiveGradientBuffer(state["positive_gradients"])
        if migration:
            state["resume_migration"] = migration
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])
        if state.get("training_complete"):
            if selector and ((not state["best_epoch"] or state.get("selection_failure")) if v413 else not (state["best_epoch"] or state["diagnostic_epoch"])):
                raise RuntimeError("No finite/clipping-qualified joint epoch; diagnostic checkpoints preserved")
            return (state["best_epoch"] or state["diagnostic_epoch"]) if selector else epochs
    natural_dataset = None
    replay_payload = None
    if v413:
        ContributionBank(state)
        natural_dataset, _ = make_joint_dataset(inputs, subjects, config, preprocessing, training=True)
        natural_dataset.training = False
    truth, _ = partition_evaluation_events(dataset.events, subjects)
    session_lengths = {key: len(group) for key, group in dataset.session_groups.items()}
    contribution_weights = subject_weights(session_lengths) if v412 else {subject: 1.0 for subject in subjects}
    if v412:
        write_json_atomic(directory / "subject_weights.json", {
            "weights": contribution_weights,
            "session_counts": {subject: sum(key[0] == subject for key in session_lengths) for subject in sorted(subjects)},
            "natural_calibration_weights": True})
    session_events = {key: max(1, int(((truth.subject_key.astype(str) == key[0])
                        & (truth.session_id.astype(str) == key[1])).sum())) for key in session_lengths}
    started = time.monotonic()
    journals = {name: CheckedJournal(directory / (name + ".jsonl"), state["journal_offsets"].get(name, 0))
                for name in ("sampling", "gradients", "candidate_history", "training_history")} if v411 else {}

    def checkpoint():
        checkpoint_started = time.monotonic()
        if config_identity(config) != bound_config:
            raise RuntimeError("Resolved training configuration changed before checkpoint")
        if source_signature() != bound_signature and source_identity(source_root) != bound_source:
            raise RuntimeError("Runtime source changed before checkpoint")
        if input_signature is not None and inputs.file_signature() != input_signature:
            raise RuntimeError("Runtime input data changed before checkpoint")
        finite = tensors_finite(model.state_dict().values())
        if not finite or not tensors_finite(parameter.grad for parameter in model.parameters()
                                            if parameter.grad is not None):
            raise RuntimeError("Finite gate failed before chunk checkpoint")
        if v411:
            positive_values = (PositiveGradientBuffer(state["positive_gradients"]).tensors() if v412 else
                               (value for event in state["positive_gradients"].values()
                                for value in event["gradients"].values()))
            if v413 and not tensors_finite(ContributionBank(state).tensors()):
                raise RuntimeError("Contribution bank finite gate failed")
            if not tensors_finite(positive_values):
                raise RuntimeError("Positive event gradient buffer finite gate failed")
            if not tensors_finite(value for moment in optimizer.state.values()
                                  for value in moment.values() if isinstance(value, torch.Tensor)):
                raise RuntimeError("Optimizer state finite gate failed")
            for name, key in (("sampling", "sample_audit"), ("gradients", "gradient_diagnostics"),
                              ("candidate_history", "epoch_candidates")):
                records = state[key]
                payloads = ([{"epoch": state["epoch"], "candidate": candidate.to_dict()}
                             for candidate in records] if name == "candidate_history" else records)
                journals[name].append(payloads)
                state["journal_offsets"][name] = journals[name].offset
                state[key] = []
        if v413:
            state["checkpoint_precommit_seconds"] = time.monotonic() - checkpoint_started
        save_torch_atomic(checkpoint_path, training_checkpoint_snapshot({"identity": identity, "training_subjects": sorted(subjects),
            "requested_epochs": epochs, "deployment_epoch": deployment_epoch, "selector_subjects": sorted(selector or set()),
            "excluded_subjects": sorted((outer_excluded or set()) | (selector or set())),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
            "gradients": {name: parameter.grad.detach() if parameter.grad is not None else None
                          for name, parameter in model.named_parameters()},
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "config_sha256": bound_config, "source_sha256": digest(bound_source),
            "data_sha256": digest(inputs.identity), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}))
        if v413:
            state["previous_checkpoint_total_seconds"] = time.monotonic() - checkpoint_started

    loader = None
    while state["epoch"] <= epochs:
        active_cache = None
        epoch = state["epoch"]
        sampler = SessionChunkSampler(dataset, epoch, state["next_chunk"])
        if state.get("sampler_identity") and state["sampler_identity"] != sampler.identity:
            raise RuntimeError("Resume session order changed")
        state["sampler_identity"] = sampler.identity
        expected_optimizer_steps = math.ceil(
            sum(chunk[4] for chunk in sampler.chunks) / training["accumulate_sessions"])
        epoch_clipping_report(epoch, state["clipped_steps"], state["optimizer_steps"],
                              expected_optimizer_steps, training["clipping_abort_fraction"])
        model.train()
        loader = (reuse_session_loader(loader, dataset, sampler, config)
                  if isinstance(loader, torch.utils.data.DataLoader)
                  else session_loader(dataset, sampler, config))
        chunk_progress = progress(enumerate(loader, start=sampler.start_chunk),
            total=len(sampler.chunks), initial=sampler.start_chunk,
            description=f"{directory.parent.name}/{directory.name} epoch {epoch}/{epochs}")
        for chunk_index, batch in chunk_progress:
            chunk_started = time.monotonic()
            chunk_timing = {}
            _, _, _, key, last, new_steps = sampler.chunks[chunk_index]
            if hasattr(inputs, "assert_session_unchanged"):
                inputs.assert_session_unchanged(key)
            if state["session_key"] != key:
                if state["session_key"] is not None:
                    raise RuntimeError("Session changed before completion")
                state["session_key"] = key
                state["cache_rows"] = 0
                state["cache_sha256"] = None
                state["stream"] = make_stream(*key, identity, config["decoder"])
                if v412:
                    state["augmentation_identity"] = batch["augmentation_identity"]
                    state["augmentation_description"] = batch["augmentation_description"]
            if v412 and state.get("augmentation_identity") != batch["augmentation_identity"]:
                raise RuntimeError("Session augmentation changed across chunks or resume")
            if v413 and epoch >= training["natural_replay_from_epoch"] and (replay_payload is None or
                    replay_payload["binding"]["session"] != key[1] or replay_payload["binding"]["subject"] != key[0]):
                from bme_eating.joint_v413_replay import replay_session
                stage = time.monotonic()
                replay_payload, state["natural_replay"] = replay_session(model, inputs, config, preprocessing,
                    natural_dataset, sampler, key, directory / "natural_replay", identity, epoch,
                    state["update_count"], stored=state.get("natural_replay"))
                chunk_timing["natural_replay_seconds"] = time.monotonic() - stage
            cache_path = directory / "cache" / f"epoch_{epoch}" / stable_id(*key)
            cache_binding = {"identity": identity, "epoch": epoch, "update": state["update_count"],
                             "subject": key[0], "session": key[1]}
            if active_cache is None or active_cache.path != cache_path:
                active_cache = TimelineCache(cache_path, cache_binding, capabilities(config).feature_width,
                    state["cache_rows"], committed_sha256=state.get("cache_sha256"))
            if active_cache.binding != cache_binding or active_cache.rows != state["cache_rows"]:
                raise RuntimeError("Active training cache identity or cursor changed")
            cache = active_cache
            history_times, history_features = cache.tensors()
            cpu_batch = batch
            batch = {name: value.to(device, non_blocking=device.type == "cuda")
                     if isinstance(value, torch.Tensor) else value
                     for name, value in batch.items()}
            mask = torch.zeros_like(batch["supervision_mask"], dtype=torch.bool)
            mask[:, -new_steps:] = True
            current_times = batch["timestamp_ms"][0, mask[0]]
            schedule = loss_schedule(epoch)
            if v411:
                schedule["quality_ranking"] = schedule["quality"]
            rng = np.random.default_rng(2026 + epoch * 1_000_000 + chunk_index)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                stage = time.monotonic()
                outputs = model(batch)
                features = pack_features(outputs, mask[0])
                times = torch.cat((history_times, current_times.cpu()))
                full_features = (FeatureBlocks([history_features, features.cpu()]) if v411 else
                                 torch.cat((history_features, features.cpu())))
                chunk_timing["encoder_seconds"] = time.monotonic() - stage
                stage = time.monotonic()
                if v413 and replay_payload is not None:
                    from bme_eating.joint_v413_replay import release_candidates
                    candidates = copy.deepcopy(release_candidates(replay_payload, state["natural_replay"], int(current_times[-1])))
                else:
                    candidates = generate_chunk(state["stream"], current_times, features)
                    if last:
                        candidates.extend(state["stream"].finish())
                state["epoch_candidates"].extend(copy.deepcopy(candidates))
                previous = int(history_times[-1]) if len(history_times) else int(current_times[0]) - 3000
                candidates.extend(truth_jitter_candidates(truth, *key, int(current_times[-1]),
                    previous, schedule["truth_jitter"], identity, rng))
                sampling_features = (replay_payload["features"][:len(times)] if v413 and replay_payload is not None else full_features)
                background_options = ({"maximum": training["random_background_families"],
                    "durations": training["random_background_seconds"]} if v413 else {})
                candidates.extend(random_background_candidates(truth, dataset.ignore_events, *key,
                    times, sampling_features, identity, rng, **background_options))
                if v413:
                    from bme_eating.joint_v413_replay import naturalize_jitter
                    natural = (replay_payload["candidates"][:state["natural_replay"]["cursor"]]
                               if replay_payload is not None else [])
                    naturalize_jitter(candidates, times, sampling_features, natural)
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
                if v411:
                    chunk_timing["proposal_seconds"] = time.monotonic() - stage
                    stage = time.monotonic()
                    if v412:
                        from bme_eating.joint_v412_training import sample_groups
                        sampling_options = {"performance": chunk_timing} if v413 else {}
                        candidate_groups, sample_report = sample_groups(model, candidates, truth, config, rng,
                            state["positive_gradients"], state["family_contributions"], times,
                            full_features, device, schedule["truth_jitter"], **sampling_options)
                    else:
                        from bme_eating.joint_v411_training import sample_groups
                        candidate_groups, sample_report = sample_groups(candidates, truth, config, rng,
                            state["positive_group_counts"], schedule["truth_jitter"])
                    chunk_timing["sampling_seconds"] = time.monotonic() - stage
                    candidates = [candidate for group in candidate_groups for candidate in group["versions"]]
                else:
                    candidates, sample_report = sample_candidates(candidates, truth,
                        training["candidates_per_chunk"], training["sample_composition"], rng,
                        truth_jitter_fraction=schedule["truth_jitter"])
                state["sample_audit"].append({"epoch": epoch, "chunk": chunk_index,
                    "subject": key[0], "session": key[1], **sample_report})
                if v412:
                    state["sample_audit"][-1]["augmentation"] = batch["augmentation_description"]
                target = batch["state_target"].float()
                valid = mask.float() * batch["state_loss_mask"] * (outputs["sensor_valid"] > 0)
                def masked_bce(logit, label, weight, count=None):
                    return (F.binary_cross_entropy_with_logits(logit.float(), label, reduction="none")
                            * weight).sum() / (weight.sum() if count is None else count.sum()).clamp_min(1e-8)
                tasks = {"state": masked_bce(outputs["state_logit"], target, valid,
                    batch["state_session_count"] if v413 else None)}
                proposals = []
                for scale, (logits, scale_valid) in outputs["proposal_scales"].items():
                    factor = int(scale) // 3
                    proposal_options = {"fractional": True} if v413 else {}
                    scale_target, scale_mask = proposal_supervision(batch, cpu_batch, valid, scale_valid, factor,
                                                                     **proposal_options)
                    proposals.append(masked_bce(logits, scale_target, scale_mask,
                        batch["proposal_" + scale + "_session_count"] if v413 else None))
                tasks["proposal"] = torch.stack(proposals).mean()
                tasks["transitions"] = sum(masked_bce(outputs[name + "_logit"],
                    batch[name + "_target"], mask.float() * batch[name + "_loss_mask"])
                    for name in ("onset", "offset"))
                if v412:
                    tasks["transitions"] = sum(heatmap_focal_loss(outputs[name + "_logit"],
                        batch[name + "_target"], mask.float() * batch[name + "_loss_mask"]
                        * batch["supervision_mask"] * (outputs["sensor_valid"] > 0),
                        batch[name + "_positive_count"], batch[name + "_background_count"])
                        for name in ("onset", "offset")) / (2 if v413 else 1)
                smooth_mask = valid[:, 1:] * valid[:, :-1] * batch["smooth_mask"][:, 1:] * batch["smooth_loss_mask"][:, 1:]
                if v413:
                    previous_valid = batch["state_loss_mask"] * (outputs["sensor_valid"] > 0)
                    smooth_mask = mask[:, 1:] * previous_valid[:, 1:] * previous_valid[:, :-1] * batch["smooth_mask"][:, 1:] * batch["smooth_loss_mask"][:, 1:]
                tasks["smooth"] = ((outputs["state_logit"][:, 1:].sigmoid()
                    - outputs["state_logit"][:, :-1].sigmoid()).square() * smooth_mask).sum() / (batch["smooth_session_count"].sum() if v413 else smooth_mask.sum()).clamp_min(1e-8)
                event_losses, quality_losses, boundary_losses = [], [], []
                negative_event_losses, negative_quality_losses = [], []
                for candidate in ([] if v411 else candidates):
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
                positive_group_losses = []
                gradient_strata = {}
                if v411:
                    from bme_eating.joint_v411_training import group_losses
                    scoring_options = {"performance": chunk_timing} if v413 else {}
                    grouped, label_changes = group_losses(model, candidate_groups, truth, times,
                        full_features, device, state["boundary"], config, schedule, **scoring_options)
                    negative_tasks = {name: [] for name in ("event", "quality", "boundary", "quality_ranking")}
                    used_groups = []
                    bank_groups = []
                    for group, group_tasks in grouped:
                        if v413:
                            if group["category"] != "positive":
                                if (key, group["family"]) in state["family_contributions"]:
                                    continue
                                state["family_contributions"].add((key, group["family"]))
                            bank_tasks = {"classification": group_tasks["event"] * config["loss"]["event"] * schedule["event"]
                                + group_tasks["quality"] * config["loss"]["quality"] * schedule["quality"]}
                            for name in ("boundary", "quality_ranking"):
                                if float(group_tasks[name].detach()) > 0:
                                    bank_tasks[name] = group_tasks[name] * config["loss"][name] * schedule.get(name, schedule["event"])
                            bank_tasks = {name: value * contribution_weights[key[0]] for name, value in bank_tasks.items()}
                            bank_groups.append((group, bank_tasks))
                            if group["category"] == "positive":
                                positive_group_losses.append((group, sum(bank_tasks.values())))
                            else:
                                negative_tasks["event"].append(group_tasks["event"])
                            used_groups.append(group)
                            continue
                        if group["category"] == "positive":
                            positive_group_losses.append((group if v412 else group["event_key"], sum(
                                value * config["loss"][name] * schedule.get(name, schedule["event"])
                                for name, value in group_tasks.items()) * contribution_weights[key[0]]))
                        elif (key, group["family"]) not in state["family_contributions"]:
                            state["family_contributions"].add((key, group["family"]))
                            for name, value in group_tasks.items():
                                negative_tasks[name].append(value)
                        else:
                            continue
                        used_groups.append(group)
                        category = group["category"]
                        stratum_loss = sum(value * config["loss"][name] * schedule.get(name, schedule["event"])
                                           for name, value in group_tasks.items()) * contribution_weights[key[0]] / training["accumulate_sessions"]
                        gradient_strata[category] = gradient_strata.get(category, zero) + stratum_loss
                    for name, values in negative_tasks.items():
                        tasks[name] = (torch.stack(values).mean() * new_steps / session_lengths[key]
                                       if values and not v413 else zero)
                    state["sample_audit"][-1].update({"refined_label_changes": label_changes,
                        "positive_loss_groups": len(positive_group_losses),
                        "negative_loss_groups": len(negative_tasks["event"]),
                        "actual_loss_versions": sum(len(group["versions"]) for group in used_groups),
                        "actual_loss_proposal_ids": [item.proposal_id for group in used_groups for item in group["versions"]],
                        "positive_group_version_mean": True,
                        "positive_session_event_mean_at_commit": True})
                state["sample_audit"][-1]["training_weights"] = {
                    "subject": contribution_weights[key[0]],
                    "valid_timeline": new_steps / session_lengths[key], "positive_event": 1 / session_events[key],
                    "negative_family": new_steps / session_lengths[key] / max(len(negative_event_losses), 1),
                    "natural_calibration": 1.0}
                if v411:
                    state["sample_audit"][-1]["training_weights"].update({
                        "positive_event": "1 / actual_supervised_session_events",
                        "positive_group": "1 / actual_supervised_groups_for_event",
                        "version": "1 / actual_unique_versions_for_group",
                        "accumulate_sessions": training["accumulate_sessions"], "schedule": schedule})
                if v412:
                    state["sample_audit"][-1]["training_weights"]["negative_family"] = (
                        new_steps / session_lengths[key] / max(len(negative_tasks["event"]), 1))
                event_ids = torch.full((new_steps,), -2, dtype=torch.long, device=device)
                phases = torch.zeros_like(event_ids)
                invalid = batch["state_loss_mask"][0, mask[0]] <= 0
                if v411:
                    invalid |= outputs["sensor_valid"][0, mask[0]] <= 0
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
                if v411 and schedule["contrastive"]:
                    from bme_eating.joint_v411_training import historical_contrastive
                    tasks["contrastive"], contrastive_pairs = historical_contrastive(model,
                        outputs["semantic_features"][0, mask[0]], event_ids, phases, history_times,
                        history_features, truth, *key, config, ignore=dataset.ignore_events)
                state["sample_audit"][-1]["contrastive_pairs"] = contrastive_pairs
                state["sample_audit"][-1]["contrastive_no_negative_or_positive_pairs"] = contrastive_pairs == 0
                weighted = {name: loss * config["loss"][name] * schedule.get(name, 1)
                            * (new_steps / session_lengths[key] if name not in {"event", "quality", "boundary", "quality_ranking"}
                               and not (v412 and name == "transitions") and not (v413 and name in {"state", "proposal", "smooth"}) else 1)
                            * contribution_weights[key[0]]
                            / training["accumulate_sessions"] for name, loss in tasks.items()}
                loss = sum(weighted.values())
                if v413:
                    state["sample_audit"][-1]["task_bank_normalization"] = {
                        "pure_negative": "session_actual_family_mean",
                        "classification": "contributing_session_mean_then_class_mean",
                        "boundary_and_ranking": "legal_groups_events_sessions"}
                    state["sample_audit"][-1]["training_weights"].update({
                        "valid_timeline": "1 / actual_valid_session_supervision",
                        "negative_family": "1 / actual_supervised_session_negative_families",
                        "classification_sessions": "contributing_sessions_then_present_classes_mean"})
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite joint loss")
            if chunk_index % training["gradient_diagnostics_every_chunks"] == 0:
                diagnostic_tasks = dict(weighted)
                if v411 and positive_group_losses:
                    diagnostic_tasks["positive_event"] = sum(value for _, value in positive_group_losses) / training["accumulate_sessions"]
                if v411:
                    diagnostic_tasks.update({"stratum_" + name: value for name, value in gradient_strata.items()})
                if v413:
                    for group, bank_tasks in bank_groups:
                        category = group["category"]
                        diagnostic_tasks["stratum_" + category] = diagnostic_tasks.get(
                            "stratum_" + category, features.sum() * 0) + sum(bank_tasks.values())
                report = gradient_diagnostics(diagnostic_tasks, model)
                state["gradient_diagnostics"].append({"epoch": epoch, "chunk": chunk_index,
                    "contrastive_pairs": contrastive_pairs, **report})
            if v413:
                buffer = PositiveGradientBuffer(state["positive_gradients"])
                bank = ContributionBank(state)
                for group, bank_tasks in bank_groups:
                    if group["category"] == "positive":
                        change = buffer.add(group, bank_tasks, model)
                        state["sample_audit"][-1].setdefault("positive_retention", []).append(change)
                    else:
                        bank.negative(bank_tasks["classification"], model)
            if v411 and not v413:
                buffer = PositiveGradientBuffer(state["positive_gradients"])
                for event_key, positive_loss in positive_group_losses:
                    change = buffer.add(event_key, positive_loss, model)
                    if v412:
                        state["sample_audit"][-1].setdefault("positive_retention", []).append(change)
            loss.backward()
            stage = time.monotonic()
            cache.append(current_times.detach().cpu().numpy(), features.detach().cpu().numpy())
            chunk_timing["cache_write_seconds"] = time.monotonic() - stage
            state["cache_rows"] = cache.rows
            state["cache_sha256"] = cache.checksum
            state["next_chunk"] = chunk_index + 1
            if last:
                if v411:
                    state["sample_audit"][-1]["positive_committed_events"] = len(state["positive_gradients"])
                    if v412:
                        state["sample_audit"][-1]["positive_committed_groups"] = {
                            str(event): [item["descriptor"] for item in groups.values()]
                            for event, groups in state["positive_gradients"].items()}
                    if v413:
                        ContributionBank(state).finish_session(state["positive_gradients"])
                        state["sample_audit"][-1]["actual_session_contributions"] = state["last_session_contributions"]
                        state["natural_replay"] = None
                        replay_payload = None
                    else:
                        PositiveGradientBuffer(state["positive_gradients"]).finish(model, training["accumulate_sessions"])
                state["session_count"] += 1
                state["session_key"], state["stream"] = None, None
                state["cache_rows"] = 0
                state["cache_sha256"] = None
                active_cache = None
                if state["session_count"] == training["accumulate_sessions"] or chunk_index == len(sampler.chunks) - 1:
                    if state["session_count"] < training["accumulate_sessions"]:
                        factor = training["accumulate_sessions"] / state["session_count"]
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(factor)
                    if v413:
                        ContributionBank(state).commit(model)
                        state["sample_audit"][-1]["actual_update_contributions"] = state["last_update_contributions"]
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"],
                                                         error_if_nonfinite=True)
                    state["clipped_steps"] += int(float(norm) > training["gradient_clip_norm"])
                    state["optimizer_steps"] += 1
                    state["maximum_gradient_norm"] = max(
                        state.get("maximum_gradient_norm", 0), float(norm))
                    clipping_report = epoch_clipping_report(
                        epoch, state["clipped_steps"], state["optimizer_steps"],
                        expected_optimizer_steps, training["clipping_abort_fraction"])
                    if clipping_report["abort_failed"]:
                        write_json_atomic(directory / "clipping_failure_report.json", {
                            **clipping_report, "epoch_complete": False, "promotion_eligible": False,
                            "pre_clip_gradient_norm": float(norm), "failed_chunk": chunk_index,
                            "committed_next_chunk": chunk_index, "identity": identity,
                            "training_subjects": sorted(subjects),
                            "excluded_subjects": sorted((outer_excluded or set()) | (selector or set())),
                            "config_sha256": bound_config, "source_sha256": digest(bound_source),
                            "data_sha256": digest(inputs.identity)})
                        raise RuntimeError(
                            "Joint clipping abort gate failed: "
                            f"{state['clipped_steps']}/{expected_optimizer_steps} planned updates clipped; "
                            "minimum full-epoch fraction "
                            f"{clipping_report['minimum_epoch_clipping_fraction']:.4f} "
                            f"> {training['clipping_abort_fraction']:.4f}")
                    if v412:
                        apply_learning_rate(optimizer, state, training,
                            epoch - 1 + (state["optimizer_steps"] - 1) / expected_optimizer_steps)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    state["update_count"] += 1
                    state["session_count"] = 0
            usage = enforce_resources(config, device)
            if v411:
                state["sample_audit"][-1].update({"elapsed_seconds": time.monotonic() - chunk_started,
                    "cache_write_bytes": len(current_times) * (8 + capabilities(config).feature_width * 4),
                    "positive_buffer_events": len(state["positive_gradients"]), "resources": usage,
                    "workers": training["num_workers"], "prefetch": training["loader_prefetch_factor"],
                    "reader_cache": training["session_reader_cache_size"]})
                if v413:
                    chunk_timing["previous_checkpoint_total_seconds"] = state.get("previous_checkpoint_total_seconds", 0.0)
                    state["sample_audit"][-1]["stage_timing"] = chunk_timing
            if hasattr(inputs, "assert_session_unchanged"):
                inputs.assert_session_unchanged(key)
            checkpoint()
            displayed_loss = float(loss.detach())
            if v413:
                displayed_loss += sum(float(value.detach()) for _, tasks in bank_groups for value in tasks.values())
            chunk_progress.set_postfix(loss=f"{displayed_loss:.5f}", subject=key[0],
                                       refresh=False)
        chunk_progress.close()
        if state["optimizer_steps"] != expected_optimizer_steps:
            raise RuntimeError("Completed joint epoch optimizer count differs from session traversal")
        clipping_report = epoch_clipping_report(
            epoch, state["clipped_steps"], state["optimizer_steps"], expected_optimizer_steps,
            training["clipping_abort_fraction"])
        fraction = clipping_report["clipping_fraction"]
        if v411:
            def restore_candidate(payload):
                payload = dict(payload)
                payload.pop("source_flags", None)
                return JointCandidate(**payload)
            state["epoch_candidates"] = [restore_candidate(record["candidate"]) for record in
                journals["candidate_history"].records() if record["epoch"] == epoch]
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
        if v411:
            state["boundary"].update({"qualified": epoch >= settings["train_from_epoch"]
                and config.get("ablation", {}).get("name") != "no_boundary", "mode": "both",
                "training_qualification_separate_from_oof_enablement": True})
            state["epoch_candidates"] = []
        state["epoch_complete"] = True
        qualification = epoch >= training["checkpoint_min_epoch"] and fraction <= training["clipping_promotion_fraction"]
        epoch_payload = {"model": model.state_dict(), "identity": identity, "epoch": epoch,
            "epoch_complete": True, "promotion_eligible": qualification, "clipping_fraction": fraction,
            "training_subjects": sorted(subjects),
            "excluded_subjects": sorted((outer_excluded or set()) | (selector or set())),
            "preprocessing": preprocessing, "boundary": state["boundary"],
            "config_sha256": bound_config, "source_sha256": digest(bound_source),
            "data_sha256": digest(inputs.identity)}
        if v412:
            epoch_payload.update({"lr_position": state.get("lr_position", 0),
                                  "learning_rates": state.get("learning_rates", [])})
        if state.get("resume_migration"):
            epoch_payload["resume_migration"] = state["resume_migration"]
        checkpoint()
        if not tensors_finite(epoch_payload["model"].values()):
            raise RuntimeError("Epoch checkpoint finite gate failed")
        if deployment_epoch is None or epoch <= deployment_epoch:
            save_torch_atomic(directory / "latest.pt", epoch_payload)
        if deployment_epoch is not None and epoch in {
                deployment_epoch - 4, deployment_epoch - 2, deployment_epoch,
                deployment_epoch + 2, deployment_epoch + 4}:
            save_torch_atomic(directory / "nearby" / f"epoch_{epoch:03d}.pt", epoch_payload)
        record = {"epoch": epoch, "clipping_fraction": fraction, "promotion_eligible": qualification,
                  "chunks": len(sampler.chunks), "sessions": len(session_lengths),
                  "optimizer_steps": state["optimizer_steps"], "candidate_coverage": coverage,
                  "clipping_gate": clipping_report,
                  "maximum_gradient_norm": state.get("maximum_gradient_norm", 0),
                  "elapsed_seconds": time.monotonic() - started, **enforce_resources(config, device)}
        if v412:
            record.update({"lr_position": state.get("lr_position", 0),
                           "learning_rates": state.get("learning_rates", [])})
        stop = False
        if selector and epoch >= training["checkpoint_min_epoch"] and epoch % training["validation_every_epochs"] == 0:
            if v411:
                from bme_eating.data.joint_sessions import release_session_loader
                release_session_loader(loader)
                loader = None
            assert_excluded(subjects, selector, outer_excluded or set())
            fitted_mapping = fit_source_mapping(model, inputs, subjects, config, preprocessing,
                                                directory / "source_fit" / str(epoch), identity)
            predictions = predict_joint_model(model, inputs, selector, config, preprocessing,
                                               directory / "selector" / str(epoch), identity,
                                               {**state["boundary"], **({"search_eligible": False} if v412 else {})},
                                               mapping=fitted_mapping)
            _, _, selector_events, _ = inputs.frames(selector, labels=True)
            selector_truth, selector_ignore = partition_evaluation_events(selector_events, selector)
            point, search_records = choose_operating_point(predictions["coarse"], predictions["windows"],
                selector_truth, selector_ignore, {identity: fitted_mapping}, config)
            record["selector"] = point
            print_metrics(f"selector epoch {epoch}/{epochs}", point)
            write_json_atomic(directory / "selector" / str(epoch) / "search.json", search_records)
            f1 = point["f1"]
            eligible = qualification and point["promotion_eligible"]
            if qualification and f1 > state["diagnostic_f1"]:
                state["diagnostic_epoch"], state["diagnostic_f1"] = epoch, f1
                save_torch_atomic(directory / "diagnostic_selected.pt", {
                    **epoch_payload, "promotion_eligible": False, "diagnostic_only": True})
            if v413:
                improved, stop = selection_update(state, epoch, eligible, f1, training)
                if improved:
                    save_torch_atomic(directory / "selected.pt", epoch_payload)
                record["selection_state"] = {name: state.get(name) for name in (
                    "best_epoch", "stale_checks", "unqualified_checks", "lost_qualification_checks", "selection_failure")}
            else:
                if eligible and (state["best_epoch"] is None or f1 > state["best_f1"] + training["f1_min_delta"]):
                    state["best_epoch"], state["best_f1"], state["stale_checks"] = epoch, f1, 0
                    save_torch_atomic(directory / "selected.pt", epoch_payload)
                else:
                    state["stale_checks"] += 1
                stop = state["stale_checks"] >= training["patience_checks"]
            print(f"[selection] best_epoch={state['best_epoch']} best_F1={state['best_f1']:.4f} "
                  f"stale_checks={state['stale_checks']}/{training['patience_checks']} "
                  f"clipping={fraction:.4f} early_stop={stop} "
                  f"unqualified_checks={state.get('unqualified_checks', 0)} "
                  f"lost_qualification_checks={state.get('lost_qualification_checks', 0)}", flush=True)
        if v411:
            journals["training_history"].append([record])
            state["journal_offsets"]["training_history"] = journals["training_history"].offset
            if "selector" in record:
                record = {**record, "selector": {name: value for name, value in record["selector"].items()
                           if name != "budget_audit"}}
        state["history"].append(record)
        write_json_atomic(directory / "training_history.json", state["history"])
        write_json_atomic(directory / "sampling.json", state["sample_audit"])
        write_json_atomic(directory / "gradients.json", state["gradient_diagnostics"])
        state["epoch"] += 1
        state["next_chunk"] = 0
        state["sampler_identity"] = None
        state["epoch_complete"] = False
        state["clipped_steps"], state["optimizer_steps"] = 0, 0
        state["maximum_gradient_norm"] = 0
        state["event_contributions"], state["family_contributions"] = set(), set()
        if v411:
            state["positive_group_counts"] = {}
            if state["positive_gradients"]:
                raise RuntimeError("Positive gradients remain uncommitted after epoch completion")
        state["epoch_candidates"] = []
        checkpoint()
        if stop:
            break
    state["training_complete"] = True
    if v411:
        from bme_eating.data.joint_sessions import release_session_loader
        release_session_loader(loader)
    checkpoint()
    if selector:
        if state["best_epoch"] is None and state["diagnostic_epoch"] is None:
            raise RuntimeError("No finite/clipping-qualified joint epoch; diagnostic checkpoints preserved")
        if v413 and (state["best_epoch"] is None or state.get("selection_failure")):
            write_json_atomic(directory / "selector_failure_report.json", {
                "promotion_eligible": False, "diagnostic_epoch": state["diagnostic_epoch"],
                "reason": state.get("selection_failure") or "No qualified checkpoint at maximum epoch"})
            raise RuntimeError("v413 selection failed; diagnostic weights cannot enter retraining")
        if state["best_epoch"] is None:
            write_json_atomic(directory / "selector_failure_report.json", {
                "promotion_eligible": False, "diagnostic_epoch": state["diagnostic_epoch"],
                "reason": "No feasible selector operating point"})
        return state["best_epoch"] or state["diagnostic_epoch"]
    return epochs


def decoder_settings(config):
    if supported(config, "stabilized"):
        return {f"high_{high:g}_low_{ratio:g}_gap_{gap}":
                {**config["decoder"], "high_threshold": high, "low_ratio": ratio,
                 "gap_merge_seconds": gap}
                for high, ratio, gap in product(config["search"]["high_thresholds"],
                    config["search"]["low_ratios"], config["search"]["gap_seconds"])}
    return {f"high_{high:g}_gap_{gap}": {**config["decoder"], "high_threshold": high,
                                         "gap_merge_seconds": gap}
            for high, gap in product(config["search"]["high_thresholds"], config["search"]["gap_seconds"])}


@torch.no_grad()
def predict_joint_model(model, inputs, subjects, config, preprocessing, directory, identity,
                        boundary, *, training_source: bool = False, mapping=None, operating_point=None,
                        development_calibration=None):
    if supported(model, "grouped"):
        from bme_eating.joint_v411_prediction import predict
        return predict(model, inputs, subjects, config, preprocessing, directory, identity, boundary,
                       source_only=training_source, mapping=mapping, operating_point=operating_point,
                       development_calibration=development_calibration)
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
    if supported(model, "grouped"):
        from bme_eating.joint_v411_prediction import fit_mapping
        return fit_mapping(model, inputs, subjects, config, preprocessing, directory, identity)
    outputs = predict_joint_model(model, inputs, subjects, config, preprocessing, directory, identity,
                                   {"qualified": False}, training_source=True)
    candidates = [candidate for pool in outputs["coarse"].values() for candidate in pool]
    mapping = FrozenSourceCDF.fit(candidates, subjects)
    write_json_atomic(directory / "source_mapping.json", mapping.to_json())
    return mapping


def load_trained(directory, config, identity, subjects, data_identity, *, checkpoint_name="latest.pt"):
    if checkpoint_name not in {"latest.pt", "selected.pt"}:
        raise ValueError("Unregistered checkpoint role")
    payload = torch.load(directory / checkpoint_name, map_location=config["training"]["device"], weights_only=True)
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
                       refined_candidates=None, mode="both"):
    coarse_metrics, coarse_match = metrics_with_exposure(truth, coarse, ignore, windows)
    refined_metrics, refined_match = metrics_with_exposure(truth, refined, ignore, windows)
    event_keys = ["subject_key", "session_id", "event_id"]
    keys = [*event_keys, "prediction_event_id"]
    proposal_keys = ["subject_key", "session_id", "prediction_event_id"]
    before = set(map(tuple, coarse_match[proposal_keys].values)) if len(coarse_match) else set()
    after = set(map(tuple, refined_match[proposal_keys].values)) if len(refined_match) else set()
    v411 = supported(config, "grouped")
    paired = coarse_match.merge(refined_match, on=event_keys if v411 else keys,
                                suffixes=("_coarse", "_refined")) if before and after else pd.DataFrame()
    improvements = {}
    for endpoint in ("start", "end"):
        old = float(paired[endpoint + "_absolute_error_ms_coarse"].mean()) if len(paired) else float("nan")
        new = float(paired[endpoint + "_absolute_error_ms_refined"].mean()) if len(paired) else float("nan")
        improvements[endpoint] = (old - new) / old if old > 0 else 0
    truth_before = set(map(tuple, coarse_match[event_keys].values)) if len(coarse_match) else set()
    truth_after = set(map(tuple, refined_match[event_keys].values)) if len(refined_match) else set()
    from bme_eating.joint_evaluation import event_frame
    refined_frame = event_frame(refined)
    refined_keys = ["subject_key", "session_id", "proposal_id"]
    emitted = set(map(tuple, refined_frame[refined_keys].values))
    ignored = prediction_ignore_mask(refined_frame, ignore)
    false_positive_ids = set(map(tuple, refined_frame.loc[~ignored, refined_keys].values)) - after
    conversions = before & false_positive_ids if v411 else before - after
    tp_to_fp = len(conversions) / max(len(before), 1)
    population = {candidate.proposal_id: candidate for candidate in
                  (refined_candidates if refined_candidates is not None else refined)}
    population.update({candidate.proposal_id: candidate for candidate in refined})
    population = list(population.values())
    clipping = sum(candidate.boundary_clipped for candidate in population) / max(len(population), 1)
    conflicts = sum(candidate.boundary_conflict for candidate in population)
    gates = config["gates"]
    positive_events = len(paired[event_keys].drop_duplicates()) if len(paired) else 0
    passed = (positive_events >= (config["boundary"]["minimum_positive_events"] if require_sixty else 1)
              and all(improvements[name] >= gates["boundary_minimum_mae_improvement"]
                      for name in (("start",) if mode == "start_only" else ("start", "end")))
              and coarse_metrics["f1"] - refined_metrics["f1"] <= gates["boundary_maximum_f1_drop"]
              and tp_to_fp <= gates["boundary_maximum_tp_to_fp_rate"]
              and clipping <= gates["boundary_maximum_clipping_rate"])
    if mode == "start_only":
        passed &= all(candidate.end_ms == candidate.unrefined_end_ms for candidate in population
                      if candidate.unrefined_end_ms is not None)
    from scipy.stats import beta
    failures = len(conversions)
    denominator = len(before)
    interval = [float(beta.ppf(.025, failures, denominator - failures + 1)) if failures else 0,
                float(beta.ppf(.975, failures + 1, denominator - failures)) if failures < denominator else 1]
    return {"passed": bool(passed), "boundary_mode": mode,
            "paired_positive_events": positive_events, "improvements": improvements,
            "tp_to_fp_count": failures, "tp_to_fp_denominator": denominator,
            "tp_to_fp_exact_95_interval": interval,
            "tp_to_fp_basis": ("coarse_matched_prediction_id_to_refined_false_positive" if v411
                               else "coarse_matched_prediction_id_not_refined_matched"),
            "tp_prediction_loss_count": len(before - after),
            "dropped_tp_prediction_count": len(before - emitted),
            "ignored_tp_prediction_count": len((before & emitted) - after - false_positive_ids),
            "lost_truth_count": len(truth_before - truth_after),
            "truth_retention_rate": len(truth_before & truth_after) / max(len(truth_before), 1),
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
    executed = internal_partitions(config)
    boundary_search_eligible = True
    if supported(config, "consistent_inputs"):
        calibration_subjects = {subject for subject, partition in partitions.items() if partition in executed}
        _, _, calibration_events, _ = inputs.frames(calibration_subjects, labels=True)
        calibration_truth, _ = partition_evaluation_events(calibration_events, calibration_subjects)
        boundary_search_eligible = len(calibration_truth) >= config["boundary"]["minimum_positive_events"]
    for partition in executed:
        print(f"[internal] {directory.parent.name} partition {partition + 1}/{len(executed)} "
              f"({calibration_scope(config)['calibration_kind']})", flush=True)
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
            partition_root / "prediction", identity,
            {**payload["boundary"], "search_eligible": boundary_search_eligible}, mapping=mapping)
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
    calibration_subjects = {subject for lineage in lineages for subject in lineage["prediction_subjects"]}
    _, _, events, _ = inputs.frames(calibration_subjects, labels=True)
    truth, ignore = partition_evaluation_events(events, calibration_subjects)
    point, search = choose_operating_point(coarse, windows, truth, ignore, mappings, config)
    coarse_point = copy.deepcopy(point)
    natural, _ = apply_online_budget(coarse[point["setting"]], windows, mappings, point["budget"],
                                     config["decoder"]["quotas"])
    calibrator = NonnegativePlatt(**point["calibration"])
    coarse_predictions = commit_candidates(natural, calibrator, point["acceptance"], point["nms"])
    boundary_reports, enabled, best_boundary_f1, best_entropy, winning_point = [], False, -1.0, None, None
    v411 = supported(config, "grouped")
    if v411:
        from bme_eating.joint_v411_boundary import lock_boundary_modes
        point, boundary_reports, search_boundary = lock_boundary_modes(
            point, natural, coarse_predictions, outputs, truth, ignore, windows, config)
        search.extend(search_boundary)
        enabled, best_entropy = point["boundary_enabled"], point["boundary_entropy"]
    for entropy in ([] if v411 else config["boundary"]["entropy_thresholds"]):
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
    point.update({**calibration_scope(config), "calibration_subjects": sorted(calibration_subjects)})
    verify_calibration_lineage(lineages, subjects, outer, config,
                               identities=protocol_identities(config, inputs.identity))
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
    write_json_atomic(directory / "internal_manifest.json", {"protocol": protocol_for(config),
        "run_identity": run_identity, "training_subjects": sorted(subjects),
        "outer_excluded_subjects": sorted(outer), **protocol_identities(config, inputs.identity),
        **calibration_scope(config), "calibration_subjects": sorted(calibration_subjects),
        "evidence_sha256": evidence})
    return int(np.median(epochs)), point
