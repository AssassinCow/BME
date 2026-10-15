from __future__ import annotations

import copy
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

from bme_eating.artifacts import write_json_atomic
from bme_eating.candidates import (
    JointCandidate,
    NoSourceCandidatesError,
    StreamingProposals,
    TimelineCache,
    candidate_targets,
    geometry_key,
    stable_id,
)
from bme_eating.data.joint_sessions import (
    JointInputs,
    SessionChunkSampler,
    make_joint_dataset,
    release_loaders_after,
    reuse_session_loader,
    session_loader,
)
from bme_eating.evaluation import (
    candidate_coverage,
    choose_operating_point,
    metrics_with_exposure,
)
from bme_eating.metrics import partition_evaluation_events, prediction_ignore_mask
from bme_eating.models.statsfusion_joint_event import (
    gaussian_distribution,
    loss_schedule,
)
from bme_eating.progress import print_metrics, progress
from bme_eating.protocol import (
    CHECKPOINT_STATE_VERSION,
    assert_excluded,
    capabilities,
    checkpoint_qualified,
    config_identity,
    configure_numerics,
    digest,
    protocol_for,
    save_torch_atomic,
    supported,
    training_source_identity,
)


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
    from bme_eating.models.statsfusion_joint_event import build_joint_model
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
    from bme_eating.resources import private_process_tree
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
    from bme_eating.candidate_scoring import score_candidates
    return score_candidates(model, [candidate], times, features, device)[0]


def refine_candidate(model, candidate, times, features, device, boundary, entropy):
    from bme_eating.candidate_scoring import refine
    return refine(model, candidate, times, features, device, boundary, entropy)


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
            from bme_eating.evaluation import event_frame
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


def _training_result(state, selector, epochs, directory):
    if not selector:
        return epochs
    if state["best_epoch"] is None and state["diagnostic_epoch"] is None:
        raise RuntimeError("No finite/clipping-qualified joint epoch; diagnostic checkpoints preserved")
    if state["best_epoch"] is None or state.get("selection_failure"):
        write_json_atomic(directory / "selector_failure_report.json", {
            "promotion_eligible": False, "diagnostic_epoch": state["diagnostic_epoch"],
            "reason": state.get("selection_failure") or "No qualified checkpoint at maximum epoch"})
        raise RuntimeError("Joint selection failed; diagnostic weights cannot enter retraining")
    return state["best_epoch"]


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
    from bme_eating.feature_cache import FeatureBlocks
    from bme_eating.journal import CheckedJournal
    from bme_eating.proposal_stream import make_stream
    from bme_eating.training.candidate_supervision import (
        ContributionBank,
        PositiveGradientBuffer,
        apply_learning_rate,
        selection_update,
        subject_weights,
    )
    source_root = Path(config["_config_path"]).resolve().parents[1]
    bound_source = training_source_identity(source_root, config)
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
             "training_complete": False, "stop_requested": False,
             "stop_reason": None, "completed_epoch": 0}
    state.update({"positive_gradients": {}, "positive_group_counts": {},
                  "journal_offsets": {}, "contrastive_history": None})
    ContributionBank(state)
    state.update({"natural_replay": None, "unqualified_checks": 0, "lost_qualification_checks": 0})
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (payload.get("state_version") != CHECKPOINT_STATE_VERSION
                or payload.get("protocol") != protocol_for(config)):
            raise RuntimeError("Resume uses an incompatible training state/protocol; create a new run")
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
        if payload.get("source_sha256") != digest(bound_source):
            raise RuntimeError("Resume source changed; create a new run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        for name, parameter in model.named_parameters():
            gradient = payload["gradients"].get(name)
            parameter.grad = gradient.to(device) if gradient is not None else None
        state = payload["state"]
        if config.get("_output_only_resume"):
            compatibility = config["_output_only_resume"]
            state["resume_migration"] = {"policy": compatibility["policy"],
                "training_source_sha256": digest(bound_source),
                "execution_source_sha256": digest(compatibility["execution_source"])}
        PositiveGradientBuffer(state["positive_gradients"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])
        if state.get("stop_requested") and not state.get("training_complete"):
            raise RuntimeError("Inconsistent terminal training state; refusing to continue")
        if state.get("training_complete"):
            return _training_result(state, selector, epochs, directory)
    natural_dataset = None
    replay_payload = None
    ContributionBank(state)
    natural_dataset, _ = make_joint_dataset(inputs, subjects, config, preprocessing, training=True)
    natural_dataset.training = False
    truth, _ = partition_evaluation_events(dataset.events, subjects)
    session_lengths = {key: len(group) for key, group in dataset.session_groups.items()}
    contribution_weights = subject_weights(session_lengths)
    count_dataset, _ = make_joint_dataset(inputs, subjects, config, preprocessing, training=True)
    state.setdefault("locked_geometries", set())
    state.setdefault("endpoint_balance", None)
    state.setdefault("pending_timeline_objective", 0.0)
    write_json_atomic(directory / "subject_weights.json", {
        "weights": contribution_weights,
        "session_counts": {subject: sum(key[0] == subject for key in session_lengths) for subject in sorted(subjects)},
        "natural_calibration_weights": True})
    session_events = {key: max(1, int(((truth.subject_key.astype(str) == key[0])
                        & (truth.session_id.astype(str) == key[1])).sum())) for key in session_lengths}
    started = time.monotonic()
    journals = {name: CheckedJournal(directory / (name + ".jsonl"), state["journal_offsets"].get(name, 0))
                for name in ("sampling", "gradients", "candidate_history", "training_history")}

    def checkpoint():
        checkpoint_started = time.monotonic()
        if config_identity(config) != bound_config:
            raise RuntimeError("Resolved training configuration changed before checkpoint")
        if source_signature() != bound_signature and training_source_identity(source_root, config) != bound_source:
            raise RuntimeError("Runtime source changed before checkpoint")
        if input_signature is not None and inputs.file_signature() != input_signature:
            raise RuntimeError("Runtime input data changed before checkpoint")
        finite = tensors_finite(model.state_dict().values())
        if not finite or not tensors_finite(parameter.grad for parameter in model.parameters()
                                            if parameter.grad is not None):
            raise RuntimeError("Finite gate failed before chunk checkpoint")
        positive_values = (PositiveGradientBuffer(state["positive_gradients"]).tensors())
        if not tensors_finite(ContributionBank(state).tensors()):
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
        state["checkpoint_precommit_seconds"] = time.monotonic() - checkpoint_started
        save_torch_atomic(checkpoint_path, training_checkpoint_snapshot({"identity": identity, "training_subjects": sorted(subjects),
            "protocol": protocol_for(config), "state_version": CHECKPOINT_STATE_VERSION,
            "requested_epochs": epochs, "deployment_epoch": deployment_epoch, "selector_subjects": sorted(selector or set()),
            "excluded_subjects": sorted((outer_excluded or set()) | (selector or set())),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
            "gradients": {name: parameter.grad.detach() if parameter.grad is not None else None
                          for name, parameter in model.named_parameters()},
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "config_sha256": bound_config, "source_sha256": digest(bound_source),
            "data_sha256": digest(inputs.identity), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}))
        state["previous_checkpoint_total_seconds"] = time.monotonic() - checkpoint_started

    loader = None
    while state["epoch"] <= epochs:
        active_cache = None
        epoch = state["epoch"]
        from bme_eating.training.endpoint_balance import epoch_balance
        balance = epoch_balance(count_dataset, epoch, contribution_weights, config)
        if state["endpoint_balance"] is not None and state["endpoint_balance"] != balance:
            raise RuntimeError("Endpoint count table differs from committed resume state")
        state["endpoint_balance"] = balance
        write_json_atomic(directory / "endpoint_counts" / f"epoch_{epoch:03d}.json", balance)
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
                state["locked_geometries"] = set()
                state["cache_rows"] = 0
                state["cache_sha256"] = None
                state["stream"] = make_stream(*key, identity, config["decoder"])
                state["augmentation_identity"] = batch["augmentation_identity"]
                state["augmentation_description"] = batch["augmentation_description"]
            if state.get("augmentation_identity") != batch["augmentation_identity"]:
                raise RuntimeError("Session augmentation changed across chunks or resume")
            if epoch >= training["natural_replay_from_epoch"] and (replay_payload is None or
                    replay_payload["binding"]["session"] != key[1] or replay_payload["binding"]["subject"] != key[0]):
                from bme_eating.training.natural_replay import replay_session
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
            schedule["quality_ranking"] = schedule["quality"]
            rng = np.random.default_rng(2026 + epoch * 1_000_000 + chunk_index)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                stage = time.monotonic()
                outputs = model(batch)
                features = pack_features(outputs, mask[0])
                times = torch.cat((history_times, current_times.cpu()))
                full_features = (FeatureBlocks([history_features, features.cpu()]))
                chunk_timing["encoder_seconds"] = time.monotonic() - stage
                stage = time.monotonic()
                if replay_payload is not None:
                    from bme_eating.training.natural_replay import release_candidates
                    candidates = copy.deepcopy(release_candidates(replay_payload, state["natural_replay"], int(current_times[-1])))
                else:
                    candidates = generate_chunk(state["stream"], current_times, features)
                    if last:
                        candidates.extend(state["stream"].finish())
                state["epoch_candidates"].extend(copy.deepcopy(candidates))
                previous = int(history_times[-1]) if len(history_times) else int(current_times[0]) - 3000
                candidates.extend(truth_jitter_candidates(truth, *key, int(current_times[-1]),
                    previous, schedule["truth_jitter"], identity, rng))
                sampling_features = (replay_payload["features"][:len(times)] if replay_payload is not None else full_features)
                background_options = ({"maximum": training["random_background_families"],
                    "durations": training["random_background_seconds"]})
                candidates.extend(random_background_candidates(truth, dataset.ignore_events, *key,
                    times, sampling_features, identity, rng, **background_options))
                from bme_eating.training.natural_replay import naturalize_jitter
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
                    from bme_eating.evaluation import event_frame
                    ignored = prediction_ignore_mask(event_frame(candidates), ignore)
                    candidates = [candidate for candidate, drop in zip(candidates, ignored)
                                  if not drop or candidate_targets(candidate, truth)[0] > 0]
                chunk_timing["proposal_seconds"] = time.monotonic() - stage
                stage = time.monotonic()
                from bme_eating.training.candidate_supervision import sample_groups
                sampling_options = {"performance": chunk_timing}
                candidate_groups, sample_report = sample_groups(model, candidates, truth, config, rng,
                    state["positive_gradients"], state["family_contributions"], times,
                    full_features, device, schedule["truth_jitter"],
                    locked_geometries=state["locked_geometries"], **sampling_options)
                chunk_timing["sampling_seconds"] = time.monotonic() - stage
                candidates = [candidate for group in candidate_groups for candidate in group["versions"]]
                state["sample_audit"].append({"epoch": epoch, "chunk": chunk_index,
                    "subject": key[0], "session": key[1], **sample_report})
                state["sample_audit"][-1]["natural_replay_duplicates"] = (
                    state["natural_replay"].get("last_duplicates", []) if replay_payload is not None else [])
                state["sample_audit"][-1]["augmentation"] = batch["augmentation_description"]
                target = batch["state_target"].float()
                valid = mask.float() * batch["state_loss_mask"] * (outputs["sensor_valid"] > 0)
                def masked_bce(logit, label, weight, count=None):
                    return (F.binary_cross_entropy_with_logits(logit.float(), label, reduction="none")
                            * weight).sum() / (weight.sum() if count is None else count.sum()).clamp_min(1e-8)
                tasks = {"state": masked_bce(outputs["state_logit"], target, valid,
                    batch["state_session_count"])}
                proposals = []
                for scale, (logits, scale_valid) in outputs["proposal_scales"].items():
                    factor = int(scale) // 3
                    proposal_options = {"fractional": True}
                    scale_target, scale_mask = proposal_supervision(batch, cpu_batch, valid, scale_valid, factor,
                                                                     **proposal_options)
                    proposals.append(masked_bce(logits, scale_target, scale_mask,
                        batch["proposal_" + scale + "_session_count"]))
                tasks["proposal"] = torch.stack(proposals).mean()
                from bme_eating.training.candidate_supervision import heatmap_focal_components
                endpoint_parts, endpoint_counts = {}, {}
                for name in ("onset", "offset"):
                    endpoint_mask = mask.float() * batch[name + "_loss_mask"] * batch["supervision_mask"] * (outputs["sensor_valid"] > 0)
                    parts = heatmap_focal_components(outputs[name + "_logit"], batch[name + "_target"],
                        endpoint_mask, batch[name + "_positive_count"], batch[name + "_background_count"],
                        peak_scale=balance["scales"][name]["peak_scale"],
                        background_scale=balance["scales"][name]["background_scale"])
                    endpoint_parts.update({name + "_" + term: value for term, value in parts.items()})
                    endpoint_counts[name] = {"peaks": int(((batch[name + "_target"] == 1) * endpoint_mask).sum()),
                        "background_bins": int(((batch[name + "_target"] < 1) * endpoint_mask).sum())}
                tasks["transitions"] = sum(endpoint_parts.values()) / 2
                previous_valid = batch["state_loss_mask"] * (outputs["sensor_valid"] > 0)
                smooth_mask = mask[:, 1:] * previous_valid[:, 1:] * previous_valid[:, :-1] * batch["smooth_mask"][:, 1:] * batch["smooth_loss_mask"][:, 1:]
                tasks["smooth"] = ((outputs["state_logit"][:, 1:].sigmoid()
                    - outputs["state_logit"][:, :-1].sigmoid()).square() * smooth_mask).sum() / (batch["smooth_session_count"].sum()).clamp_min(1e-8)
                zero = features.sum() * 0
                tasks.update({name: zero for name in ("event", "quality", "boundary", "quality_ranking")})
                positive_group_losses = []
                gradient_strata = {}
                from bme_eating.training.candidate_supervision import group_losses
                scoring_options = {"performance": chunk_timing}
                grouped, label_changes = group_losses(model, candidate_groups, truth, times,
                    full_features, device, state["boundary"], config, schedule, **scoring_options)
                negative_tasks = {name: [] for name in ("event", "quality", "boundary", "quality_ranking")}
                used_groups = []
                bank_groups = []
                for group, group_tasks in grouped:
                    if group["category"] != "positive":
                        if (key, group["family"]) in state["family_contributions"]:
                            continue
                        state["family_contributions"].add((key, group["family"]))
                    bank_tasks = {"classification": group_tasks["event"] * config["loss"]["event"] * schedule["event"]
                        + group_tasks["quality"] * config["loss"]["quality"] * schedule["quality"]}
                    if not (schedule["event"] or schedule["quality"]):
                        bank_tasks.clear()
                    for name in ("boundary", "quality_ranking"):
                        if group_tasks["_legal_auxiliary"][name] and schedule.get(name, schedule["event"]):
                            bank_tasks[name] = group_tasks[name] * config["loss"][name] * schedule.get(name, schedule["event"])
                    bank_tasks = {name: value * contribution_weights[key[0]] for name, value in bank_tasks.items()}
                    if not bank_tasks:
                        continue
                    atoms = []
                    for version in group_tasks["_versions"]:
                        atom_tasks = {"classification": (version["event"] * config["loss"]["event"] * schedule["event"]
                            + version["quality"] * config["loss"]["quality"] * schedule["quality"]) * contribution_weights[key[0]]}
                        if not (schedule["event"] or schedule["quality"]):
                            atom_tasks.clear()
                        if version["boundary_legal"] and schedule["boundary"]:
                            atom_tasks["boundary"] = version["boundary"] * config["loss"]["boundary"] * schedule["boundary"] * contribution_weights[key[0]]
                        atoms.append({"geometries": [version["geometry"]], "tasks": atom_tasks})
                    for pair in group_tasks["_pairs"] if schedule["quality"] else []:
                        atoms.append({"geometries": pair["geometries"], "tasks": {"quality_ranking":
                            pair["quality_ranking"] * config["loss"]["quality_ranking"] * schedule["quality"] * contribution_weights[key[0]]}})
                    if group["category"] == "positive":
                        bank_tasks["_atoms"] = atoms
                    bank_groups.append((group, bank_tasks))
                    if group["category"] == "positive":
                        positive_group_losses.append((group, sum(value for name, value in bank_tasks.items() if not name.startswith("_"))))
                    else:
                        negative_tasks["event"].append(group_tasks["event"])
                    used_groups.append(group)
                for name in negative_tasks:
                    tasks[name] = (zero)
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
                    "negative_family": "session_actual_supervised_families",
                    "natural_calibration": 1.0}
                state["sample_audit"][-1]["training_weights"].update({
                    "positive_event": "1 / actual_supervised_session_events",
                    "positive_group": "1 / actual_supervised_groups_for_event",
                    "version": "1 / actual_unique_versions_for_group",
                    "accumulate_sessions": training["accumulate_sessions"], "schedule": schedule})
                event_ids = torch.full((new_steps,), -2, dtype=torch.long, device=device)
                phases = torch.zeros_like(event_ids)
                invalid = batch["state_loss_mask"][0, mask[0]] <= 0
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
                tasks["contrastive"], contrastive_pairs = zero, 0
                contrastive_report = {"positive_anchors": 0, "history_positive_bins": 0,
                    "history_background_bins": 0, "valid_pairs": 0, "zero_reason": "inactive_warmup"}
                if schedule["contrastive"]:
                    from bme_eating.training.candidate_supervision import historical_contrastive
                    tasks["contrastive"], contrastive_pairs = historical_contrastive(model,
                        outputs["semantic_features"][0, mask[0]], event_ids, phases, history_times,
                        history_features, truth, *key, config, ignore=dataset.ignore_events,
                        diagnostics=contrastive_report)
                state["sample_audit"][-1]["contrastive_pairs"] = contrastive_pairs
                state["sample_audit"][-1]["contrastive_supervision"] = contrastive_report
                state["sample_audit"][-1]["contrastive_no_negative_or_positive_pairs"] = contrastive_pairs == 0
                weighted = {name: loss * config["loss"][name] * schedule.get(name, 1)
                            * (new_steps / session_lengths[key] if name not in {"event", "quality", "boundary", "quality_ranking"}
                               and not (name == "transitions") and name not in {"state", "proposal", "smooth"} else 1)
                            * contribution_weights[key[0]]
                            / training["accumulate_sessions"] for name, loss in tasks.items()}
                loss = sum(weighted.values())
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
            if not all(torch.isfinite(value).all() for _, candidate_tasks in bank_groups
                       for name, value in candidate_tasks.items() if not name.startswith("_")):
                raise RuntimeError("Nonfinite candidate task loss")
            if chunk_index % training["gradient_diagnostics_every_chunks"] == 0:
                diagnostic_tasks = dict(weighted)
                diagnostic_tasks.update({name: value * config["loss"]["transitions"] * .5
                    * contribution_weights[key[0]] / training["accumulate_sessions"]
                    for name, value in endpoint_parts.items()})
                if positive_group_losses:
                    diagnostic_tasks["positive_event"] = sum(value for _, value in positive_group_losses) / training["accumulate_sessions"]
                diagnostic_tasks.update({"stratum_" + name: value for name, value in gradient_strata.items()})
                for group, bank_tasks in bank_groups:
                    category = group["category"]
                    diagnostic_tasks["stratum_" + category] = diagnostic_tasks.get(
                        "stratum_" + category, features.sum() * 0) + sum(value for name, value in bank_tasks.items() if not name.startswith("_"))
                report = gradient_diagnostics(diagnostic_tasks, model)
                state["gradient_diagnostics"].append({"epoch": epoch, "chunk": chunk_index,
                    "contrastive_pairs": contrastive_pairs, **report})
            buffer = PositiveGradientBuffer(state["positive_gradients"])
            bank = ContributionBank(state)
            negative_banks = []
            stage = time.monotonic()
            for group, bank_tasks in bank_groups:
                if group["category"] == "positive":
                    change = buffer.add(group, bank_tasks, model)
                    state["sample_audit"][-1].setdefault("positive_retention", []).append(change)
                else:
                    negative_banks.append(bank_tasks)
                    state["locked_geometries"].update(geometry_key(item) for item in group["versions"])
            bank.negative_many(negative_banks, model)
            chunk_timing["candidate_gradient_seconds"] = time.monotonic() - stage
            loss.backward()
            state["pending_timeline_objective"] += float(loss.detach())
            state["sample_audit"][-1]["endpoint_scales"] = balance["scales"]
            state["sample_audit"][-1]["endpoint_supervision"] = endpoint_counts
            state["sample_audit"][-1]["endpoint_weighted_components"] = {
                name: float(value.detach()) * config["loss"]["transitions"] * .5
                    * contribution_weights[key[0]] / training["accumulate_sessions"]
                for name, value in endpoint_parts.items()}
            state["sample_audit"][-1]["inactive_tasks"] = [name for name, scale in schedule.items() if scale == 0]
            stage = time.monotonic()
            cache.append(current_times.detach().cpu().numpy(), features.detach().cpu().numpy())
            chunk_timing["cache_write_seconds"] = time.monotonic() - stage
            state["cache_rows"] = cache.rows
            state["cache_sha256"] = cache.checksum
            state["next_chunk"] = chunk_index + 1
            if last:
                state["sample_audit"][-1]["positive_committed_events"] = len(state["positive_gradients"])
                state["sample_audit"][-1]["positive_committed_groups"] = {
                    str(event): [item["descriptor"] for item in groups.values()]
                    for event, groups in state["positive_gradients"].items()}
                ContributionBank(state).finish_session(state["positive_gradients"])
                state["sample_audit"][-1]["actual_session_contributions"] = state["last_session_contributions"]
                state["natural_replay"] = None
                replay_payload = None
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
                        state["pending_timeline_objective"] *= factor
                    ContributionBank(state).commit(model)
                    state["sample_audit"][-1]["actual_update_contributions"] = state["last_update_contributions"]
                    state["last_update_objective"] = state["pending_timeline_objective"] + sum(state["last_update_candidate_objective"].values())
                    state["sample_audit"][-1]["update_objective_for_retained_set"] = state["last_update_objective"]
                    state["pending_timeline_objective"] = 0.0
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
                    apply_learning_rate(optimizer, state, training,
                        epoch - 1 + (state["optimizer_steps"] - 1) / expected_optimizer_steps)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    state["update_count"] += 1
                    state["session_count"] = 0
            usage = enforce_resources(config, device)
            state["sample_audit"][-1].update({"elapsed_seconds": time.monotonic() - chunk_started,
                "cache_write_bytes": len(current_times) * (8 + capabilities(config).feature_width * 4),
                "positive_buffer_events": len(state["positive_gradients"]), "resources": usage,
                "workers": training["num_workers"], "prefetch": training["loader_prefetch_factor"],
                "reader_cache": training["session_reader_cache_size"]})
            chunk_timing["previous_checkpoint_total_seconds"] = state.get("previous_checkpoint_total_seconds", 0.0)
            state["sample_audit"][-1]["stage_timing"] = chunk_timing
            if hasattr(inputs, "assert_session_unchanged"):
                inputs.assert_session_unchanged(key)
            checkpoint()
            displayed_loss = float(loss.detach())
            chunk_progress.set_postfix(chunk=f"{displayed_loss:.5f}",
                                       update=f"{state.get('last_update_objective', 0):.5f}", subject=key[0],
                                       refresh=False)
        chunk_progress.close()
        if state["optimizer_steps"] != expected_optimizer_steps:
            raise RuntimeError("Completed joint epoch optimizer count differs from session traversal")
        clipping_report = epoch_clipping_report(
            epoch, state["clipped_steps"], state["optimizer_steps"], expected_optimizer_steps,
            training["clipping_abort_fraction"])
        fraction = clipping_report["clipping_fraction"]
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
        state["boundary"].update({"qualified": epoch >= settings["train_from_epoch"]
            and config.get("ablation", {}).get("name") != "no_boundary", "mode": "both",
            "training_qualification_separate_from_oof_enablement": True})
        state["epoch_candidates"] = []
        state["epoch_complete"] = True
        qualification = epoch >= training["checkpoint_min_epoch"] and fraction <= training["clipping_promotion_fraction"]
        epoch_payload = {"model": model.state_dict(), "identity": identity, "epoch": epoch,
            "protocol": protocol_for(config), "state_version": CHECKPOINT_STATE_VERSION,
            "epoch_complete": True, "promotion_eligible": qualification, "clipping_fraction": fraction,
            "training_subjects": sorted(subjects),
            "excluded_subjects": sorted((outer_excluded or set()) | (selector or set())),
            "preprocessing": preprocessing, "boundary": state["boundary"],
            "config_sha256": bound_config, "source_sha256": digest(bound_source),
            "data_sha256": digest(inputs.identity)}
        epoch_payload.update({"lr_position": state.get("lr_position", 0),
                              "learning_rates": state.get("learning_rates", [])})
        if state.get("resume_migration"):
            epoch_payload["resume_migration"] = state["resume_migration"]
        checkpoint()
        if not tensors_finite(epoch_payload["model"].values()):
            raise RuntimeError("Epoch checkpoint finite gate failed")
        if deployment_epoch is None or epoch <= deployment_epoch:
            save_torch_atomic(directory / "latest.pt", epoch_payload)
        if selector and epoch >= 10 and (epoch - 10) % 4 == 0:
            save_torch_atomic(directory / "periodic" / f"epoch_{epoch:03d}.pt", epoch_payload)
        if deployment_epoch is not None and (epoch % 4 == 0 or epoch in {
                deployment_epoch - 4, deployment_epoch - 2, deployment_epoch,
                deployment_epoch + 2, deployment_epoch + 4}):
            save_torch_atomic(directory / "nearby" / f"epoch_{epoch:03d}.pt", epoch_payload)
        record = {"epoch": epoch, "clipping_fraction": fraction, "promotion_eligible": qualification,
                  "chunks": len(sampler.chunks), "sessions": len(session_lengths),
                  "optimizer_steps": state["optimizer_steps"], "candidate_coverage": coverage,
                  "clipping_gate": clipping_report,
                  "maximum_gradient_norm": state.get("maximum_gradient_norm", 0),
                  "elapsed_seconds": time.monotonic() - started, **enforce_resources(config, device)}
        record.update({"lr_position": state.get("lr_position", 0),
                       "learning_rates": state.get("learning_rates", [])})
        stop = False
        if selector and epoch >= training["checkpoint_min_epoch"] and epoch % training["validation_every_epochs"] == 0:
            from bme_eating.data.joint_sessions import release_session_loader
            release_session_loader(loader)
            loader = None
            assert_excluded(subjects, selector, outer_excluded or set())
            try:
                fitted_mapping = fit_source_mapping(model, inputs, subjects, config, preprocessing,
                                                    directory / "source_fit" / str(epoch), identity)
                predictions = predict_joint_model(model, inputs, selector, config, preprocessing,
                    directory / "selector" / str(epoch), identity,
                    {**state["boundary"], "search_eligible": False}, mapping=fitted_mapping)
                _, _, selector_events, _ = inputs.frames(selector, labels=True)
                selector_truth, selector_ignore = partition_evaluation_events(selector_events, selector)
                point, search_records = choose_operating_point(predictions["coarse"], predictions["windows"],
                    selector_truth, selector_ignore, {identity: fitted_mapping}, config)
                print_metrics(f"selector epoch {epoch}/{epochs}", point)
                f1 = point["f1"]
                eligible = qualification and point["promotion_eligible"]
            except NoSourceCandidatesError as error:
                point = {"promotion_eligible": False, "f1": None, "calibration": None,
                         "operating_point": None, "reason": "no_training_source_candidates",
                         "source_candidates": 0, "message": str(error)}
                search_records = []
                f1, eligible = None, False
                print(f"[selector epoch {epoch}] unqualified: no training source candidates", flush=True)
            record["selector"] = point
            write_json_atomic(directory / "selector" / str(epoch) / "search.json", search_records)
            write_json_atomic(directory / "selector" / str(epoch) / "validation_status.json", point)
            if qualification and ((f1 is not None and f1 > state["diagnostic_f1"])
                                  or (f1 is None and state["diagnostic_epoch"] is None)):
                state["diagnostic_epoch"], state["diagnostic_f1"] = epoch, f1 if f1 is not None else -1.0
                save_torch_atomic(directory / "diagnostic_selected.pt", {
                    **epoch_payload, "promotion_eligible": False, "diagnostic_only": True})
            improved, stop = selection_update(state, epoch, eligible, f1, training)
            if improved:
                save_torch_atomic(directory / "selected.pt", epoch_payload)
            record["selection_state"] = {name: state.get(name) for name in (
                "best_epoch", "stale_checks", "unqualified_checks", "lost_qualification_checks", "selection_failure")}
            print(f"[selection] best_epoch={state['best_epoch']} best_F1={state['best_f1']:.4f} "
                  f"stale_checks={state['stale_checks']}/{training['patience_checks']} "
                  f"clipping={fraction:.4f} early_stop={stop} "
                  f"unqualified_checks={state.get('unqualified_checks', 0)} "
                  f"lost_qualification_checks={state.get('lost_qualification_checks', 0)}", flush=True)
        stop_reason = ((state.get("selection_failure") or "f1_patience") if stop else
                       "maximum_epochs" if epoch >= epochs else None)
        record["stop_reason"] = stop_reason
        journals["training_history"].append([record])
        state["journal_offsets"]["training_history"] = journals["training_history"].offset
        if "selector" in record:
            record = {**record, "selector": {name: value for name, value in record["selector"].items()
                       if name != "budget_audit"}}
        state["history"].append(record)
        write_json_atomic(directory / "training_history.json", state["history"])
        write_json_atomic(directory / "sampling.json", state["sample_audit"])
        write_json_atomic(directory / "gradients.json", state["gradient_diagnostics"])
        state["completed_epoch"] = epoch
        if stop_reason:
            state.update({"training_complete": True, "stop_requested": stop,
                          "stop_reason": stop_reason})
        state["epoch"] += 1
        state["endpoint_balance"] = None
        state["next_chunk"] = 0
        state["sampler_identity"] = None
        state["epoch_complete"] = False
        state["clipped_steps"], state["optimizer_steps"] = 0, 0
        state["maximum_gradient_norm"] = 0
        state["event_contributions"], state["family_contributions"] = set(), set()
        state["positive_group_counts"] = {}
        if state["positive_gradients"]:
            raise RuntimeError("Positive gradients remain uncommitted after epoch completion")
        state["epoch_candidates"] = []
        checkpoint()
        if stop:
            break
    state["training_complete"] = True
    from bme_eating.data.joint_sessions import release_session_loader
    release_session_loader(loader)
    checkpoint()
    return _training_result(state, selector, epochs, directory)


def decoder_settings(config):
    return {f"high_{high:g}_low_{ratio:g}_gap_{gap}":
            {**config["decoder"], "high_threshold": high, "low_ratio": ratio,
             "gap_merge_seconds": gap}
            for high, ratio, gap in product(config["search"]["high_thresholds"],
                config["search"]["low_ratios"], config["search"]["gap_seconds"])}


@torch.no_grad()
def predict_joint_model(model, inputs, subjects, config, preprocessing, directory, identity,
                        boundary, *, training_source: bool = False, mapping=None, operating_point=None,
                        development_calibration=None):
    from bme_eating.prediction import predict
    return predict(model, inputs, subjects, config, preprocessing, directory, identity, boundary,
                   source_only=training_source, mapping=mapping, operating_point=operating_point,
                   development_calibration=development_calibration)


def fit_source_mapping(model, inputs, subjects, config, preprocessing, directory, identity):
    from bme_eating.prediction import fit_mapping
    return fit_mapping(model, inputs, subjects, config, preprocessing, directory, identity)


def load_trained(directory, config, identity, subjects, data_identity, *, checkpoint_name="latest.pt"):
    if checkpoint_name not in {"latest.pt", "selected.pt"}:
        raise ValueError("Unregistered checkpoint role")
    payload = torch.load(directory / checkpoint_name, map_location=config["training"]["device"], weights_only=True)
    if not checkpoint_qualified(payload, identity, subjects, config["training"]["checkpoint_min_epoch"]):
        raise RuntimeError("Unqualified joint checkpoint cannot enter OOF/promotion")
    if payload.get("config_sha256") != config_identity(config) or payload.get("source_sha256") != digest(
            training_source_identity(Path(config["_config_path"]).resolve().parents[1], config)):
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
    proposal_keys = ["subject_key", "session_id", "prediction_event_id"]
    before = set(map(tuple, coarse_match[proposal_keys].values)) if len(coarse_match) else set()
    after = set(map(tuple, refined_match[proposal_keys].values)) if len(refined_match) else set()
    paired = coarse_match.merge(refined_match, on=event_keys,
                                suffixes=("_coarse", "_refined")) if before and after else pd.DataFrame()
    improvements = {}
    for endpoint in ("start", "end"):
        old = float(paired[endpoint + "_absolute_error_ms_coarse"].mean()) if len(paired) else float("nan")
        new = float(paired[endpoint + "_absolute_error_ms_refined"].mean()) if len(paired) else float("nan")
        improvements[endpoint] = (old - new) / old if old > 0 else 0
    truth_before = set(map(tuple, coarse_match[event_keys].values)) if len(coarse_match) else set()
    truth_after = set(map(tuple, refined_match[event_keys].values)) if len(refined_match) else set()
    from bme_eating.evaluation import event_frame
    refined_frame = event_frame(refined)
    refined_keys = ["subject_key", "session_id", "proposal_id"]
    emitted = set(map(tuple, refined_frame[refined_keys].values))
    ignored = prediction_ignore_mask(refined_frame, ignore)
    false_positive_ids = set(map(tuple, refined_frame.loc[~ignored, refined_keys].values)) - after
    conversions = before & false_positive_ids
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
            "tp_to_fp_basis": ("coarse_matched_prediction_id_to_refined_false_positive"),
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
