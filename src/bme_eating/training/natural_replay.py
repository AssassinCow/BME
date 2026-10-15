from __future__ import annotations

import random
import time
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch

from bme_eating.artifacts import sha256_file
from bme_eating.candidates import candidate_order_key, decision_key, geometry_key, stable_id
from bme_eating.prediction_cache import feature_binding
from bme_eating.proposal_stream import make_stream


@contextmanager
def preserve_training_state(model):
    modes = {name: module.training for name, module in model.named_modules()}
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)


def merge_replay_candidates(candidates):
    representatives, duplicates = {}, []
    for candidate in candidates:
        candidate.validate()
    for candidate in sorted(candidates, key=candidate_order_key):
        geometry = geometry_key(candidate)
        if geometry not in representatives:
            representatives[geometry] = deepcopy(candidate)
            continue
        first = representatives[geometry]
        flags = candidate.generation_parameters.get("source_flags", [candidate.source])
        if decision_key(first) == decision_key(candidate):
            first.generation_parameters["source_flags"] = sorted(set(
                first.generation_parameters.get("source_flags", [first.source]) + flags))
        duplicates.append({"geometry": geometry, "proposal_id": candidate.proposal_id,
            "representative_id": first.proposal_id, "decision_time_ms": candidate.decision_time_ms,
            "decision_phase": candidate.decision_phase, "source_flags": sorted(set(flags))})
    return sorted(representatives.values(), key=candidate_order_key), duplicates


def replay_session(model, inputs, config, preprocessing, dataset, sampler, key, directory,
                   identity, epoch, update, stored=None):
    from bme_eating.training.statsfusion_joint_trainer import (
        decoder_settings,
        enforce_resources,
        generate_chunk,
        pack_features,
        save_torch_atomic,
    )
    binding = {**feature_binding(model, inputs, config, preprocessing, identity),
               "epoch": epoch, "update": update, "subject": key[0], "session": key[1]}
    path = directory / f"epoch_{epoch}" / (stable_id(*key) + ".pt")
    if stored:
        if stored["binding"] != binding or stored["path"] != str(path) or sha256_file(path) != stored["sha256"]:
            raise RuntimeError("Natural replay cache hash or session weights changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["binding"] != binding:
            raise RuntimeError("Natural replay payload binding mismatch")
        return payload, stored
    started = time.monotonic()
    streams = {name: make_stream(*key, identity, decoder) for name, decoder in decoder_settings(config).items()}
    candidates, timeline, features = [], [], []
    device = next(model.parameters()).device
    chunks = [item for item in sampler.chunks if item[3] == key]
    previous_training = dataset.training
    dataset.training = False
    try:
        with preserve_training_state(model):
            for row, _, importance, _, last, new_steps in chunks:
                batch = dataset[(row, epoch, importance)]
                batch = {name: value[None] if isinstance(value, torch.Tensor) else value
                         for name, value in batch.items()}
                batch = {name: value.to(device) if isinstance(value, torch.Tensor) else value
                         for name, value in batch.items()}
                mask = torch.zeros_like(batch["state_target"], dtype=torch.bool)
                mask[:, -new_steps:] = True
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = model(batch)
                times = batch["timestamp_ms"][0, mask[0]].detach().cpu()
                packed = pack_features(outputs, mask[0]).detach().cpu()
                timeline.append(times)
                features.append(packed)
                for stream in streams.values():
                    pool = generate_chunk(stream, times, packed)
                    if last:
                        pool.extend(stream.finish())
                    candidates.extend(pool)
                enforce_resources(config, device)
    finally:
        dataset.training = previous_training
    candidates, duplicates = merge_replay_candidates(candidates)
    payload = {"binding": binding, "times": torch.cat(timeline), "features": torch.cat(features),
               "candidates": candidates, "duplicates": duplicates,
               "seconds": time.monotonic() - started}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_atomic(path, payload)
    manifest = {"binding": binding, "path": str(path), "sha256": sha256_file(path),
                "cursor": 0, "released": 0, "duplicate_cursor": 0, "duplicates_released": 0,
                "seconds": payload["seconds"]}
    return payload, manifest


def release_candidates(payload, state, observed_ms):
    pool, cursor = payload["candidates"], state["cursor"]
    stop = cursor
    while stop < len(pool) and pool[stop].decision_time_ms <= observed_ms:
        stop += 1
    state["cursor"], state["released"] = stop, state["released"] + stop - cursor
    duplicates = payload.get("duplicates", [])
    duplicate_cursor = state.get("duplicate_cursor", 0)
    duplicate_stop = duplicate_cursor
    while duplicate_stop < len(duplicates) and duplicates[duplicate_stop]["decision_time_ms"] <= observed_ms:
        duplicate_stop += 1
    state["duplicate_cursor"] = duplicate_stop
    state["duplicates_released"] = state.get("duplicates_released", 0) + duplicate_stop - duplicate_cursor
    state["last_duplicates"] = duplicates[duplicate_cursor:duplicate_stop]
    return pool[cursor:stop]


def naturalize_jitter(candidates, times, features, natural_candidates):
    for candidate in candidates:
        if candidate.source != "truth_jitter":
            continue
        observed = [item for item in natural_candidates if item.decision_time_ms <= candidate.decision_time_ms]
        parent = min(observed, key=lambda item: (abs(item.start_ms - candidate.start_ms)
                     + abs(item.end_ms - candidate.end_ms), candidate_order_key(item))) if observed else None
        if parent:
            candidate.raw_score = parent.raw_score
            candidate.scale_seconds = parent.scale_seconds
            candidate.coarse_start_ms, candidate.coarse_end_ms = parent.coarse_start_ms, parent.coarse_end_ms
            source = parent.generation_parameters.get("parent_source", parent.source)
        else:
            support = (times > candidate.start_ms) & (times <= min(candidate.end_ms, candidate.decision_time_ms))
            weights = features[support, 128]
            candidate.raw_score = float((features[support, 129] * weights).sum() / weights.sum().clamp_min(1e-8))
            candidate.coarse_start_ms, candidate.coarse_end_ms = candidate.start_ms, candidate.end_ms
            source = "state"
        candidate.generation_parameters["metadata_source"] = source
        candidate.generation_parameters["natural_metadata"] = True
