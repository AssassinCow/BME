from __future__ import annotations

import copy
import time

import pandas as pd
import torch

from bme_eating.data.joint_sessions import (
    SessionChunkSampler,
    make_joint_dataset,
    release_session_loader,
    session_loader,
)
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.joint_capabilities import capabilities, supported
from bme_eating.joint_progress import progress
from bme_eating.joint_stream import FrozenSourceCDF, OnlineBudget, TimelineCache, stable_id
from bme_eating.joint_v411_features import ExactROICache
from bme_eating.joint_v411_scoring import refine, score_candidates
from bme_eating.joint_v411_stream import make_stream


@torch.no_grad()
def predict(model, inputs, subjects, config, preprocessing, directory, identity, boundary,
            *, source_only=False, mapping=None, operating_point=None, development_calibration=None):
    from bme_eating.training.statsfusion_joint_trainer import (
        decoder_settings,
        enforce_resources,
        pack_features,
    )
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = next(model.parameters()).device
    dataset, _ = make_joint_dataset(inputs, subjects, config, preprocessing, training=source_only,
                                   development_calibration=development_calibration)
    dataset.training = False
    sampler = SessionChunkSampler(dataset, 0, training=False)
    model.eval()
    reference_execution = config.get("_reference_execution", False)
    head_batch = 1 if reference_execution else config["training"]["validation_head_batch_size"]
    settings = decoder_settings(config)
    v412 = supported(model, "consistent_inputs")
    deferred = supported(model, "stabilized") and not source_only
    if v412:
        from bme_eating.joint_protocol import digest
        from bme_eating.joint_v412_cache import PredictionFeatureCache, feature_binding
        feature_identity = feature_binding(model, inputs, config, preprocessing, identity)
        feature_root = directory.parent / "feature_cache" / digest(feature_identity)
        feature_caches = {key: PredictionFeatureCache(feature_root / stable_id(*key),
            {**feature_identity, "subject": key[0], "session": key[1]}) for key in dataset.session_groups}
        reader_sampler = copy.copy(sampler)
        reader_sampler.chunks = [chunk for chunk in sampler.chunks if not feature_caches[chunk[3]].reused]
    else:
        reader_sampler = sampler
    loader = session_loader(dataset, reader_sampler, config) if reader_sampler.chunks else None
    batches = iter(loader) if loader is not None else iter(())
    if operating_point is not None:
        settings = {operating_point["setting"]: settings[operating_point["setting"]]}
    if source_only and v412:
        settings = {name: {**decoder, "parent_proposals_only": True} for name, decoder in settings.items()}
    modes = [(mode, entropy) for mode in ("start_only", "both")
             for entropy in config["boundary"]["entropy_thresholds"]]
    if v412 and (source_only or not boundary.get("search_eligible", True)):
        modes = []
    if operating_point is not None:
        modes = ([(operating_point["boundary_mode"], operating_point["boundary_entropy"])]
                 if operating_point["boundary_enabled"] else [])
    coarse = {name: [] for name in settings}
    refined = {f"{mode}:{entropy}": {name: [] for name in settings} for mode, entropy in modes}
    windows, current_key, streams, trackers = [], None, {}, {}
    previous_roi_hits, previous_roi_misses = 0, 0
    counters = {"head_calls": 0, "boundary_rescore_calls": 0, "parent_proposals": 0,
                "generated_geometries": 0, "admitted_geometries": 0,
                "roi_hits": 0, "roi_misses": 0, "cache_write_bytes": 0,
                "encoder_seconds": 0.0, "proposal_seconds": 0.0, "head_seconds": 0.0,
                "roi_seconds": 0.0, "candidate_head_seconds": 0.0, "cache_write_seconds": 0.0}
    scoring_options = {"performance": counters} if supported(model, "stabilized") else {}
    try:
        prediction_progress = progress(range(len(sampler.chunks)), total=len(sampler.chunks),
            description=f"{'Source CDF' if source_only else 'Validation inference'} {directory.parent.name}/{directory.name}")
        for chunk_index in prediction_progress:
            row, _, _, key, last, count = sampler.chunks[chunk_index]
            if hasattr(inputs, "assert_session_unchanged"):
                inputs.assert_session_unchanged(key)
            if key != current_key:
                current_key = key
                streams = {name: make_stream(*key, identity, decoder) for name, decoder in settings.items()}
                pending = {name: [] for name in settings}
                budgets = [operating_point["budget"]] if operating_point is not None else config["search"]["budgets"]
                trackers = {name: {budget: OnlineBudget(budget, {}) for budget in budgets}
                            for name in settings}
                if v412:
                    cache = feature_caches[key]
                else:
                    cache = TimelineCache(directory / "cache" / stable_id(*key),
                        {"identity": identity, "epoch": "prediction", "subject": key[0], "session": key[1]}, 137)
                roi_cache = ExactROICache((identity, key))
                previous_roi_hits, previous_roi_misses = 0, 0
            if v412 and cache.reused:
                group = dataset.session_groups[key]
                stop = int((group["_row_id"].to_numpy() == row).nonzero()[0][0]) + 1
                batch = {"timestamp_ms": torch.from_numpy(group.timestamp_ms.to_numpy()[stop - count:stop])[None]}
            else:
                batch = next(batches)
            batch = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
            stage = time.monotonic()
            mask = torch.zeros(batch["timestamp_ms"].shape[1], dtype=torch.bool, device=device)
            mask[-count:] = True
            if v412 and cache.reused:
                times, features = cache.read(batch["timestamp_ms"][0, mask].cpu().numpy())
                counters["feature_cache_hit_chunks"] = counters.get("feature_cache_hit_chunks", 0) + 1
            else:
                outputs = model(batch)
                times, features = batch["timestamp_ms"][0, mask].cpu(), pack_features(outputs, mask).cpu()
            counters["encoder_seconds"] += time.monotonic() - stage
            if v412:
                if not cache.reused:
                    stage = time.monotonic()
                    cache.append(times, features)
                    counters["cache_write_seconds"] += time.monotonic() - stage
                    counters["cache_write_bytes"] += len(times) * (8 + capabilities(config).feature_width * 4)
                all_times, all_features = cache.prefix()
                if last:
                    stage = time.monotonic()
                    cache.finish()
                    counters["cache_write_seconds"] += time.monotonic() - stage
            elif not source_only:
                stage = time.monotonic()
                cache.append(times.numpy(), features.numpy())
                counters["cache_write_seconds"] += time.monotonic() - stage
                all_times, all_features = cache.tensors()
                counters["cache_write_bytes"] += len(times) * (8 + capabilities(config).feature_width * 4)
            for timestamp, values in zip(times.tolist(), features.numpy()):
                windows.append({"subject_key": key[0], "session_id": key[1], "timestamp_ms": timestamp,
                                "valid": float(values[128]), "gyro_valid": float(values[135]), "model_identity": identity,
                                **({"onset_probability": float(values[133]), "offset_probability": float(values[134])}
                                   if v412 else {})})
            stage = time.monotonic()
            generated = {name: [] for name in settings}
            for name, stream in streams.items():
                for timestamp, values in zip(times.tolist(), features.numpy()):
                    for tracker in trackers[name].values():
                        tracker.observe(3, float(values[128]))
                    probabilities = {"state": float(values[129]), **{f"proposal_{scale}": float(values[130 + index])
                                      for index, scale in enumerate((3, 15, 60))}}
                    pool = stream.observe(timestamp, probabilities, float(values[128]), float(values[134]),
                                          onset_probability=float(values[133]))
                    if last and timestamp == int(times[-1]):
                        pool.extend(stream.finish())
                    counters["parent_proposals"] += len({item.ancestor_id for item in pool})
                    counters["generated_geometries"] += len(pool)
                    if source_only:
                        generated[name].extend(item for item in pool if item.source in {"state", "proposal"})
                    else:
                        if mapping is None:
                            raise RuntimeError("v411 prediction requires a training-fitted source CDF")
                        union = {}
                        for tracker in trackers[name].values():
                            for item in tracker.select(copy.deepcopy(pool), mapping) if pool else []:
                                union[item.proposal_id] = item
                        generated[name].extend(pool if reference_execution else union.values())
            counters["proposal_seconds"] += time.monotonic() - stage
            if deferred:
                for name, pool in generated.items():
                    pending[name].extend(pool)
                generated = pending if last else {name: [] for name in settings}
            if source_only:
                for name, pool in generated.items():
                    coarse[name].extend(pool)
            else:
                stage = time.monotonic()
                for name, pool in generated.items():
                    scores = score_candidates(model, pool, all_times, all_features, device,
                        batch_size=head_batch, cache=None if reference_execution else roi_cache,
                        **scoring_options)
                    counters["head_calls"] += (len(pool) + head_batch - 1) // head_batch
                    counters["admitted_geometries"] += len(pool)
                    coarse[name].extend(pool)
                    for mode, entropy in modes:
                        versions = [refine(model, item, all_times, all_features, device,
                                {**boundary, "mode": mode}, entropy, embedding=score["candidate_embedding"][0],
                                rescore=False) for item, score in zip(pool, scores)]
                        changed = [item for item in versions if item.refined]
                        score_candidates(model, changed, all_times, all_features, device,
                                batch_size=head_batch, cache=None if reference_execution else roi_cache,
                                **scoring_options)
                        counters["boundary_rescore_calls"] += (len(changed) + head_batch - 1) // head_batch
                        refined[f"{mode}:{entropy}"][name].extend(versions)
                counters["head_seconds"] += time.monotonic() - stage
                counters["roi_hits"] += roi_cache.hits - previous_roi_hits
                counters["roi_misses"] += roi_cache.misses - previous_roi_misses
                previous_roi_hits, previous_roi_misses = roi_cache.hits, roi_cache.misses
            usage = enforce_resources(config, device)
            prediction_progress.set_postfix(head_calls=counters["head_calls"],
                candidates=counters["admitted_geometries"], refresh=False)
            if hasattr(inputs, "assert_session_unchanged"):
                inputs.assert_session_unchanged(key)
        prediction_progress.close()
        counters.update({"elapsed_seconds": time.monotonic() - started, "source_only": source_only,
                         "cdf_fallback_counts": dict(mapping.fallback_counts) if mapping is not None else {},
                         "reader_workers": config["training"]["num_workers"] if loader is not None else 0,
                         "boundary_search_skipped": not bool(modes), "decoder_settings": list(settings),
                         "locked_operating_point": operating_point is not None,
                         "candidate_throughput_per_second": counters["admitted_geometries"] /
                         max(time.monotonic() - started, 1e-9), **usage})
        write_json_atomic(directory / "performance.json", counters)
        frame = pd.DataFrame(windows)
        if v412 and not source_only:
            from bme_eating.joint_v412_diagnostics import endpoint_diagnostics
            from bme_eating.metrics import partition_evaluation_events
            _, _, events, _ = inputs.frames(subjects, labels=True)
            truth, ignore = partition_evaluation_events(events, subjects)
            diagnostics = endpoint_diagnostics(frame, truth, ignore,
                [item for pool in coarse.values() for item in pool])
            write_json_atomic(directory / "endpoint_diagnostics.json", diagnostics)
        return {"coarse": coarse, "refined": refined, "windows": frame, "performance": counters}
    finally:
        release_session_loader(loader)


def fit_mapping(model, inputs, subjects, config, preprocessing, directory, identity):
    outputs = predict(model, inputs, subjects, config, preprocessing, directory, identity,
                       {}, source_only=True)
    mapping = FrozenSourceCDF.fit([item for pool in outputs["coarse"].values() for item in pool], subjects)
    mapping.fallback = supported(config, "stabilized")
    write_json_atomic(directory / "source_mapping.json", mapping.to_json())
    return mapping
