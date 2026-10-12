from __future__ import annotations

import copy
import time

import torch

from bme_eating.models.statsfusion_joint_event import distribution_endpoint, gaussian_distribution
from bme_eating.protocol import capabilities


def score_candidates(model, candidates, times, features, device, *, batch_size=128, cache=None,
                     performance=None):
    if not candidates:
        return []
    outputs = []
    from bme_eating.models.statsfusion_joint_event import indexed_roi as roi_builder
    local_slice = slice(137, 201)
    for first in range(0, len(candidates), batch_size):
        started = time.monotonic()
        pool, rois = candidates[first:first + batch_size], []
        for candidate in pool:
            left = min(candidate.start_ms, candidate.end_ms) - 60_000
            right = min(candidate.end_ms + 60_000, candidate.decision_time_ms)
            limits = torch.searchsorted(times, torch.tensor([left, right]), right=True).tolist()
            support_times, support = times[limits[0]:limits[1]], features[limits[0]:limits[1]]
            if len(support):
                candidate.missing_fraction = float(support[:, 136].mean().detach())
                candidate.gyro_missing = bool((support[:, 135] <= 0).all())
            roi = cache.get(candidate) if cache is not None else None
            if roi is None:
                roi = roi_builder(support[:, :64], support[:, local_slice], support_times,
                                  support[:, 128], candidate.start_ms, candidate.end_ms,
                                  candidate.decision_time_ms)
                if cache is not None:
                    cache.put(candidate, roi)
            rois.append(roi)
        if performance is not None:
            performance["roi_seconds"] = performance.get("roi_seconds", 0.0) + time.monotonic() - started
        started = time.monotonic()
        scores = model.candidate_head(torch.stack(rois).to(device),
                                      torch.stack([candidate.metadata(device) for candidate in pool]))
        detached = torch.stack((scores["event_logit"], scores["quality_logit"].sigmoid()), -1).detach().cpu()
        if performance is not None:
            performance["candidate_head_seconds"] = performance.get("candidate_head_seconds", 0.0) + time.monotonic() - started
            performance["scoring_head_calls"] = performance.get("scoring_head_calls", 0) + 1
            performance["scoring_candidates"] = performance.get("scoring_candidates", 0) + len(pool)
        for index, candidate in enumerate(pool):
            candidate.event_logit, candidate.predicted_quality = detached[index].tolist()
            candidate.score_formula = capabilities(model).score_formula
            outputs.append({name: value[index:index + 1] for name, value in scores.items()})
    return outputs


def boundary_support(model, candidate, times, features, device, boundary, embedding, endpoint):
    center = candidate.start_ms if endpoint == 0 else candidate.end_ms
    seconds = boundary["start_range_seconds"] if endpoint == 0 else boundary.get("end_range_seconds", 30)
    limits = torch.searchsorted(times, torch.tensor([center - seconds * 1000 - 1,
                                                    min(center + seconds * 1000,
                                                        candidate.decision_time_ms)]), right=True).tolist()
    support, values = times[limits[0]:limits[1]], features[limits[0]:limits[1]]
    valid = values[:, 128] > 0
    support, values = support[valid].to(device), values[valid].to(device)
    if len(support) < 3:
        return support, None
    local = values[:, 137:201]
    logits = model.boundary_head(local, (support - center).float() / 1000,
                                 embedding, values[:, 128], values[:, 133], values[:, 134])[:, endpoint]
    return support, logits


def refine(model, candidate, times, features, device, boundary, entropy, *, embedding=None,
           rescore=True):
    result = copy.deepcopy(candidate)
    result.unrefined_start_ms, result.unrefined_end_ms = candidate.start_ms, candidate.end_ms
    result.coarse_event_logit, result.coarse_quality = candidate.event_logit, candidate.predicted_quality
    mode = boundary.get("mode", "both")
    result.boundary_mode = mode

    def fallback(reason):
        result.start_ms, result.end_ms, result.refined = candidate.start_ms, candidate.end_ms, False
        result.boundary_fallback_reason = reason
        result.boundary_conflict = reason in {"conflict_or_latency", "insufficient_valid_bins"}
        return result

    if mode == "off" or not boundary.get("qualified") or candidate.source == "truth_jitter":
        return fallback("disabled")
    if embedding is None:
        embedding = score_candidates(model, [candidate], times, features, device)[0]["candidate_embedding"][0]
    for endpoint in ((0,) if mode == "start_only" else (0, 1)):
        support, logits = boundary_support(model, candidate, times, features, device, boundary,
                                           embedding, endpoint)
        name = "start" if endpoint == 0 else "end"
        setattr(result, "boundary_valid_" + name + "_bins", len(support))
        if logits is None:
            return fallback("insufficient_valid_bins")
        proposal, uncertainty = distribution_endpoint(logits, support)
        setattr(result, "boundary_" + name + "_entropy", uncertainty)
        if uncertainty > entropy:
            return fallback("entropy")
        seconds = boundary["start_range_seconds"] if endpoint == 0 else boundary.get("end_range_seconds", 30)
        result.boundary_clipped |= abs(proposal - getattr(candidate, name + "_ms")) >= seconds * 1000 - 3000
        setattr(result, name + "_ms", proposal)
    if not result.start_ms < result.end_ms <= candidate.decision_time_ms or (
            candidate.decision_time_ms - result.end_ms > 60_000):
        return fallback("conflict_or_latency")
    result.refined = result.start_ms != candidate.start_ms or result.end_ms != candidate.end_ms
    if result.refined and rescore:
        score_candidates(model, [result], times, features, device)
    result.validate()
    return result


def boundary_loss(model, candidate, target, times, features, device, boundary, embedding):
    positive, _, _, true_start, true_end = target
    zero = embedding.sum() * 0
    if not positive:
        return zero
    losses = []
    for endpoint, truth_endpoint in enumerate((true_start, true_end)):
        if truth_endpoint > candidate.decision_time_ms:
            continue
        center = candidate.start_ms if endpoint == 0 else candidate.end_ms
        maximum = boundary["start_range_seconds"] if endpoint == 0 else boundary.get("end_range_seconds", 30)
        if abs(truth_endpoint - center) > maximum * 1000:
            continue
        support, logits = boundary_support(model, candidate, times, features, device, boundary,
                                           embedding, endpoint)
        if logits is not None and int(support[0]) <= truth_endpoint <= int(support[-1]):
            distribution = gaussian_distribution(support, truth_endpoint, boundary.get("sigma_seconds", 9))
            losses.append(-(distribution * logits.float().log_softmax(0)).sum())
    return torch.stack(losses).mean() if losses else zero
