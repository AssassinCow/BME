from __future__ import annotations

import math
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from bme_eating.amp import training_inference
from bme_eating.candidate_scoring import boundary_loss, score_candidates
from bme_eating.candidates import candidate_order_key, candidate_targets, geometry_key, stable_id
from bme_eating.models.statsfusion_joint_event import quality_focal_loss
from bme_eating.protocol import supported
from bme_eating.training.contributions import (
    ContributionBank as ContributionBank,
)
from bme_eating.training.contributions import (
    PositiveGradientBuffer as PositiveGradientBuffer,
)


def group_losses(model, groups, truth, times, features, device, boundary, config, schedule,
                 *, performance=None):
    candidates = [candidate for group in groups for candidate in group["versions"]]
    scoring_options = {"performance": performance} if performance is not None else {}
    scores = score_candidates(model, candidates, times, features, device,
                               batch_size=max(1, len(candidates)), **scoring_options)
    score_by_id = {candidate.proposal_id: score for candidate, score in zip(candidates, scores)}
    results, label_changes = [], []
    for group in groups:
        losses = {name: [] for name in ("event", "quality", "boundary")}
        quality_pairs, version_tasks, pair_tasks = [], [], []
        for candidate in group["versions"]:
            score = score_by_id[candidate.proposal_id]
            target = candidate_targets(candidate, truth)
            positive, quality = target[:2]
            ranking_quality, ranking_logit = quality, score["quality_logit"].float().squeeze()
            event_loss = F.binary_cross_entropy_with_logits(score["event_logit"].float(),
                                                            torch.tensor([positive], device=device))
            quality_loss = quality_focal_loss(score["quality_logit"].float(),
                torch.tensor([quality], device=device), config["loss"]["quality_beta"]).mean()
            candidate_boundary, boundary_legal = boundary_loss(model, candidate, target, times, features, device,
                boundary, score["candidate_embedding"][0], return_eligibility=True) if schedule["boundary"] and (
                    config.get("ablation", {}).get("name") != "no_boundary") else (score["event_logit"].sum() * 0, False)
            losses["event"].append(event_loss)
            losses["quality"].append(quality_loss)
            if boundary_legal:
                losses["boundary"].append(candidate_boundary)
            quality_pairs.append((ranking_quality, ranking_logit))
            version_tasks.append({"geometry": (candidate.start_ms, candidate.end_ms),
                "event": event_loss, "quality": quality_loss, "boundary": candidate_boundary,
                "boundary_legal": boundary_legal})
        ranking = []
        if config.get("ablation", {}).get("name") != "single_candidate":
            for first, (left_quality, left_score) in enumerate(quality_pairs):
                for second, (right_quality, right_score) in enumerate(quality_pairs[first + 1:], first + 1):
                    difference = left_quality - right_quality
                    if abs(difference) >= config["loss"]["ranking_minimum_iou_difference"]:
                        value = F.softplus(-(left_score - right_score) * (1 if difference > 0 else -1))
                        ranking.append(value)
                        pair_tasks.append({"geometries": (version_tasks[first]["geometry"],
                            version_tasks[second]["geometry"]), "quality_ranking": value})
        tasks = {name: torch.stack(values).mean() if values else scores[0]["event_logit"].sum() * 0
                 for name, values in losses.items()}
        tasks["quality_ranking"] = torch.stack(ranking).mean() if ranking else tasks["quality"] * 0
        tasks["_legal_auxiliary"] = {"boundary": bool(losses["boundary"]), "quality_ranking": bool(ranking)}
        tasks["_versions"], tasks["_pairs"] = version_tasks, pair_tasks
        results.append((group, tasks))
    return results, label_changes


def historical_contrastive(model, current, current_ids, current_phases, times, history, truth,
                           subject, session, config, *, ignore=None, diagnostics=None):
    from bme_eating.models.statsfusion_joint_event import event_contrastive_loss
    limit = config["loss"]["contrastive_max_pairs"]
    selected = truth[(truth.subject_key.astype(str) == subject) & (truth.session_id.astype(str) == session)]
    masks, identities, phases = [], [], []
    valid_history = history[:, 128] > 0
    if ignore is not None and len(ignore):
        excluded = ignore[(ignore.subject_key.astype(str) == subject) & (ignore.session_id.astype(str) == session)]
        for interval in excluded.itertuples(index=False):
            valid_history &= ~((times > interval.start_ms) & (times <= interval.end_ms))
    for index, event in enumerate(selected.itertuples(index=False)):
        eligible = (times > event.start_ms + 30_000) & (times < event.end_ms - 30_000) & valid_history
        indices = eligible.nonzero().flatten()
        if len(indices):
            indices = indices[torch.linspace(0, len(indices) - 1, min(len(indices), 12)).long()]
            masks.extend(indices.tolist())
            identities.extend([index] * len(indices))
            phases.extend((((times[indices] - event.start_ms) * 3) //
                           (event.end_ms - event.start_ms)).clamp(0, 2).tolist())
    background = valid_history.clone()
    for event in selected.itertuples(index=False):
        background &= (times <= event.start_ms - 30_000) | (times >= event.end_ms + 30_000)
    indices = background.nonzero().flatten()
    if len(indices):
        indices = indices[torch.linspace(0, len(indices) - 1, min(len(indices), 12)).long()]
        masks.extend(indices.tolist())
        identities.extend([-2] * len(indices))
        phases.extend([0] * len(indices))
    projected = model.contrastive_projection(current)
    chosen = bounded_history(masks, identities, phases, config["loss"]["contrastive_max_history_bins"])
    if diagnostics is not None:
        diagnostics.update({"positive_anchors": int(((current_ids >= 0) & (current_phases >= 0)).sum()),
            "history_positive_bins": sum(identity >= 0 for _, identity, _ in chosen),
            "history_background_bins": sum(identity == -2 for _, identity, _ in chosen),
            "history_discovered_bins": len(masks), "history_retained_bins": len(chosen),
            "history_truncated_bins": len(masks) - len(chosen)})
    if not chosen:
        result = event_contrastive_loss(projected, current_ids, current_phases,
            temperature=config["loss"]["contrastive_temperature"], maximum_pairs=limit)
        if diagnostics is not None:
            diagnostics.update({"valid_pairs": result[1], "zero_reason":
                "no_legal_current_positive_negative_pairs" if result[1] == 0 else None})
        return result
    mask = torch.zeros(len(times), dtype=torch.bool)
    mask[[item[0] for item in chosen]] = True
    with torch.no_grad():
        values = history[mask]
        historic = model.contrastive_projection(values[:, :64].to(current.device)).detach()
    all_values = F.normalize(torch.cat((projected, historic)), dim=-1)
    all_ids = torch.cat((current_ids, torch.tensor([item[1] for item in chosen], device=current.device)))
    all_phases = torch.cat((current_phases, torch.tensor([item[2] for item in chosen], device=current.device)))
    losses = []
    anchors = ((current_ids >= 0) & (current_phases >= 0)).nonzero().flatten()[:limit]
    for anchor in anchors.tolist():
        positive = (all_ids == all_ids[anchor]) & (all_phases == all_phases[anchor])
        positive[anchor] = False
        negative = ((all_ids >= 0) | (all_ids == -2)) & (all_phases >= 0) & (all_ids != all_ids[anchor])
        if positive.any() and negative.any():
            similarities = all_values[anchor] @ all_values.T / config["loss"]["contrastive_temperature"]
            losses.append(-(similarities[positive] - torch.logsumexp(similarities[positive | negative], 0)).mean())
    if diagnostics is not None:
        diagnostics.update({"valid_pairs": len(losses), "zero_reason":
            "no_legal_positive_negative_pairs" if not losses else None})
    return (torch.stack(losses).mean(), len(losses)) if losses else (projected.sum() * 0, 0)


def bounded_history(indices, identities, phases, maximum):
    unique = {}
    for index, identity, phase in zip(indices, identities, phases, strict=True):
        if index in unique and unique[index] != (identity, phase):
            raise RuntimeError("Conflicting historical contrastive labels")
        unique[index] = identity, phase
    records = sorted((index, *label) for index, label in unique.items())
    if len(records) <= maximum:
        return records
    selected = [record for record in records if record[1] == -2]
    if len(selected) > maximum:
        raise RuntimeError("Historical background reservation exceeds reference limit")
    buckets = {}
    for record in records:
        if record[1] >= 0:
            buckets.setdefault(record[1:], []).append(record)
    counts = dict.fromkeys(buckets, 0)
    available = maximum - len(selected)
    while available and any(counts[key] < len(bucket) for key, bucket in buckets.items()):
        for key in sorted(buckets):
            if available and counts[key] < len(buckets[key]):
                counts[key] += 1
                available -= 1
    for key, count in counts.items():
        if count:
            positions = np.linspace(0, len(buckets[key]) - 1, count).astype(int)
            selected.extend(buckets[key][position] for position in positions)
    return sorted(selected)


def subject_weights(session_keys):
    counts = Counter(key[0] for key in session_keys)
    lower, upper = 0.0, max(math.sqrt(count) * 2 for count in counts.values())
    for _ in range(80):
        middle = (lower + upper) / 2
        average = sum(count * np.clip(middle / math.sqrt(count), .5, 2)
                      for count in counts.values()) / sum(counts.values())
        if average < 1:
            lower = middle
        else:
            upper = middle
    constant = (lower + upper) / 2
    return {subject: float(np.clip(constant / math.sqrt(count), .5, 2))
            for subject, count in counts.items()}


def learning_rate_fraction(epoch_position, training):
    warmup = training["lr_warmup_epochs"]
    horizon = training["lr_horizon_epochs"]
    floor = training["lr_minimum_fraction"]
    position = min(horizon, max(0, epoch_position))
    if position <= warmup:
        return .1 + .9 * position / warmup
    return floor + (1 - floor) * (1 + math.cos(math.pi * (position - warmup) / (horizon - warmup))) / 2


def apply_learning_rate(optimizer, state, training, epoch_position):
    fraction = learning_rate_fraction(epoch_position, training)
    for group, base in zip(optimizer.param_groups,
                           (training["encoder_learning_rate"], training["head_learning_rate"])):
        group["lr"] = base * fraction
    state["lr_position"] = epoch_position
    state["learning_rates"] = [group["lr"] for group in optimizer.param_groups]


def describe_group(group, truth):
    versions = group["versions"]
    qualities = [candidate_targets(candidate, truth)[1] for candidate in versions]
    return {"id": stable_id(group["event_key"], sorted(candidate_order_key(item) for item in versions)),
            "maximum_iou": max(qualities), "bins": sorted({min(2, math.ceil(quality * 4) - 2)
                for quality in qualities if quality > .25}),
            "decision": max(item.decision_time_ms for item in versions),
            "geometries": [(item.start_ms, item.end_ms) for item in versions],
            "qualities": qualities}


def retained_groups(descriptors, maximum=3, *, diagnostics=None):
    remaining = {item["id"]: deepcopy(item) for item in descriptors if item["maximum_iou"] > .25}
    selected = []
    while remaining and len(selected) < maximum:
        covered = {band for item in selected for band in item["bins"]}
        if not selected:
            def key(item):
                return -item["maximum_iou"], item["id"]
        else:
            def key(item, covered=covered, selected=tuple(selected)):
                return (-len(set(item["bins"]) - covered),
                        -min(abs(item["decision"] - previous["decision"]) for previous in selected),
                        -item["maximum_iou"], item["id"])
        item = min(remaining.values(), key=key)
        selected.append(item)
        remaining.pop(item["id"])
        used = set(map(tuple, item["geometries"]))
        for candidate in remaining.values():
            qualities = candidate.get("qualities", [candidate["maximum_iou"]] * len(candidate["geometries"]))
            unique = [(geometry, quality) for geometry, quality in zip(candidate["geometries"], qualities)
                      if tuple(geometry) not in used]
            candidate["geometries"] = [geometry for geometry, _ in unique]
            candidate["qualities"] = [quality for _, quality in unique]
            candidate["maximum_iou"] = max(candidate["qualities"], default=0)
            candidate["bins"] = sorted({min(2, math.ceil(quality * 4) - 2)
                for quality in candidate["qualities"] if quality > .25})
            if candidate["maximum_iou"] <= .25 and diagnostics is not None:
                diagnostics.append({"group": candidate["id"], "reason": "no_positive_after_dedup",
                                    "remaining_versions": len(candidate["geometries"])})
        remaining = {identity: candidate for identity, candidate in remaining.items()
                     if candidate["geometries"] and candidate["maximum_iou"] > .25}
    return selected


def _versions(candidates, truth, maximum):
    ordered = sorted(candidates, key=lambda item: (-candidate_targets(item, truth)[1], candidate_order_key(item)))
    selected, bins = [], set()
    for item in ordered:
        band = min(3, max(0, math.ceil(candidate_targets(item, truth)[1] * 4) - 1))
        if band not in bins:
            selected.append(item)
            bins.add(band)
        if len(selected) == maximum:
            break
    for item in ordered:
        if len(selected) == maximum:
            break
        if item not in selected:
            selected.append(item)
    return selected


def _stratum(group):
    candidate = group["versions"][0]
    return (candidate.source.split("_")[0], int((candidate.end_ms - candidate.start_ms) // 300_000),
            candidate.wear_hand, candidate.hand_relation or "unknown", candidate.gyro_missing)


def _rotate_strata(groups, rng):
    strata = {}
    for group in sorted(groups, key=lambda item: candidate_order_key(item["versions"][0])):
        strata.setdefault(_stratum(group), []).append(group)
    keys = sorted(strata, key=str)
    rng.shuffle(keys)
    for pool in strata.values():
        rng.shuffle(pool)
    result = []
    while any(strata.values()):
        for key in keys:
            if strata[key]:
                result.append(strata[key].pop())
    return result


def sample_groups(model, candidates, truth, config, rng, positive_state, used_families,
                  times, features, device, teacher_fraction=0, *, performance=None, locked_geometries=None):
    started = time.monotonic()
    families = {}
    unique = {}
    owned = {(*event[:2], *geometry) for event, groups in positive_state.items()
             for group in groups.values() for geometry in group["descriptor"]["geometries"]
             if isinstance(event, tuple)}
    locked_geometries = locked_geometries or set()
    for candidate in sorted(candidates, key=candidate_order_key):
        key = geometry_key(candidate)
        if key in locked_geometries:
            continue
        if key in unique:
            previous = unique[key]
            previous.generation_parameters["source_flags"] = sorted(set(
                previous.generation_parameters.get("source_flags", [previous.source])
                + candidate.generation_parameters.get("source_flags", [candidate.source])))
        else:
            unique[key] = candidate
    for candidate in unique.values():
        families.setdefault(candidate.proposal_family_id, {})[(candidate.start_ms, candidate.end_ms)] = candidate
    positives, negatives = [], {"near_miss": [], "background": []}
    for family, geometries in families.items():
        partitioned = {}
        positive_events = {candidate_targets(item, truth)[2] for item in geometries.values()
                           if candidate_targets(item, truth)[0]}
        for item in geometries.values():
            target = candidate_targets(item, truth)
            event = target[2] if target[2] in positive_events else None
            partitioned.setdefault(event, []).append(item)
        for event, items in partitioned.items():
            versions = _versions(items, truth, config["training"]["versions_per_group"])
            representative = versions[0]
            key = (representative.subject_key, representative.session_id, event)
            group = {"family": family, "versions": versions, "event_key": key}
            if event is not None:
                group["category"] = "positive"
                group["descriptor"] = describe_group(group, truth)
                positives.append(group)
            elif ((representative.subject_key, representative.session_id), family) not in used_families:
                group["versions"] = [item for item in versions if geometry_key(item) not in owned]
                if not group["versions"]:
                    continue
                category = "near_miss" if max(candidate_targets(item, truth)[1]
                    for item in group["versions"]) > 0 else "background"
                group["category"] = category
                negatives[category].append(group)
    discovered_events = len({group["event_key"] for group in positives})
    selected, reservations, retention_drops = [], {}, []
    teacher_limit = int(2 * teacher_fraction + .5)
    teacher_count = 0
    while positives and len(selected) < 2:
        group = min(positives, key=lambda item: (bool(reservations.get(item["event_key"])
                      or positive_state.get(item["event_key"])),
                      -item["descriptor"]["maximum_iou"], item["descriptor"]["id"]))
        positives.remove(group)
        teacher = any(item.source == "truth_jitter" for item in group["versions"])
        if teacher and teacher_count >= teacher_limit:
            continue
        key = group["event_key"]
        existing = reservations.get(key, [item.get("original_descriptor", item["descriptor"])
                                          for item in positive_state.get(key, {}).values()])
        retained = retained_groups([*existing, group["descriptor"]], diagnostics=retention_drops)
        retained_ids = {item["id"] for item in retained}
        if group["descriptor"]["id"] in retained_ids and group["descriptor"]["id"] not in {item["id"] for item in existing}:
            reservations[key] = [item for item in [*existing, group["descriptor"]] if item["id"] in retained_ids]
            group["retained_ids"] = retained_ids
            selected.append(group)
            teacher_count += int(teacher)
    near = _rotate_strata(negatives["near_miss"], rng)
    background = _rotate_strata(negatives["background"], rng)
    maximum = config["training"]["mining_families"]
    reserve = config["training"]["mining_near_miss_families"]
    stabilized = supported(config, "stabilized")
    anchor = background.pop(0) if stabilized and background else None
    pool = [anchor] if anchor else []
    pool.extend(near[:reserve])
    pool.extend(background[:maximum - len(pool)])
    if len(pool) < maximum:
        pool.extend(near[reserve:reserve + maximum - len(pool)])
    mining_versions = {group["family"]: _versions(group["versions"], truth,
        config["training"]["mining_versions_per_family"]) for group in pool}
    flat = [item for group in pool for item in mining_versions[group["family"]]]
    training_state = model.training
    try:
        model.eval()
        with training_inference(device, cache_safe=stabilized):
            scoring_options = {"performance": performance} if performance is not None else {}
            scores = score_candidates(model, flat, times, features, device,
                                      batch_size=max(1, len(flat)), **scoring_options)
            difficulty = {candidate.proposal_id: float(F.softplus(score["event_logit"].float()).mean())
                          for candidate, score in zip(flat, scores)}
    finally:
        model.train(training_state)
    for group in pool:
        group["difficulty"] = float(np.mean([difficulty[item.proposal_id]
            for item in mining_versions[group["family"]]]))
    ranked = sorted(pool, key=lambda group: (-group["difficulty"], candidate_order_key(group["versions"][0])))
    chosen_families = set()
    if anchor:
        chosen_families.add(anchor["family"])
        selected.append({**anchor, "category": "random_background", "selection_reason": "reserved_before_mining"})
    quotas = (("near_miss", 2), ("hard_false_positive", 3)) if stabilized else (
        ("near_miss", 2), ("hard_false_positive", 3), ("random_background", 1))
    for category, quota in quotas:
        eligible = [group for group in ranked if group["family"] not in chosen_families
                    and (group["category"] == "near_miss") == (category == "near_miss")]
        if category == "random_background":
            rng.shuffle(eligible)
        for group in eligible[:quota]:
            chosen_families.add(group["family"])
            selected.append({**group, "category": category})
    if stabilized:
        for group in ranked:
            if len(chosen_families) >= 6:
                break
            if group["family"] not in chosen_families:
                chosen_families.add(group["family"])
                selected.append({**group, "category": "near_miss" if group["category"] == "near_miss" else "hard_false_positive",
                                 "selection_reason": "quota_shortfall_replacement"})
    counts = Counter(group["category"] for group in selected)
    return selected, {"groups": len(selected), "versions": sum(len(group["versions"]) for group in selected),
        "independent_positive_events": len({group["event_key"] for group in selected if group["category"] == "positive"}),
        "independent_negative_families": len(chosen_families), "mining_families": len(pool),
        "mining_head_candidates": len(flat), "mining_seconds": time.monotonic() - started,
        "mining_versions_per_family": config["training"]["mining_versions_per_family"],
        "supervised_versions_per_group": config["training"]["versions_per_group"],
        "random_anchor_reserved": anchor is not None, "discovered_positive_events": discovered_events,
        "positive_retention_drops": retention_drops,
        "mining_ranking": [{"family": group["family"], "bce": group["difficulty"],
                             "category": group["category"]} for group in ranked],
        "categories": {category: {"requested_groups": quota, "actual_groups": counts[category],
                                    "deficit": max(0, quota - counts[category]),
                                    "surplus": max(0, counts[category] - quota)}
                       for category, quota in config["training"]["group_quotas"].items()}}


def heatmap_focal_loss(logits, targets, valid, positive_count, background_count,
                      *, peak_scale=1.0, background_scale=1.0):
    return sum(heatmap_focal_components(logits, targets, valid, positive_count, background_count,
        peak_scale=peak_scale, background_scale=background_scale).values())


def heatmap_focal_components(logits, targets, valid, positive_count, background_count,
                             *, peak_scale=1.0, background_scale=1.0):
    logits, targets, valid = logits.float(), targets.float(), valid.float()
    probability = logits.sigmoid()
    peaks = F.softplus(-logits) * (1 - probability).square() * (targets == 1) * valid
    background = F.softplus(logits) * probability.square() * (1 - targets).pow(4) * (targets < 1) * valid
    return {"peak": peak_scale * peaks.sum() / positive_count.sum().clamp_min(1) / 21,
            "background": 20 * background_scale * background.sum() / background_count.sum().clamp_min(1) / 21}


def selection_update(state, epoch, eligible, f1, training):
    state.setdefault("unqualified_checks", 0)
    state.setdefault("lost_qualification_checks", 0)
    improved = eligible and (state["best_epoch"] is None or f1 > state["best_f1"] + training["f1_min_delta"])
    if eligible:
        state["lost_qualification_checks"] = 0
        if improved:
            state["best_epoch"], state["best_f1"], state["stale_checks"] = epoch, f1, 0
        else:
            state["stale_checks"] += 1
    else:
        state["unqualified_checks"] += 1
        if state["best_epoch"] is not None:
            state["lost_qualification_checks"] += 1
    failure = ("qualification_deadline" if state["best_epoch"] is None and
               epoch >= training["qualification_deadline_epoch"] else
               "qualification_lost" if state["lost_qualification_checks"] >= 3 else None)
    stop = bool(failure or state["stale_checks"] >= training["patience_checks"])
    state["selection_failure"] = failure
    return improved, stop


def sampling_counts(groups):
    return dict(Counter(group["category"] for group in groups))
