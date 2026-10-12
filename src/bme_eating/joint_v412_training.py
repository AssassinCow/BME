from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from bme_eating.joint_amp import training_inference
from bme_eating.joint_capabilities import supported
from bme_eating.joint_stream import candidate_targets, stable_id
from bme_eating.joint_v411_scoring import score_candidates


def heatmap_focal_loss(logits, targets, valid, positive_count, background_count):
    logits, targets, valid = logits.float(), targets.float(), valid.float()
    probability = logits.sigmoid()
    positive = (targets == 1).float() * valid
    background = (targets < 1).float() * valid
    positive_loss = F.softplus(-logits) * (1 - probability).square() * positive
    background_loss = F.softplus(logits) * probability.square() * (1 - targets).pow(4) * background
    return (positive_loss.sum() / positive_count.sum().clamp_min(1)
            + background_loss.sum() / background_count.sum().clamp_min(1))


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
    return {"id": stable_id(group["family"], group["event_key"],
                            sorted((item.start_ms, item.end_ms) for item in versions)),
            "maximum_iou": max(qualities), "bins": sorted({min(2, math.ceil(quality * 4) - 2)
                for quality in qualities if quality > .25}),
            "decision": max(item.decision_time_ms for item in versions),
            "geometries": sorted((item.start_ms, item.end_ms) for item in versions)}


def retained_groups(descriptors, maximum=3):
    remaining = {item["id"]: item for item in descriptors}
    selected = []
    while remaining and len(selected) < maximum:
        covered = {band for item in selected for band in item["bins"]}
        if not selected:
            key = lambda item: (-item["maximum_iou"], item["id"])
        else:
            key = lambda item, covered=covered, selected=tuple(selected): (-len(set(item["bins"]) - covered),
                -min(abs(item["decision"] - previous["decision"]) for previous in selected),
                -item["maximum_iou"], item["id"])
        item = min(remaining.values(), key=key)
        selected.append(item)
        remaining.pop(item["id"])
        used = set(map(tuple, item["geometries"]))
        remaining = {identity: candidate for identity, candidate in remaining.items()
                     if not used.intersection(map(tuple, candidate["geometries"]))}
    return selected


def _versions(candidates, truth):
    ordered = sorted(candidates, key=lambda item: (-candidate_targets(item, truth)[1], item.proposal_id))
    selected, bins = [], set()
    for item in ordered:
        band = min(3, max(0, math.ceil(candidate_targets(item, truth)[1] * 4) - 1))
        if band not in bins:
            selected.append(item)
            bins.add(band)
        if len(selected) == 3:
            break
    for item in ordered:
        if len(selected) == 3:
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
    for group in sorted(groups, key=lambda item: item["family"]):
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
                  times, features, device, teacher_fraction=0, *, performance=None):
    started = time.monotonic()
    families = {}
    for candidate in candidates:
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
            versions = _versions(items, truth)
            representative = versions[0]
            key = (representative.subject_key, representative.session_id, event)
            group = {"family": family, "versions": versions, "event_key": key}
            if event is not None:
                group["category"] = "positive"
                group["descriptor"] = describe_group(group, truth)
                positives.append(group)
            elif ((representative.subject_key, representative.session_id), family) not in used_families:
                category = "near_miss" if max(candidate_targets(item, truth)[1] for item in versions) > 0 else "background"
                group["category"] = category
                negatives[category].append(group)
    discovered_events = len({group["event_key"] for group in positives})
    selected, reservations = [], {}
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
        existing = reservations.get(key, [item["descriptor"] for item in positive_state.get(key, {}).values()])
        retained = retained_groups([*existing, group["descriptor"]])
        retained_ids = {item["id"] for item in retained}
        if group["descriptor"]["id"] in retained_ids and group["descriptor"]["id"] not in {item["id"] for item in existing}:
            reservations[key] = retained
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
    flat = [item for group in pool for item in group["versions"]]
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
        group["difficulty"] = float(np.mean([difficulty[item.proposal_id] for item in group["versions"]]))
    ranked = sorted(pool, key=lambda group: (-group["difficulty"], group["family"]))
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
        "random_anchor_reserved": anchor is not None, "discovered_positive_events": discovered_events,
        "mining_ranking": [{"family": group["family"], "bce": group["difficulty"],
                             "category": group["category"]} for group in ranked],
        "categories": {category: {"requested_groups": quota, "actual_groups": counts[category],
                                    "deficit": max(0, quota - counts[category]),
                                    "surplus": max(0, counts[category] - quota)}
                       for category, quota in config["training"]["group_quotas"].items()}}


class ReplaceablePositiveGradientBuffer:
    def __init__(self, state):
        self.state = state
        for groups in state.values():
            for group in groups.values():
                group["gradients"] = {name: value.detach().float().cpu()
                                      for name, value in group["gradients"].items()}

    def add(self, group, loss, model):
        key = group["event_key"]
        parameters = list(model.named_parameters())
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters],
                                        retain_graph=True, allow_unused=True)
        previous = self.state.setdefault(key, {})
        descriptor = group["descriptor"]
        retained = {item["id"] for item in retained_groups(
            [*[item["descriptor"] for item in previous.values()], descriptor])}
        if descriptor["id"] not in retained:
            return
        for identity in list(previous):
            if identity not in retained:
                previous.pop(identity)
        previous[descriptor["id"]] = {"descriptor": descriptor,
            "gradients": {name: gradient.detach().float().cpu().clone()
                          for (name, _), gradient in zip(parameters, gradients) if gradient is not None}}
        return {"retained": sorted(retained), "added": descriptor["id"]}

    def finish(self, model, accumulate_sessions):
        count = len(self.state)
        for name, parameter in model.named_parameters():
            values = []
            for groups in self.state.values():
                contributions = [group["gradients"][name] for group in groups.values()
                                 if name in group["gradients"]]
                if contributions:
                    values.append(sum(contributions) / len(groups))
            if values:
                gradient = sum(values).to(parameter.device) / (count * accumulate_sessions)
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        self.state.clear()

    def tensors(self):
        return (value for groups in self.state.values() for group in groups.values()
                for value in group["gradients"].values())
