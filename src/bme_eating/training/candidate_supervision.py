from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from bme_eating.amp import training_inference
from bme_eating.candidate_scoring import boundary_loss, refine, score_candidates
from bme_eating.candidates import candidate_targets, stable_id
from bme_eating.models.statsfusion_joint_event import quality_focal_loss
from bme_eating.protocol import supported


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
        quality_pairs = []
        for candidate in group["versions"]:
            score = score_by_id[candidate.proposal_id]
            target = candidate_targets(candidate, truth)
            positive, quality = target[:2]
            ranking_quality, ranking_logit = quality, score["quality_logit"].float().squeeze()
            event_loss = F.binary_cross_entropy_with_logits(score["event_logit"].float(),
                                                            torch.tensor([positive], device=device))
            quality_loss = quality_focal_loss(score["quality_logit"].float(),
                torch.tensor([quality], device=device), config["loss"]["quality_beta"]).mean()
            candidate_boundary = boundary_loss(model, candidate, target, times, features, device,
                boundary, score["candidate_embedding"][0]) if schedule["boundary"] and (
                    config.get("ablation", {}).get("name") != "no_boundary") else score["event_logit"].sum() * 0
            if schedule["boundary"] and config.get("ablation", {}).get("name") != "no_boundary":
                with training_inference(device, cache_safe=supported(model, "stabilized")):
                    refined = refine(model, candidate, times, features, device,
                        {**boundary, "qualified": True, "mode": "both"}, .65,
                        embedding=score["candidate_embedding"][0].detach(), rescore=False)
                if refined.refined:
                    new_target = candidate_targets(refined, truth)
                    new_score = score_candidates(model, [refined], times, features, device, **scoring_options)[0]
                    event_loss = (event_loss + F.binary_cross_entropy_with_logits(
                        new_score["event_logit"].float(), torch.tensor([new_target[0]], device=device))) / 2
                    quality_loss = (quality_loss + quality_focal_loss(new_score["quality_logit"].float(),
                        torch.tensor([new_target[1]], device=device), config["loss"]["quality_beta"]).mean()) / 2
                    candidate_boundary = (candidate_boundary + boundary_loss(model, refined, new_target,
                        times, features, device, boundary, new_score["candidate_embedding"][0])) / 2
                    label_changes.append({"proposal_id": candidate.proposal_id,
                        "before_target": list(target), "after_target": list(new_target)})
                    ranking_quality, ranking_logit = new_target[1], new_score["quality_logit"].float().squeeze()
            losses["event"].append(event_loss)
            losses["quality"].append(quality_loss)
            losses["boundary"].append(candidate_boundary)
            quality_pairs.append((ranking_quality, ranking_logit))
        ranking = []
        if config.get("ablation", {}).get("name") != "single_candidate":
            for first, (left_quality, left_score) in enumerate(quality_pairs):
                for right_quality, right_score in quality_pairs[first + 1:]:
                    difference = left_quality - right_quality
                    if abs(difference) >= config["loss"]["ranking_minimum_iou_difference"]:
                        ranking.append(F.softplus(-(left_score - right_score) * (1 if difference > 0 else -1)))
        losses["boundary"] = [value for value in losses["boundary"] if float(value.detach()) > 0]
        tasks = {name: torch.stack(values).mean() if values else scores[0]["event_logit"].sum() * 0
                 for name, values in losses.items()}
        tasks["quality_ranking"] = torch.stack(ranking).mean() if ranking else tasks["quality"] * 0
        results.append((group, tasks))
    return results, label_changes


def historical_contrastive(model, current, current_ids, current_phases, times, history, truth,
                           subject, session, config, *, ignore=None):
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
    projected = model.contrastive_projection(current)
    if not masks:
        return event_contrastive_loss(projected, current_ids, current_phases,
            temperature=config["loss"]["contrastive_temperature"], maximum_pairs=limit)
    chosen = sorted(zip(masks, identities, phases))[:limit]
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
    return (torch.stack(losses).mean(), len(losses)) if losses else (projected.sum() * 0, 0)


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


def heatmap_focal_loss(logits, targets, valid, positive_count, background_count):
    logits, targets, valid = logits.float(), targets.float(), valid.float()
    probability = logits.sigmoid()
    peaks = F.softplus(-logits) * (1 - probability).square() * (targets == 1) * valid
    background = F.softplus(logits) * probability.square() * (1 - targets).pow(4) * (targets < 1) * valid
    return (peaks.sum() / positive_count.sum().clamp_min(1)
            + 20 * background.sum() / background_count.sum().clamp_min(1)) / 21


def cpu_gradients(loss, model):
    parameters = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters],
                                    retain_graph=True, allow_unused=True)
    return {name: gradient.detach().float().cpu().clone()
            for (name, _), gradient in zip(parameters, gradients) if gradient is not None}


def add_gradients(destination, gradients, divisor=1):
    for name, value in gradients.items():
        if name in destination:
            destination[name].add_(value, alpha=1 / divisor)
        else:
            destination[name] = value.clone() / divisor


class ContributionBank:
    def __init__(self, state):
        self.state = state
        state.setdefault("negative_task_gradients", {})
        state.setdefault("negative_family_count", 0)
        state.setdefault("pending_task_gradients", {})
        state.setdefault("pending_task_sessions", {})

    def negative(self, loss, model):
        add_gradients(self.state["negative_task_gradients"], cpu_gradients(loss, model))
        self.state["negative_family_count"] += 1

    def pending(self, name, gradients):
        if gradients:
            destination = self.state["pending_task_gradients"].setdefault(name, {})
            add_gradients(destination, gradients)
            counts = self.state["pending_task_sessions"]
            counts[name] = counts.get(name, 0) + 1

    def finish_session(self, positive):
        for task in ("classification", "boundary", "quality_ranking"):
            event_gradients = []
            for groups in positive.values():
                legal = [group["task_gradients"][task] for group in groups.values()
                         if task in group["task_gradients"]]
                if legal:
                    gradients = {}
                    for contribution in legal:
                        add_gradients(gradients, contribution, len(legal))
                    event_gradients.append(gradients)
            session = {}
            for event in event_gradients:
                add_gradients(session, event, len(event_gradients))
            self.pending("positive" if task == "classification" else task, session)
        if self.state["negative_family_count"]:
            self.pending("negative", {name: value / self.state["negative_family_count"]
                for name, value in self.state["negative_task_gradients"].items()})
        self.state["last_session_contributions"] = {
            "positive_events": len(positive), "negative_families": self.state["negative_family_count"],
            "groups": sum(map(len, positive.values()))}
        positive.clear()
        self.state["negative_task_gradients"] = {}
        self.state["negative_family_count"] = 0

    def commit(self, model):
        pending, counts = self.state["pending_task_gradients"], self.state["pending_task_sessions"]
        classes = sum(counts.get(name, 0) > 0 for name in ("positive", "negative"))
        parameters = dict(model.named_parameters())
        if classes:
            classification = [pending.get(task, {}) for task in ("positive", "negative")]
            for name in ("candidate_head.event.weight", "candidate_head.event.bias",
                         "candidate_head.quality.weight", "candidate_head.quality.bias"):
                if name in parameters and not any(name in gradients for gradients in classification):
                    raise RuntimeError(f"Candidate classification gradient disconnected: {name}")
        if counts.get("boundary", 0):
            for name in parameters:
                if name.startswith("boundary_head.") and name not in pending.get("boundary", {}):
                    raise RuntimeError(f"Candidate boundary gradient disconnected: {name}")
        for task, gradients in pending.items():
            divisor = counts[task] * (classes if task in {"positive", "negative"} else 1)
            for name, value in gradients.items():
                gradient = value.to(parameters[name].device) / divisor
                if parameters[name].grad is None:
                    parameters[name].grad = gradient
                else:
                    parameters[name].grad.add_(gradient)
        self.state["last_update_contributions"] = dict(counts)
        pending.clear()
        counts.clear()

    def tensors(self):
        return (value for bank in (self.state["negative_task_gradients"],
            *self.state["pending_task_gradients"].values()) for value in bank.values())


class PositiveGradientBuffer:
    def __init__(self, state):
        self.state = state

    def add(self, group, tasks, model):
        groups = self.state.setdefault(group["event_key"], {})
        descriptor = group["descriptor"]
        retained = {item["id"] for item in retained_groups(
            [*[item["descriptor"] for item in groups.values()], descriptor])}
        if descriptor["id"] not in retained:
            return {"added": None, "removed": []}
        removed = sorted(set(groups) - retained)
        for identity in removed:
            groups.pop(identity)
        groups[descriptor["id"]] = {"descriptor": descriptor,
            "task_gradients": {name: cpu_gradients(loss, model) for name, loss in tasks.items()}}
        return {"added": descriptor["id"], "removed": removed, "retained": sorted(retained)}

    def tensors(self):
        return (value for groups in self.state.values() for group in groups.values()
                for gradients in group["task_gradients"].values() for value in gradients.values())


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
