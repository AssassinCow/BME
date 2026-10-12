from __future__ import annotations

import torch
from torch.nn import functional as F

from bme_eating.joint_amp import training_inference
from bme_eating.joint_capabilities import supported
from bme_eating.joint_stream import candidate_targets
from bme_eating.joint_v411_scoring import boundary_loss, refine, score_candidates
from bme_eating.models.statsfusion_joint_event import quality_focal_loss


def sample_groups(candidates, truth, config, rng, counts, teacher_fraction=0):
    groups = {}
    for candidate in candidates:
        parent = (candidate.subject_key, candidate.session_id, candidate.ancestor_id)
        groups.setdefault(parent, {})[(candidate.start_ms, candidate.end_ms)] = candidate
    buckets = {name: [] for name in ("positive", "near_miss", "hard_false_positive", "random_background")}
    quotas = config["training"]["group_quotas"]
    if config.get("ablation", {}).get("name") == "single_candidate":
        maximum_versions = 1
    else:
        maximum_versions = config["training"]["versions_per_group"]
    maximum_groups = (1 if maximum_versions == 1 else config["training"]["maximum_groups_per_positive_event"])
    event_groups = []
    for versions in groups.values():
        versions = list(versions.values())
        targets = {item.proposal_id: candidate_targets(item, truth) for item in versions}
        positive_events = {target[2] for target in targets.values() if target[0]}
        partitioned = {}
        for item in versions:
            target = targets[item.proposal_id]
            event = target[2] if target[2] in positive_events else None
            partitioned.setdefault(event, []).append(item)
        event_groups.extend(partitioned.values())
    for versions in event_groups:
        targets = {item.proposal_id: candidate_targets(item, truth) for item in versions}
        representative = max(versions, key=lambda item: (targets[item.proposal_id][1], item.proposal_id))
        target = targets[representative.proposal_id]
        category = "positive" if target[0] else "near_miss" if target[1] > 0 else (
            "random_background" if representative.generation_parameters.get("sampling") == "random_background"
            or representative.raw_score < .10 else "hard_false_positive")
        identity = (representative.subject_key, representative.session_id, target[2])
        if category == "positive" and counts.get(identity, 0) >= maximum_groups:
            continue
        bins, selected = set(), []
        for candidate in sorted(versions, key=lambda item: (-targets[item.proposal_id][1], item.proposal_id)):
            iou_bin = int(targets[candidate.proposal_id][1] * 10)
            if iou_bin not in bins:
                selected.append(candidate)
                bins.add(iou_bin)
            if len(selected) == maximum_versions:
                break
        for candidate in sorted(versions, key=lambda item: item.proposal_id):
            if len(selected) >= maximum_versions:
                break
            if candidate not in selected:
                selected.append(candidate)
        buckets[category].append({"versions": selected, "category": category,
                                  "event_key": identity, "family": representative.proposal_family_id})
    chosen, negative_families, audit = [], set(), {}
    for category, quota in quotas.items():
        pool = buckets[category]
        rng.shuffle(pool)
        selected = 0
        teacher_count = 0
        for group in pool:
            teacher = any(item.source == "truth_jitter" for item in group["versions"])
            if teacher and teacher_count >= int(quota * teacher_fraction + .5):
                continue
            if category == "positive":
                key = group["event_key"]
                if counts.get(key, 0) >= maximum_groups:
                    continue
                counts[key] = counts.get(key, 0) + 1
            elif group["family"] in negative_families:
                continue
            else:
                negative_families.add(group["family"])
            chosen.append(group)
            selected += 1
            teacher_count += int(teacher)
            if selected == quota:
                break
        audit[category] = {"requested_groups": quota, "actual_groups": selected,
                           "deficit": quota - selected}
    return chosen, {"groups": len(chosen), "versions": sum(len(group["versions"]) for group in chosen),
                    "independent_positive_events": len({group["event_key"] for group in chosen if group["category"] == "positive"}),
                    "independent_negative_families": len(negative_families), "categories": audit}


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
                    if supported(model, "consistent_inputs"):
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
        if supported(model, "stabilized"):
            losses["boundary"] = [value for value in losses["boundary"] if float(value.detach()) > 0]
        tasks = {name: torch.stack(values).mean() if values else scores[0]["event_logit"].sum() * 0
                 for name, values in losses.items()}
        tasks["quality_ranking"] = torch.stack(ranking).mean() if ranking else tasks["quality"] * 0
        results.append((group, tasks))
    return results, label_changes


class PositiveGradientBuffer:
    def __init__(self, state):
        self.state = state

    def add(self, key, loss, model):
        parameters = list(model.named_parameters())
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters],
                                        retain_graph=True, allow_unused=True)
        event = self.state.setdefault(key, {"groups": 0, "gradients": {}})
        event["groups"] += 1
        for (name, _), gradient in zip(parameters, gradients):
            if gradient is not None:
                value = gradient.detach().float().cpu()
                if name not in event["gradients"]:
                    event["gradients"][name] = value.clone()
                else:
                    event["gradients"][name].add_(value)

    def finish(self, model, accumulate_sessions):
        count = len(self.state)
        for name, parameter in model.named_parameters():
            values = [event["gradients"][name] / event["groups"] for event in self.state.values()
                      if name in event["gradients"]]
            if values:
                gradient = torch.stack(values).sum(0).to(parameter.device) / (count * accumulate_sessions)
                if parameter.grad is None:
                    parameter.grad = gradient
                else:
                    parameter.grad.add_(gradient)
        self.state.clear()


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
