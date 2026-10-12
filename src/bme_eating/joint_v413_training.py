from __future__ import annotations

from collections import Counter

import torch
from torch.nn import functional as F

from bme_eating.joint_v412_training import retained_groups


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
