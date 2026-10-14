from __future__ import annotations

import torch


def add_gradients(destination, gradients, divisor=1):
    for name, value in gradients.items():
        if name in destination:
            destination[name].add_(value, alpha=1 / divisor)
        else:
            destination[name] = value.clone() / divisor


def cpu_gradients(loss, model):
    parameters = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters],
                                    retain_graph=True, allow_unused=True)
    if not all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None):
        raise RuntimeError("Nonfinite candidate task gradient")
    return {name: gradient.detach().float().cpu().clone()
            for (name, _), gradient in zip(parameters, gradients) if gradient is not None}


def atomic_gradients(losses, model):
    parameters = list(model.named_parameters())
    results, fallback = [], False
    for first in range(0, len(losses), 3):
        chunk = losses[first:first + 3]
        if not all(torch.isfinite(loss).all() for loss in chunk):
            raise RuntimeError("Nonfinite candidate task loss")
        try:
            values = torch.stack([loss.reshape(()) for loss in chunk])
            gradients = torch.autograd.grad(values, [value for _, value in parameters],
                grad_outputs=torch.eye(len(chunk), device=values.device, dtype=values.dtype),
                retain_graph=True, allow_unused=True, is_grads_batched=True)
            if not all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None):
                raise RuntimeError("Nonfinite atomic candidate gradient")
            results.extend({name: gradient[index].detach().float().cpu().clone()
                for (name, _), gradient in zip(parameters, gradients) if gradient is not None}
                for index in range(len(chunk)))
        except RuntimeError as error:
            if not any(word in str(error).lower() for word in ("vmap", "batching", "is_grads_batched")):
                raise
            fallback = True
            results.extend(cpu_gradients(loss, model) for loss in chunk)
    return results, fallback


def materialize(group):
    active = set(map(tuple, group["descriptor"]["geometries"]))
    tasks, values, counts = {}, {}, {}
    for atom in group["atoms"]:
        if not set(map(tuple, atom["geometries"])).issubset(active):
            continue
        for task, gradients in atom["gradients"].items():
            add_gradients(tasks.setdefault(task, {}), gradients)
            values[task] = values.get(task, 0.0) + atom["values"][task]
            counts[task] = counts.get(task, 0) + 1
    group["task_gradients"] = {task: {name: value / counts[task] for name, value in gradients.items()}
                               for task, gradients in tasks.items()}
    group["task_values"] = {task: value / counts[task] for task, value in values.items()}


class PositiveGradientBuffer:
    def __init__(self, state):
        self.state = state

    def add(self, group, tasks, model):
        from bme_eating.training.candidate_supervision import retained_groups
        groups = self.state.get(group["event_key"], {})
        descriptor = group["descriptor"]
        dropped = []
        selected = retained_groups([*[item["original_descriptor"] for item in groups.values()], descriptor],
                                   diagnostics=dropped)
        retained = {item["id"]: item for item in selected}
        if descriptor["id"] not in retained:
            return {"added": None, "removed": [], "dropped": dropped}
        removed = sorted(set(groups) - retained.keys())
        for identity in removed:
            groups.pop(identity)
        specifications = tasks.get("_atoms")
        if specifications is None:
            specifications = [{"geometries": descriptor["geometries"], "tasks": tasks}]
        requests = [(index, task, loss) for index, atom in enumerate(specifications)
                    for task, loss in atom["tasks"].items()]
        gradients, fallback = atomic_gradients([loss for _, _, loss in requests], model)
        atoms = [{"geometries": atom["geometries"], "gradients": {}, "values": {}}
                 for atom in specifications]
        for (index, task, loss), gradient in zip(requests, gradients):
            atoms[index]["gradients"][task] = gradient
            atoms[index]["values"][task] = float(loss.detach())
        groups[descriptor["id"]] = {"original_descriptor": descriptor, "descriptor": descriptor, "atoms": atoms}
        self.state[group["event_key"]] = groups
        trimmed = []
        for identity, item in groups.items():
            previous = len(item["descriptor"]["geometries"])
            item["descriptor"] = retained[identity]
            materialize(item)
            if previous != len(item["descriptor"]["geometries"]):
                trimmed.append({"group": identity, "previous_versions": previous,
                                "retained_versions": len(item["descriptor"]["geometries"])})
        return {"added": descriptor["id"], "removed": removed, "retained": sorted(retained),
                "dropped": dropped, "trimmed": trimmed,
                "batched_vjp_fallback": fallback, "atomic_losses": len(requests)}

    def tensors(self):
        return (value for groups in self.state.values() for group in groups.values()
                for atom in group["atoms"] for gradients in atom["gradients"].values()
                for value in gradients.values())


class ContributionBank:
    def __init__(self, state):
        self.state = state
        for name in ("negative_task_gradients", "negative_task_counts", "negative_task_values",
                     "pending_task_gradients", "pending_task_sessions", "pending_task_values"):
            state.setdefault(name, {})
        state.setdefault("negative_family_count", 0)

    def negative(self, tasks, model):
        self.negative_many([tasks if isinstance(tasks, dict) else {"classification": tasks}], model)

    def negative_many(self, groups, model):
        tasks = {}
        for group in groups:
            for name, loss in group.items():
                if name.startswith("_"):
                    continue
                if not torch.isfinite(loss).all():
                    raise RuntimeError("Nonfinite negative task loss")
                tasks.setdefault(name, []).append(loss)
        for task, losses in tasks.items():
            total = torch.stack(losses).sum()
            add_gradients(self.state["negative_task_gradients"].setdefault(task, {}), cpu_gradients(total, model))
            counts = self.state["negative_task_counts"]
            counts[task] = counts.get(task, 0) + len(losses)
            values = self.state["negative_task_values"]
            values[task] = values.get(task, 0.0) + float(total.detach())
        self.state["negative_family_count"] = self.state["negative_task_counts"].get("classification", 0)

    def pending(self, name, gradients, value=0.0):
        if gradients:
            add_gradients(self.state["pending_task_gradients"].setdefault(name, {}), gradients)
            counts = self.state["pending_task_sessions"]
            counts[name] = counts.get(name, 0) + 1
            values = self.state["pending_task_values"]
            values[name] = values.get(name, 0.0) + value

    def finish_session(self, positive):
        legal_objects = {}
        for task in ("classification", "boundary", "quality_ranking"):
            events, event_values = [], []
            for groups in positive.values():
                legal = [group for group in groups.values() if task in group["task_gradients"]]
                if legal:
                    gradients = {}
                    for group in legal:
                        add_gradients(gradients, group["task_gradients"][task], len(legal))
                    events.append(gradients)
                    event_values.append(sum(group["task_values"][task] for group in legal) / len(legal))
            negative_count = self.state["negative_task_counts"].get(task, 0)
            if task == "classification":
                session = {}
                for event in events:
                    add_gradients(session, event, len(events))
                self.pending("positive", session, sum(event_values) / max(1, len(events)))
                if negative_count:
                    self.pending("negative", {name: value / negative_count for name, value in
                        self.state["negative_task_gradients"][task].items()},
                        self.state["negative_task_values"][task] / negative_count)
            else:
                count = len(events) + negative_count
                gradients = {}
                for event in events:
                    add_gradients(gradients, event, count)
                if negative_count:
                    add_gradients(gradients, self.state["negative_task_gradients"][task], count)
                self.pending(task, gradients, (sum(event_values) +
                    self.state["negative_task_values"].get(task, 0)) / max(1, count))
            legal_objects[task] = {"positive_events": len(events), "negative_families": negative_count}
        self.state["last_session_contributions"] = {"positive_events": len(positive),
            "negative_families": self.state["negative_family_count"],
            "groups": sum(map(len, positive.values())), "legal_objects": legal_objects}
        positive.clear()
        for name in ("negative_task_gradients", "negative_task_counts", "negative_task_values"):
            self.state[name].clear()
        self.state["negative_family_count"] = 0

    def commit(self, model):
        pending, counts = self.state["pending_task_gradients"], self.state["pending_task_sessions"]
        classes = sum(counts.get(name, 0) > 0 for name in ("positive", "negative"))
        parameters = dict(model.named_parameters())
        if classes:
            for name in ("candidate_head.event.weight", "candidate_head.event.bias",
                         "candidate_head.quality.weight", "candidate_head.quality.bias"):
                if name in parameters and not any(name in pending.get(task, {}) for task in ("positive", "negative")):
                    raise RuntimeError(f"Candidate classification gradient disconnected: {name}")
        if counts.get("boundary", 0):
            for name in parameters:
                if name.startswith("boundary_head.") and name not in pending.get("boundary", {}):
                    raise RuntimeError(f"Candidate boundary gradient disconnected: {name}")
        objective = {}
        for task, gradients in pending.items():
            divisor = counts[task] * (classes if task in {"positive", "negative"} else 1)
            objective[task] = self.state["pending_task_values"].get(task, 0) / divisor
            for name, value in gradients.items():
                gradient = value.to(parameters[name].device) / divisor
                if parameters[name].grad is None:
                    parameters[name].grad = gradient
                else:
                    parameters[name].grad.add_(gradient)
        self.state["last_update_contributions"] = dict(counts)
        self.state["last_update_candidate_objective"] = objective
        pending.clear()
        counts.clear()
        self.state["pending_task_values"].clear()

    def tensors(self):
        return (value for bank in (*self.state["negative_task_gradients"].values(),
            *self.state["pending_task_gradients"].values()) for value in bank.values())
