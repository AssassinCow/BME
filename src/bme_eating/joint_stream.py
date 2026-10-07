from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.metrics import interval_iou_matrix


def stable_id(*values: Any) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()[:24]


@dataclass
class JointCandidate:
    subject_key: str
    session_id: str
    start_ms: int
    end_ms: int
    decision_time_ms: int
    source: str
    scale_seconds: int
    raw_score: float
    coarse_start_ms: int
    coarse_end_ms: int
    proposal_family_id: str
    proposal_id: str
    ancestor_id: str
    parent_hash: str
    model_identity: str
    generation_parameters: dict[str, Any]
    max_input_time_ms: int
    missing_fraction: float = 0.0
    gyro_missing: bool = False
    event_logit: float = 0.0
    predicted_quality: float = 0.0
    normalized_source_score: float = 0.0
    event_probability: float = 0.0
    rank_score: float = 0.0
    refined: bool = False
    boundary_clipped: bool = False
    boundary_conflict: bool = False
    wear_hand: str = "unknown"
    hand_relation: str = "unknown"
    boundary_valid_start_bins: int = 0
    boundary_valid_end_bins: int = 0
    boundary_start_entropy: float | None = None
    boundary_end_entropy: float | None = None
    boundary_fallback_reason: str = ""
    unrefined_start_ms: int | None = None
    unrefined_end_ms: int | None = None
    coarse_event_logit: float | None = None
    coarse_quality: float | None = None
    coarse_probability: float | None = None

    def validate(self) -> None:
        if (any(not isinstance(value, (int, np.integer)) for value in
                (self.start_ms, self.end_ms, self.decision_time_ms, self.max_input_time_ms,
                 self.coarse_start_ms, self.coarse_end_ms))
                or not all((self.subject_key, self.session_id, self.model_identity, self.proposal_id,
                            self.proposal_family_id, self.ancestor_id, self.parent_hash))):
            raise RuntimeError("Candidate coordinates or lineage identity are invalid")
        if self.refined and (self.unrefined_start_ms is None or self.unrefined_end_ms is None):
            raise RuntimeError("Refined candidate lacks original Boundary coordinates")
        values = (self.raw_score, self.event_logit, self.predicted_quality,
                  self.event_probability, self.rank_score, self.missing_fraction)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Nonfinite candidate score or quality")
        if any(not 0 <= value <= 1 for value in (self.predicted_quality, self.event_probability,
                                                self.rank_score, self.missing_fraction)):
            raise RuntimeError("Candidate probability or missingness is outside [0, 1]")
        if any(value is not None and not math.isfinite(value) for value in
               (self.coarse_event_logit, self.coarse_quality, self.coarse_probability,
                self.boundary_start_entropy, self.boundary_end_entropy)):
            raise RuntimeError("Nonfinite candidate Boundary or coarse score")
        if any(value is not None and not 0 <= value <= 1 for value in (self.coarse_quality, self.coarse_probability)):
            raise RuntimeError("Candidate coarse probability/quality is outside [0, 1]")
        if not self.start_ms < self.end_ms <= self.decision_time_ms:
            raise RuntimeError("Invalid event coordinates")
        if self.max_input_time_ms > self.decision_time_ms:
            raise RuntimeError("Future input used by event decision")
        if self.decision_time_ms - self.end_ms > 60_000:
            raise RuntimeError("Event latency exceeds 60 seconds")

    def metadata(self, device: torch.device) -> torch.Tensor:
        duration = (self.end_ms - self.start_ms) / 1000
        return torch.tensor([
            math.log1p(duration) / 10, float(self.source == "proposal"),
            float(self.source == "state"), float(self.source == "jitter"),
            float(self.source == "backtrack"), float(self.source == "start_expansion"),
            self.scale_seconds / 60, (self.start_ms - self.coarse_start_ms) / 900_000,
            (self.end_ms - self.coarse_end_ms) / 30_000, self.missing_fraction,
            float(self.gyro_missing), self.raw_score,
        ], dtype=torch.float32, device=device)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["source_flags"] = [self.source]
        parent_source = self.generation_parameters.get("parent_source")
        if parent_source and parent_source != self.source:
            payload["source_flags"].append(parent_source)
        return payload


class FrozenSourceCDF:
    def __init__(self, distributions: dict[str, list[float]], training_subjects: list[str]) -> None:
        if not training_subjects or len(set(training_subjects)) != len(training_subjects):
            raise ValueError("Source CDF requires unique nonempty training subjects")
        if not distributions or any(not np.isfinite(values).all() or not len(values)
                                    for values in distributions.values()):
            raise ValueError("Source CDF requires nonempty finite distributions")
        self.distributions = {source: np.sort(values).tolist()
                              for source, values in distributions.items()}
        self.training_subjects = sorted(training_subjects)

    @classmethod
    def fit(cls, candidates: list[JointCandidate], subjects: set[str]) -> FrozenSourceCDF:
        distributions: dict[str, list[float]] = {}
        if not candidates:
            raise RuntimeError("No training candidates for source score normalization")
        if any(candidate.subject_key not in subjects for candidate in candidates):
            raise RuntimeError("Source normalization subject leakage")
        representatives: dict[tuple[str, str, int], JointCandidate] = {}
        for candidate in candidates:
            key = (candidate.proposal_family_id, candidate.source, candidate.scale_seconds)
            if key not in representatives or representatives[key].raw_score < candidate.raw_score:
                representatives[key] = candidate
        for candidate in representatives.values():
            source = f"{candidate.source}:{candidate.scale_seconds}"
            distributions.setdefault(source, []).append(candidate.raw_score)
        return cls(distributions, sorted(subjects))

    def transform(self, candidate: JointCandidate) -> float:
        values = self.distributions.get(f"{candidate.source}:{candidate.scale_seconds}",
                                         self.distributions.get(candidate.source, []))
        if not values:
            return 0.0
        return float(np.searchsorted(values, candidate.raw_score, side="right") / len(values))

    def to_json(self) -> dict[str, Any]:
        return {"distributions": self.distributions, "training_subjects": self.training_subjects}


class OnlineBudget:
    def __init__(self, budget: int, quotas: dict[str, float]) -> None:
        if (not math.isfinite(budget) or int(budget) != budget or budget <= 0
                or set(quotas) != {"proposal", "short", "expansion"}
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in quotas.values())
                or sum(quotas.values()) > 1):
            raise ValueError("Invalid candidate budget or source quotas")
        self.budget = int(budget)
        self.quotas = quotas
        self.valid_seconds = 0.0
        self.spent = 0
        self.counts = {name: 0 for name in quotas}
        self.seen_families: set[str] = set()
        self.audit: list[dict[str, Any]] = []
        self.seen_proposals: set[str] = set()

    def observe(self, seconds: float, valid: float) -> None:
        if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(valid) or not 0 <= valid <= 1:
            raise ValueError("Invalid budget exposure")
        self.valid_seconds += seconds * float(valid > 0)

    @property
    def allowance(self) -> int:
        return math.ceil(self.valid_seconds / 3600 * self.budget)

    def categories(self, candidate: JointCandidate) -> set[str]:
        result = set()
        if candidate.source == "proposal":
            result.add("proposal")
        if candidate.end_ms - candidate.start_ms <= 30_000:
            result.add("short")
        if candidate.source in {"backtrack", "start_expansion"}:
            result.add("expansion")
        return result

    def select(self, candidates: list[JointCandidate], mapping: FrozenSourceCDF) -> list[JointCandidate]:
        identities = {candidate.proposal_id for candidate in candidates}
        if len(identities) != len(candidates) or identities & self.seen_proposals:
            raise RuntimeError("Duplicate proposal identities cannot consume candidate budget")
        for candidate in candidates:
            candidate.validate()
            candidate.normalized_source_score = mapping.transform(candidate)
        self.seen_proposals.update(identities)
        available = max(0, self.allowance - self.spent)
        selected: list[JointCandidate] = []
        remaining = list(candidates)
        families: set[str] = set()
        while available and remaining:
            diverse = [candidate for candidate in remaining
                       if candidate.proposal_family_id not in families | self.seen_families]
            pool = diverse or remaining
            deficits = {name: max(0, math.ceil(self.allowance * fraction) - self.counts[name])
                        for name, fraction in self.quotas.items()}
            def priority(candidate: JointCandidate, deficits=deficits) -> tuple[float, float, str]:
                debt = max((deficits[name] for name in self.categories(candidate)), default=0)
                return debt, candidate.normalized_source_score, candidate.proposal_id
            choice = max(pool, key=priority)
            remaining.remove(choice)
            selected.append(choice)
            families.add(choice.proposal_family_id)
            categories = self.categories(choice)
            for category in categories:
                self.counts[category] += 1
            self.spent += 1
            available -= 1
        self.seen_families.update(families)
        self.audit.append({"decision_time_ms": candidates[0].decision_time_ms if candidates else None,
                           "valid_seconds": self.valid_seconds, "allowance": self.allowance,
                           "spent": self.spent, "rejected": len(remaining),
                           "reason": "cumulative_budget_exhausted" if remaining else "",
                           "quota_deficits": {name: max(0, math.ceil(self.allowance * fraction)
                                                         - self.counts[name])
                                              for name, fraction in self.quotas.items()},
                           "independent_families": len(self.seen_families)})
        return selected


class StreamingProposals:
    def __init__(self, subject: str, session: str, identity: str, config: dict[str, Any]) -> None:
        self.subject, self.session, self.identity = subject, session, identity
        self.config = config
        self.active: dict[str, dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self.ema: dict[str, float] = {}
        self.last_time: int | None = None
        self.origin: int | None = None
        self.families: list[tuple[int, int, str]] = []

    def observe(self, timestamp: int, probabilities: dict[str, float], valid: float,
                offset_probability: float = 0.0) -> list[JointCandidate]:
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in
               (*probabilities.values(), valid, offset_probability)):
            raise RuntimeError("Nonfinite or invalid streaming probability/mask")
        if self.last_time is not None and timestamp <= self.last_time:
            raise RuntimeError("Streaming input timestamps must increase")
        self.origin = timestamp - 3000 if self.origin is None else self.origin
        elapsed = 3 if self.last_time is None else (timestamp - self.last_time) / 1000
        self.last_time = timestamp
        high = float(self.config["high_threshold"])
        low = high / 2
        decay = 0.5 ** (elapsed / float(self.config["ema_half_life_seconds"]))
        gap = int(self.config["gap_merge_seconds"] * 1000)
        for source, probability in probabilities.items():
            previous = self.ema.get(source, probability)
            value = decay * previous + (1 - decay) * probability
            self.ema[source] = value
            if valid <= 0:
                value = 0
            if source in self.active:
                active = self.active[source]
                if value >= low and not (offset_probability >= 0.5 and value < high):
                    active["end"] = timestamp
                    active["score"] = max(active["score"], value)
                else:
                    self.pending.append(self.active.pop(source))
            elif value >= high:
                merges = [item for item in self.pending if item["source"] == source
                          and timestamp - 3000 - item["end"] <= gap]
                if merges:
                    active = merges[-1]
                    self.pending.remove(active)
                    active["end"] = timestamp
                    active["score"] = max(active["score"], value)
                else:
                    active = {"source": source, "start": timestamp - 3000,
                              "end": timestamp, "score": value}
                self.active[source] = active
        return self._ready(timestamp)

    def _ready(self, decision: int, eof: bool = False) -> list[JointCandidate]:
        ready = [item for item in self.pending
                 if decision - item["end"] >= 30_000 or eof]
        self.pending = [item for item in self.pending if item not in ready]
        ready.sort(key=lambda item: (item["start"], item["end"], item["source"]))
        candidates: list[JointCandidate] = []
        family_intervals = self.families
        for item in ready:
            start, end, source = item["start"], item["end"], item["source"]
            family = next((name for left, right, name in family_intervals
                           if min(right, end) > max(left, start)), None)
            if family is None:
                family = stable_id(self.subject, self.session, start, end, self.identity)
                family_intervals.append((start, end, family))
            parent = stable_id(family, source, start, end)
            scale = 3 if source == "state" else int(source.split("_")[-1])
            parent_kind = "state" if source == "state" else "proposal"
            parent_proposal_id = stable_id(parent, start, end, parent_kind)
            parent_hash = hashlib.sha256("|".join(map(str, (self.subject, self.session, self.identity,
                                                            source, start, end))).encode()).hexdigest()
            variants = [(start, end, "state" if source == "state" else "proposal", {})]
            jitter = self.config["jitter_seconds"]
            shifts = [(shift, 0) for shift in jitter if shift] + [
                (0, shift) for shift in jitter if shift] + [
                (shift, shift) for shift in jitter if shift]
            shifts += [(-30, 30), (30, -30), (-15, 15), (15, -15)]
            variants.extend((start + left * 1000, end + right * 1000, "jitter",
                             {"start_jitter": left, "end_jitter": right}) for left, right in shifts)
            variants.extend((start - int(self.config[name + "_seconds"] * 1000), end, name,
                             {"seconds": self.config[name + "_seconds"]})
                            for name in ("backtrack", "start_expansion"))
            seen = set()
            for left, right, kind, parameters in variants:
                left = max(int(self.origin), left)
                right = min(decision, right)
                if left >= right or decision - right > 60_000 or (left, right, kind) in seen:
                    continue
                seen.add((left, right, kind))
                identifier = stable_id(parent, left, right, kind)
                candidate = JointCandidate(self.subject, self.session, left, right, decision,
                                           kind, scale, item["score"], start, end, family,
                                           identifier, parent_proposal_id, parent_hash,
                                           self.identity, {"parent_source": parent_kind,
                                                           "parent_scale": scale, **parameters}, decision)
                candidate.validate()
                candidates.append(candidate)
        return candidates

    def finish(self) -> list[JointCandidate]:
        if self.last_time is None:
            return []
        self.pending.extend(self.active.values())
        self.active = {}
        return self._ready(self.last_time, eof=True)


class ImmutableCommitter:
    def __init__(self, acceptance: float, nms_iou: float) -> None:
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (acceptance, nms_iou)):
            raise ValueError("Invalid acceptance or NMS threshold")
        self.acceptance, self.nms_iou = acceptance, nms_iou
        self.committed: list[JointCandidate] = []
        self.diagnostics: list[JointCandidate] = []

    def submit(self, candidates: list[JointCandidate]) -> list[JointCandidate]:
        from dataclasses import replace
        emitted = []
        for candidate in sorted(candidates, key=lambda item: (-item.rank_score, item.proposal_id)):
            candidate.validate()
            previous = [item for item in self.committed
                        if item.subject_key == candidate.subject_key and item.session_id == candidate.session_id]
            if candidate.refined and any(
                    min(candidate.end_ms, item.end_ms) > max(candidate.start_ms, item.start_ms)
                    and min(candidate.unrefined_end_ms, item.end_ms) <= max(candidate.unrefined_start_ms, item.start_ms)
                    for item in previous):
                if any(value is None for value in (candidate.coarse_event_logit, candidate.coarse_quality,
                                                   candidate.coarse_probability)):
                    raise RuntimeError("Boundary conflict fallback lacks the coarse score")
                candidate = replace(candidate, start_ms=candidate.unrefined_start_ms,
                    end_ms=candidate.unrefined_end_ms, event_logit=candidate.coarse_event_logit,
                    predicted_quality=candidate.coarse_quality, event_probability=candidate.coarse_probability,
                    rank_score=candidate.coarse_probability * candidate.coarse_quality, refined=False,
                    boundary_conflict=True, boundary_fallback_reason="neighbor_conflict")
                candidate.validate()
            self.diagnostics.append(replace(candidate, generation_parameters=dict(candidate.generation_parameters)))
            if candidate.rank_score < self.acceptance:
                continue
            if previous:
                overlaps = interval_iou_matrix(
                    np.asarray([[item.start_ms, item.end_ms] for item in previous]),
                    np.asarray([[candidate.start_ms, candidate.end_ms]]))
                if bool((overlaps > self.nms_iou).any()):
                    continue
            frozen = replace(candidate, generation_parameters=dict(candidate.generation_parameters))
            self.committed.append(frozen)
            emitted.append(frozen)
        return emitted


class TimelineCache:
    def __init__(self, path: Path, binding: dict[str, Any], width: int = 138,
                 committed_rows: int = 0) -> None:
        import json
        self.path, self.binding, self.width = path, binding, width
        path.mkdir(parents=True, exist_ok=True)
        identity = path / "binding.json"
        if identity.exists() and json.loads(identity.read_text(encoding="utf-8")) != binding:
            raise RuntimeError("Cache identity differs in model/epoch/subject/session")
        write_json_atomic(identity, binding)
        self.rows = committed_rows
        for name, size in (("times.bin", 8), ("features.bin", width * 4)):
            target = path / name
            if not target.exists():
                target.touch()
            if target.stat().st_size < committed_rows * size:
                raise RuntimeError("Committed cache is truncated")
            if target.stat().st_size != committed_rows * size:
                with target.open("r+b") as handle:
                    handle.truncate(committed_rows * size)

    def append(self, times: np.ndarray, values: np.ndarray) -> None:
        import os
        if values.shape != (len(times), self.width) or not np.isfinite(values).all():
            raise RuntimeError("Cache feature shape or finite gate failed")
        for name, payload in (("times.bin", np.asarray(times, dtype=np.int64)),
                              ("features.bin", np.asarray(values, dtype=np.float32))):
            with (self.path / name).open("ab") as handle:
                handle.write(payload.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
        self.rows += len(times)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.rows:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, self.width)
        times = np.memmap(self.path / "times.bin", mode="c", dtype=np.int64, shape=(self.rows,))
        values = np.memmap(self.path / "features.bin", mode="c", dtype=np.float32,
                           shape=(self.rows, self.width))
        return torch.from_numpy(times), torch.from_numpy(values)


def candidate_targets(candidate: JointCandidate, truth: Any) -> tuple[float, float, str, int, int]:
    selected = truth[(truth.subject_key.astype(str) == candidate.subject_key)
                     & (truth.session_id.astype(str) == candidate.session_id)]
    if selected.empty:
        return 0.0, 0.0, "", 0, 0
    matrix = interval_iou_matrix(selected[["start_ms", "end_ms"]].to_numpy(),
                                np.asarray([[candidate.start_ms, candidate.end_ms]]))[:, 0]
    index = int(matrix.argmax())
    row = selected.iloc[index]
    quality = float(matrix[index])
    identity = str(row.get("source_event_key", row.get("event_id", index)))
    return float(quality > 0.25), quality, identity, int(row.start_ms), int(row.end_ms)


def sample_candidates(candidates: list[JointCandidate], truth: Any, maximum: int,
                      composition: dict[str, float], rng: np.random.Generator,
                      *, truth_jitter_fraction: float = 0.0):
    desired = {name: maximum * value for name, value in composition.items()}
    counts = {name: math.floor(value) for name, value in desired.items()}
    for name in sorted(counts, key=lambda item: (-(desired[item] - counts[item]), item))[
            :maximum - sum(counts.values())]:
        counts[name] += 1
    groups = {name: [] for name in composition}
    for candidate in candidates:
        positive, quality, *_ = candidate_targets(candidate, truth)
        category = "positive" if positive else "near_miss" if quality > 0 else (
            "random_background" if candidate.generation_parameters.get("sampling") == "random_background"
            else "hard_false_positive" if candidate.raw_score >= 0.10 else "random_background")
        groups[category].append(candidate)
    chosen, families, event_ids = [], set(), set()
    deficits = {}
    for category, quota in counts.items():
        pool = list(groups[category])
        rng.shuffle(pool)
        strata: dict[tuple[Any, ...], list[JointCandidate]] = {}
        for candidate in pool:
            _, quality, *_ = candidate_targets(candidate, truth)
            iou_bin = "zero" if quality == 0 else "near" if quality <= .25 else "positive"
            duration = (candidate.end_ms - candidate.start_ms) / 1000
            duration_bin = "short" if duration <= 30 else "medium" if duration <= 768 else "long"
            source = "expansion" if candidate.source in {"backtrack", "start_expansion"} else candidate.source
            key = (candidate.subject_key, candidate.wear_hand, candidate.hand_relation,
                   iou_bin, candidate.gyro_missing, duration_bin, source, category)
            strata.setdefault(key, []).append(candidate)
        pool = []
        while any(strata.values()):
            for key in sorted(strata):
                if strata[key]:
                    pool.append(strata[key].pop(0))
        if category == "positive" and truth_jitter_fraction:
            jitter = [item for item in pool if item.source == "truth_jitter"]
            predicted = [item for item in pool if item.source != "truth_jitter"]
            desired_jitter = min(len(jitter), math.ceil(quota * truth_jitter_fraction))
            pool = jitter[:desired_jitter] + predicted + jitter[desired_jitter:]
        count = 0
        for candidate in pool:
            _, _, event_id, *_ = candidate_targets(candidate, truth)
            if candidate.proposal_family_id in families or (category == "positive" and event_id in event_ids):
                continue
            if count >= quota:
                break
            chosen.append(candidate)
            families.add(candidate.proposal_family_id)
            if category == "positive":
                event_ids.add(event_id)
            count += 1
        deficits[category] = quota - count
    return chosen, {"desired": counts, "deficits": deficits, "candidates": len(chosen),
                    "truth_jitter_candidates": sum(item.source == "truth_jitter" for item in chosen),
                    "hand_relation_counts": {relation: sum(item.hand_relation == relation for item in chosen)
                                             for relation in ("same", "different", "unknown")},
                    "independent_families": len(families), "independent_events": len(event_ids)}
