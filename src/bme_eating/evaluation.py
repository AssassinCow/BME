from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from bme_eating.candidates import (
    FrozenSourceCDF,
    ImmutableCommitter,
    JointCandidate,
    OnlineBudget,
    candidate_targets,
)
from bme_eating.metrics import evaluate_events, interval_iou_matrix, prediction_ignore_mask
from bme_eating.protocol import EVENT_PROBABILITY_FORMULAS, supported


@dataclass
class NonnegativePlatt:
    slope: float
    intercept: float
    qualified: bool
    optimization_succeeded: bool | None = None
    probability_range: float | None = None
    discriminative: bool | None = None

    def __post_init__(self):
        if not np.isfinite([self.slope, self.intercept]).all() or self.slope < 0:
            raise ValueError("Platt calibration requires finite values and a nonnegative slope")

    @classmethod
    def fit(cls, logits: np.ndarray, targets: np.ndarray) -> NonnegativePlatt:
        if logits.ndim != 1 or logits.shape != targets.shape or not np.isin(targets, [0, 1]).all():
            raise ValueError("Calibration requires aligned logits and binary event targets")
        if not np.isfinite(logits).all() or not np.isfinite(targets).all():
            raise ValueError("Calibration values must be finite")
        if not len(logits) or len(np.unique(targets)) < 2:
            return cls(0.0, 0.0, False)
        def objective(parameters):
            prediction = parameters[0] * logits + parameters[1]
            value = np.mean(np.logaddexp(0, prediction) - targets * prediction)
            residual = expit(prediction) - targets
            gradient = np.asarray([np.mean(residual * logits), np.mean(residual)])
            return value, gradient
        result = minimize(objective, [1.0, 0.0], jac=True, bounds=[(0, None), (None, None)])
        slope, intercept = float(result.x[0]), float(result.x[1])
        spread = float(np.ptp(expit(slope * logits + intercept)))
        return cls(slope, intercept, bool(result.success), bool(result.success), spread,
                   bool(slope > 1e-6 and spread > 1e-6))

    def transform(self, value):
        return expit(self.slope * np.asarray(value) + self.intercept)

    def to_json(self):
        return asdict(self)


def event_frame(candidates: list[JointCandidate]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=["subject_key", "session_id", "start_ms", "end_ms", "proposal_id"])
    return pd.DataFrame([candidate.to_dict() for candidate in candidates])


def candidate_coverage(candidates: list[JointCandidate], truth: pd.DataFrame,
                       windows: pd.DataFrame | None = None) -> dict[str, Any]:
    covered, identities = 0, set()
    grouped: dict[tuple[str, str], list[JointCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.subject_key, candidate.session_id), []).append(candidate)
    strata: dict[str, dict[str, int]] = {}
    window_groups = ({(str(subject), str(session)): group.sort_values("timestamp_ms")
                      for (subject, session), group in windows.groupby(["subject_key", "session_id"])}
                     if windows is not None and "gyro_valid" in windows else {})
    for row in truth.itertuples(index=False):
        pool = grouped.get((str(row.subject_key), str(row.session_id)), [])
        match = bool(pool) and bool((interval_iou_matrix(
            np.asarray([[row.start_ms, row.end_ms]]),
            np.asarray([[candidate.start_ms, candidate.end_ms] for candidate in pool])) > 0.25).any())
        covered += int(match)
        if match:
            identities.add((str(row.subject_key), str(row.session_id),
                            str(getattr(row, "source_event_key", getattr(row, "event_id", "")))))
        duration = row.end_ms - row.start_ms
        labels = ["duration_short" if duration <= 30_000 else "duration_medium" if duration <= 768_000 else "duration_long",
                  "hand_" + str(getattr(row, "hand_relation", "unknown"))]
        if windows is not None and "gyro_valid" in windows:
            group = window_groups.get((str(row.subject_key), str(row.session_id)), windows.iloc[:0])
            support = group[(group.timestamp_ms > row.start_ms) & (group.timestamp_ms <= row.end_ms)
                            & (group.valid > 0)]
            if support.empty:
                missingness = "unknown"
            elif (support.gyro_valid <= 0).all():
                missingness = "missing"
            elif (support.gyro_valid > 0).all():
                missingness = "present"
            else:
                missingness = "mixed"
            labels.append("gyro_" + missingness)
        for label in labels:
            entry = strata.setdefault(label, {"truth": 0, "covered": 0})
            entry["truth"] += 1
            entry["covered"] += int(match)
    sources = {}
    for source in sorted({candidate.source for candidate in candidates}):
        subset = [candidate for candidate in candidates if candidate.source == source]
        source_covered = 0
        for row in truth.itertuples(index=False):
            pool = [candidate for candidate in subset if candidate.subject_key == str(row.subject_key)
                    and candidate.session_id == str(row.session_id)]
            if pool:
                source_covered += int((interval_iou_matrix(np.asarray([[row.start_ms, row.end_ms]]),
                    np.asarray([[candidate.start_ms, candidate.end_ms] for candidate in pool])) > 0.25).any())
        sources[source] = {"covered": source_covered, "candidate_count": len(subset),
                           "family_count": len({candidate.proposal_family_id for candidate in subset})}
    return {"truth": len(truth), "covered": covered, "recall": covered / len(truth) if len(truth) else 0,
            "independent_covered_events": len(identities), "strata": strata, "sources": sources,
            "candidates": len(candidates), "families": len({candidate.proposal_family_id for candidate in candidates}),
            "gyro_missing_candidates": sum(candidate.gyro_missing for candidate in candidates)}


def apply_online_budget(candidates: list[JointCandidate], windows: pd.DataFrame,
                        mappings: dict[str, FrozenSourceCDF], budget: int,
                        quotas: dict[str, float]):
    selected, audit = [], []
    if windows.duplicated(["subject_key", "session_id", "timestamp_ms"]).any():
        raise RuntimeError("Duplicate budget exposure timeline")
    sessions = {(str(subject), str(session)) for subject, session in
                windows[["subject_key", "session_id"]].drop_duplicates().itertuples(index=False, name=None)}
    if any((candidate.subject_key, candidate.session_id) not in sessions for candidate in candidates):
        raise RuntimeError("Candidate has no observed session exposure")
    for (subject, session), group in windows.groupby(["subject_key", "session_id"], sort=True):
        pool = [candidate for candidate in candidates
                if candidate.subject_key == str(subject) and candidate.session_id == str(session)]
        decisions: dict[int, list[JointCandidate]] = {}
        for candidate in pool:
            decisions.setdefault(candidate.decision_time_ms, []).append(candidate)
        tracker = OnlineBudget(budget, quotas)
        group = group.sort_values("timestamp_ms")
        timeline = group.timestamp_ms.to_numpy()
        exposure = np.cumsum(group.valid.to_numpy() > 0) * 3
        observed = 0
        for decision in sorted(decisions):
            position = int(np.searchsorted(timeline, decision))
            if position >= len(timeline) or timeline[position] != decision:
                raise RuntimeError("Candidate decision has no corresponding observed input time")
            tracker.observe(float(exposure[position] - observed), 1)
            observed = exposure[position]
            current = decisions[decision]
            mapping_ids = {candidate.model_identity for candidate in current}
            if len(mapping_ids) != 1:
                raise RuntimeError("A session contains multiple model source mappings")
            selected.extend(tracker.select(current, mappings[next(iter(mapping_ids))]))
        if len(exposure):
            tracker.observe(float(exposure[-1] - observed), 1)
        audit.append({"subject_key": str(subject), "session_id": str(session), "records": tracker.audit})
    return selected, audit


def commit_candidates(candidates: list[JointCandidate], calibrator: NonnegativePlatt,
                      acceptance: float, nms: float, *, return_diagnostics=False):
    committer = ImmutableCommitter(acceptance, nms)
    decisions: dict[tuple[str, str, int], list[JointCandidate]] = {}
    for candidate in candidates:
        candidate.event_probability = float(calibrator.transform(candidate.event_logit))
        candidate.rank_score = (candidate.event_probability if candidate.score_formula
            in EVENT_PROBABILITY_FORMULAS
            else candidate.event_probability * candidate.predicted_quality)
        candidate.acceptance_score, candidate.quality_score = candidate.rank_score, candidate.predicted_quality
        if candidate.coarse_event_logit is not None:
            candidate.coarse_probability = float(calibrator.transform(candidate.coarse_event_logit))
        decisions.setdefault((candidate.subject_key, candidate.session_id,
                               candidate.decision_time_ms), []).append(candidate)
    for key in sorted(decisions):
        committer.submit(decisions[key])
    return (committer.committed, committer.diagnostics) if return_diagnostics else committer.committed


def metrics_with_exposure(truth, predictions, ignore, windows, *, observed_hours=None):
    metrics, matched = evaluate_events(truth, event_frame(predictions), ignore=ignore)
    hours = float((windows.valid > 0).sum() * 3 / 3600) if observed_hours is None else observed_hours
    metrics["observed_hours"] = hours
    metrics["fp_per_hour"] = metrics["false_positive"] / hours if hours else float("inf")
    for hand in ("same", "different", "unknown"):
        hand_truth = truth[truth.get("hand_relation", pd.Series("unknown", index=truth.index)).eq(hand)]
        hand_tp = int(matched.hand_relation.eq(hand).sum()) if len(matched) else 0
        metrics[hand + "_recall"] = hand_tp / len(hand_truth) if len(hand_truth) else None
    return metrics, matched


def calibration_ece(probabilities: np.ndarray, targets: np.ndarray, bins: int) -> float:
    if not len(targets):
        return float("inf")
    result = 0.0
    for index in range(bins):
        mask = (probabilities >= index / bins) & (probabilities < (index + 1) / bins)
        if index == bins - 1:
            mask |= probabilities == 1
        if mask.any():
            result += mask.mean() * abs(probabilities[mask].mean() - targets[mask].mean())
    return float(result)


def choose_operating_point(candidate_sets: dict[str, list[JointCandidate]], windows,
                           truth, ignore, mappings, config):
    if supported(config, "grouped") and not config.get("_reference_execution"):
        from bme_eating.operating_points import choose_point
        return choose_point(candidate_sets, windows, truth, ignore, mappings, config)
    search = config["search"]
    from bme_eating.resources import SearchResourceMonitor
    monitor = SearchResourceMonitor(config)
    monitor.check(force=True)
    records, best, best_key = [], None, None
    observed_hours = float((windows.valid > 0).sum() * 3 / 3600)
    thresholds = np.arange(search["acceptance_min"], search["acceptance_max"] + 1e-8,
                           search["acceptance_step"])
    for setting, candidates in candidate_sets.items():
        for budget in search["budgets"]:
            monitor.check(force=True)
            natural, budget_audit = apply_online_budget(candidates, windows, mappings, budget,
                                                         config["decoder"]["quotas"])
            ignored = prediction_ignore_mask(event_frame(natural), ignore)
            labels = np.asarray([candidate_targets(candidate, truth)[0] for candidate in natural])
            keep = ~ignored | (labels > 0)
            logits = np.asarray([candidate.event_logit for candidate in natural])
            calibrator = NonnegativePlatt.fit(logits[keep], labels[keep])
            ece = calibration_ece(calibrator.transform(logits[keep]), labels[keep],
                                  config["calibration"]["bins"])
            coverage = candidate_coverage(natural, truth, windows)
            qualified = (calibrator.qualified and (not supported(config, "stabilized") or
                         bool(calibrator.discriminative))
                         and ece <= config["calibration"]["maximum_ece"]
                         and coverage["recall"] >= 0.70)
            for nms, acceptance in product(search["nms_ious"], thresholds):
                monitor.check()
                predictions = commit_candidates(natural, calibrator, float(acceptance), nms)
                metrics, _ = metrics_with_exposure(truth, predictions, ignore, windows,
                                                   observed_hours=observed_hours)
                feasible = qualified and metrics["fp_per_hour"] <= config["gates"]["maximum_fp_per_hour"]
                record = {"setting": setting, "budget": budget, "nms": nms,
                          "acceptance": float(acceptance), "calibration": calibrator.to_json(),
                          "calibration_ece": ece, "candidate_recall": coverage["recall"],
                          "promotion_eligible": feasible, **metrics}
                records.append(record)
                bias = metrics.get("start_signed_error_seconds", float("inf"))
                key = (int(feasible), metrics["f1"], -metrics["fp_per_hour"], coverage["recall"],
                       -abs(bias) if np.isfinite(bias) else -float("inf"))
                if best_key is None or key > best_key:
                    best_key, best = key, {**record, "budget_audit": budget_audit}
    if best is None:
        raise RuntimeError("No diagnostic operating point could be evaluated")
    best["search_resources"] = monitor.check(force=True)
    return best, records


def paired_subject_bootstrap(truth, new, reference, ignore, windows, replicates: int):
    subjects = sorted(set(truth.subject_key.astype(str)) | set(ignore.subject_key.astype(str))
                      | set(event_frame(new).subject_key.astype(str))
                      | set(reference.subject_key.astype(str)) | set(windows.subject_key.astype(str)))
    new_counts, reference_counts = [], []
    frame = event_frame(new)
    for subject in subjects:
        selected_truth = truth[truth.subject_key.astype(str).eq(subject)]
        selected_ignore = ignore[ignore.subject_key.astype(str).eq(subject)]
        for output, prediction in ((new_counts, frame), (reference_counts, reference)):
            metrics, _ = evaluate_events(selected_truth,
                prediction[prediction.subject_key.astype(str).eq(subject)], ignore=selected_ignore)
            output.append([metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]])
    rng = np.random.default_rng(2026)
    new_array, reference_array = np.asarray(new_counts), np.asarray(reference_counts)
    wins = 0
    differences = []
    for _ in range(replicates):
        indices = rng.integers(0, len(subjects), len(subjects))
        def f1(values, indices=indices):
            tp, fp, fn = values[indices].sum(0)
            return 2 * tp / max(2 * tp + fp + fn, 1)
        difference = f1(new_array) - f1(reference_array)
        wins += difference > 0
        differences.append(difference)
    return {"advantage_probability": wins / replicates, "cohort": subjects,
            "difference_interval": np.quantile(differences, [0.025, 0.975]).tolist()}
