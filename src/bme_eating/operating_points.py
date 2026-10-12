from __future__ import annotations

import copy
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from bme_eating.candidates import candidate_targets
from bme_eating.evaluation import (
    NonnegativePlatt,
    apply_online_budget,
    calibration_ece,
    candidate_coverage,
    event_frame,
)
from bme_eating.metrics import interval_iou_matrix, prediction_ignore_mask
from bme_eating.progress import progress
from bme_eating.resources import SearchResourceMonitor


class CachedSearch:
    def __init__(self, candidates, truth, ignore, windows, calibrator, resource_monitor=None):
        self.candidates, self.truth = candidates, truth
        self.resource_monitor = resource_monitor
        self.hours = float((windows.valid > 0).sum() * 3 / 3600)
        probabilities = calibrator.transform(np.asarray([item.event_logit for item in candidates]))
        self.groups = []
        sessions = sorted({(item.subject_key, item.session_id) for item in candidates}
                          | set(truth[["subject_key", "session_id"]].astype(str).itertuples(index=False, name=None)))
        for subject, session in sessions:
            indices = [index for index, item in enumerate(candidates)
                       if (item.subject_key, item.session_id) == (subject, session)]
            pool = [candidates[index] for index in indices]
            target = truth[truth.subject_key.astype(str).eq(subject) & truth.session_id.astype(str).eq(session)]
            target_coordinates = target[["start_ms", "end_ms"]].to_numpy()
            coordinates = np.asarray([[item.start_ms, item.end_ms] for item in pool]).reshape(-1, 2)
            if resource_monitor is not None:
                resource_monitor.check(allocation_bytes=8 * 6 * (len(pool) ** 2 + len(target) * len(pool)))
            iou = interval_iou_matrix(target_coordinates, coordinates)
            overlap = interval_iou_matrix(coordinates, coordinates)
            ignored = prediction_ignore_mask(event_frame(pool), ignore)
            decisions = {}
            for index, item in enumerate(pool):
                decisions.setdefault(item.decision_time_ms, {}).setdefault(item.ancestor_id, []).append(index)
            self.groups.append((pool, target, target_coordinates, coordinates, iou, overlap, ignored,
                                probabilities[indices], [decisions[key] for key in sorted(decisions)]))
            if resource_monitor is not None:
                resource_monitor.check(force=True)

    def evaluate(self, acceptance, nms):
        tp, fp, ignored_count = 0, 0, 0
        start_errors, end_errors, hands = [], [], {name: 0 for name in ("same", "different", "unknown")}
        for pool, target, target_coordinates, coordinates, iou, overlap, ignored, probabilities, decisions in self.groups:
            if self.resource_monitor is not None:
                self.resource_monitor.check()
            committed = []
            for parents in decisions:
                winners = []
                for versions in parents.values():
                    eligible = [index for index in versions if probabilities[index] >= acceptance]
                    if eligible:
                        def priority(index, pool=pool, probabilities=probabilities):
                            item = pool[index]
                            entropy = [value for value in (item.boundary_start_entropy,
                                       item.boundary_end_entropy) if value is not None]
                            return (-item.predicted_quality, max(entropy, default=1),
                                    -probabilities[index], item.proposal_id)
                        winners.append(min(eligible, key=priority))
                winners.sort(key=lambda index: (-probabilities[index], -pool[index].predicted_quality,
                                                pool[index].proposal_id))
                for index in winners:
                    if not committed or not (overlap[committed, index] > nms).any():
                        committed.append(index)
            matrix = iou[:, committed]
            count_truth, count_prediction = matrix.shape
            matches = []
            if matrix.size:
                bonus = float(min(count_truth, count_prediction) + 1)
                scores = np.zeros((count_truth + count_prediction,) * 2)
                scores[:count_truth, :count_prediction] = np.where(matrix > .25, bonus + matrix, -bonus)
                rows, columns = linear_sum_assignment(-scores)
                matches = [(row, committed[column]) for row, column in zip(rows, columns)
                           if row < count_truth and column < count_prediction and matrix[row, column] > .25]
            matched = {column for _, column in matches}
            ignored_predictions = sum(bool(ignored[index]) and index not in matched for index in committed)
            tp += len(matches)
            fp += len(committed) - ignored_predictions - len(matches)
            ignored_count += ignored_predictions
            for row, column in matches:
                start_errors.append(coordinates[column, 0] - target_coordinates[row, 0])
                end_errors.append(coordinates[column, 1] - target_coordinates[row, 1])
                hand = str(target.iloc[row].get("hand_relation", "unknown"))
                hands[hand if hand in hands else "unknown"] += 1
        fn = len(self.truth) - tp
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        metrics = {"true_positive": float(tp), "false_positive": float(fp), "false_negative": float(fn),
                   "ignored_predictions": float(ignored_count), "precision": precision,
                   "sensitivity": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                   "observed_hours": self.hours, "fp_per_hour": fp / self.hours if self.hours else float("inf")}
        for name, errors in (("start", start_errors), ("end", end_errors)):
            metrics[name + "_mae_seconds"] = float(np.mean(np.abs(errors)) / 1000) if errors else float("nan")
            metrics[name + "_signed_error_seconds"] = float(np.mean(errors) / 1000) if errors else float("nan")
        for hand, matched_count in hands.items():
            denominator = int(self.truth.get("hand_relation", pd.Series("unknown", index=self.truth.index)).eq(hand).sum())
            metrics[hand + "_recall"] = matched_count / denominator if denominator else None
        return metrics


def search_group(payload, root_pid=None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    natural, truth, ignore, windows, registered, config, calibrator, coverage, ece = payload
    monitor = SearchResourceMonitor(config, root_pid)
    monitor.check(force=True)
    cache = CachedSearch(natural, truth, ignore, windows, calibrator, monitor)
    thresholds = np.arange(config["search"]["acceptance_min"], config["search"]["acceptance_max"] + 1e-8,
                           config["search"]["acceptance_step"])
    from bme_eating.protocol import supported
    discrimination = (not supported(config, "stabilized") or
        bool(calibrator.discriminative and calibrator.slope > config["calibration"]["minimum_slope"]
             and calibrator.probability_range > config["calibration"]["minimum_probability_range"]))
    qualified = calibrator.qualified and discrimination and ece <= config["calibration"]["maximum_ece"] and coverage["recall"] >= .70
    records = []
    for nms, acceptance in product(config["search"]["nms_ious"], thresholds):
        monitor.check()
        metrics = cache.evaluate(float(acceptance), nms)
        records.append({**registered, "nms": nms, "acceptance": float(acceptance),
            "calibration": calibrator.to_json(), "calibration_ece": ece,
            "candidate_recall": coverage["recall"], "promotion_eligible": qualified and
            metrics["fp_per_hour"] <= config["gates"]["maximum_fp_per_hour"], **metrics})
    monitor.check(force=True)
    return records


def choose_point(candidate_sets, windows, truth, ignore, mappings, config):
    started = time.monotonic()
    monitor = SearchResourceMonitor(config)
    monitor.check(force=True)
    payloads, audits, diagnostic_groups = [], {}, {}
    preparation = progress(candidate_sets.items(), total=len(candidate_sets),
                           description="Selector calibration/budget", unit="decoder")
    for setting, candidates in preparation:
        for budget in config["search"]["budgets"]:
            monitor.check(force=True)
            natural, audit = apply_online_budget(candidates, windows, mappings, budget, {})
            if any(item.source == "truth_jitter" or item.generation_parameters.get("training_only") for item in natural):
                raise RuntimeError("Natural calibration contains training-only candidates")
            labels = np.asarray([candidate_targets(item, truth)[0] for item in natural])
            ignored = prediction_ignore_mask(event_frame(natural), ignore)
            keep = ~ignored | (labels > 0)
            logits = np.asarray([item.event_logit for item in natural])
            calibrator = NonnegativePlatt.fit(logits[keep], labels[keep])
            ece = calibration_ece(calibrator.transform(logits[keep]), labels[keep], config["calibration"]["bins"])
            coverage = candidate_coverage(natural, truth, windows)
            from bme_eating.diagnostics import (
                classifier_diagnostics,
                source_coverage,
            )
            diagnostic_groups[f"{setting}:budget_{budget}"] = {
                **classifier_diagnostics(logits[keep], labels[keep]),
                "coverage": coverage, "source_scale_coverage": source_coverage(natural, truth, windows),
                "calibration": calibrator.to_json(),
                "funnel": {"generated": len(candidates), "budget": len(natural), "natural_not_ignored": int(keep.sum())}}
            payloads.append((copy.deepcopy(natural), truth, ignore, windows[["valid"]],
                             {"setting": setting, "budget": budget}, config, calibrator, coverage, ece))
            audits[(setting, budget)] = audit
            monitor.check(force=True)
    workers = min(config["resources"]["search_processes"], len(payloads))
    if workers == 1:
        groups = map(search_group, payloads)
        records = [record for group in progress(groups, total=len(payloads),
            description="Selector operating-point search", unit="group") for record in group]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(search_group, payload, os.getpid()): index
                       for index, payload in enumerate(payloads)}
            completed = {}
            pending = set(futures)
            search_progress = progress(None, total=len(payloads),
                                      description="Selector operating-point search", unit="group")
            try:
                while pending:
                    monitor.check(force=True)
                    done, pending = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                    for future in done:
                        completed[futures[future]] = future.result()
                        search_progress.update(1)
                records = [record for index in range(len(payloads)) for record in completed[index]]
            finally:
                for future in pending:
                    future.cancel()
                search_progress.close()
    monitor.check(force=True)
    best, best_key = None, None
    for record in records:
        bias = record["start_signed_error_seconds"]
        key = (int(record["promotion_eligible"]), record["f1"], -record["fp_per_hour"],
               record["candidate_recall"], -abs(bias) if np.isfinite(bias) else -float("inf"))
        if best_key is None or key > best_key:
            best, best_key = record, key
    if best is None:
        raise RuntimeError("No registered operating points")
    diagnostic = {}
    if diagnostic_groups:
        diagnostic = {"natural_classifier_diagnostics": diagnostic_groups,
            "fixed_decoder_diagnostic_curve": [
                {name: record[name] for name in ("acceptance", "f1", "fp_per_hour", "true_positive",
                                                "false_positive", "false_negative", "candidate_recall")}
                for record in records if record["setting"] == "high_0.1_low_0.9_gap_15"
                and record["budget"] == 20 and record["nms"] == .3],
            "fixed_curve_used_for_selection": False}
    return {**best, "budget_audit": audits[(best["setting"], best["budget"])],
            **diagnostic,
            "search_seconds": time.monotonic() - started,
            "search_resources": {**monitor.usage, "search_processes": workers},
            "registered_operating_points": len(records)}, records
