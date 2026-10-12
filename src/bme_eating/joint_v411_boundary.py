from __future__ import annotations

import copy

import numpy as np

from bme_eating.joint_capabilities import supported
from bme_eating.joint_evaluation import (
    NonnegativePlatt,
    calibration_ece,
    commit_candidates,
    event_frame,
)
from bme_eating.joint_stream import candidate_targets
from bme_eating.metrics import prediction_ignore_mask


def refined_key(point):
    if point.get("boundary_mode"):
        return f"{point['boundary_mode']}:{point['boundary_entropy']}"
    return str(point["boundary_entropy"])


def lock_boundary_modes(coarse_point, natural, coarse_predictions, outputs, truth, ignore, windows, config):
    from bme_eating.joint_protocol import event_score_formula
    from bme_eating.joint_v411_resources import SearchResourceMonitor
    from bme_eating.training.statsfusion_joint_trainer import boundary_pair_gate
    monitor = SearchResourceMonitor(config)
    monitor.check(force=True)
    locked = copy.deepcopy(coarse_point)
    point = {**copy.deepcopy(locked), "boundary_mode": "off", "boundary_enabled": False,
             "boundary_entropy": None}
    reports, search, best = [], [], None
    if supported(config, "consistent_inputs"):
        events = {(item.subject_key, item.session_id, candidate_targets(item, truth)[2])
                  for item in natural if candidate_targets(item, truth)[0]}
        minimum = config["boundary"]["minimum_positive_events"]
        if len(events) < minimum or locked["candidate_recall"] < config["boundary"]["minimum_candidate_recall"]:
            point.update({"coarse_operating_point": locked, "score_formula": event_score_formula(config)})
            reports.append({"passed": False, "search_skipped": True,
                "reason": "insufficient_independent_event_or_coverage_evidence",
                "maximum_available_events": len(events), "truth_events": len(truth), "required_events": minimum,
                "head_learning_enabled": True, "boundary_mode": "off"})
            return point, reports, search
    thresholds = np.arange(config["search"]["acceptance_min"], config["search"]["acceptance_max"] + 1e-8,
                           config["search"]["acceptance_step"])
    for mode in ("start_only", "both"):
        for entropy in config["boundary"]["entropy_thresholds"]:
            monitor.check(force=True)
            versions = {item.proposal_id: item for output in outputs for item in
                        output["refined"][f"{mode}:{entropy}"][locked["setting"]]}
            refined = [versions[item.proposal_id] for item in natural]
            labels = np.asarray([candidate_targets(item, truth)[0] for item in refined])
            ignored = prediction_ignore_mask(event_frame(refined), ignore)
            keep = ~ignored | (labels > 0)
            logits = np.asarray([item.event_logit for item in refined])
            calibrator = NonnegativePlatt.fit(logits[keep], labels[keep])
            ece = calibration_ece(calibrator.transform(logits[keep]), labels[keep], config["calibration"]["bins"])
            mode_best, mode_report = None, None
            for acceptance in thresholds:
                monitor.check()
                prediction, diagnostics = commit_candidates(copy.deepcopy(refined), calibrator,
                    float(acceptance), locked["nms"], return_diagnostics=True)
                report = boundary_pair_gate(coarse_predictions, prediction, truth, ignore, windows,
                    config, refined_candidates=diagnostics, mode=mode)
                metrics = report["refined"]
                eligible = (locked["promotion_eligible"] and report["passed"] and calibrator.qualified
                    and (not supported(config, "stabilized") or bool(calibrator.discriminative))
                    and ece <= config["calibration"]["maximum_ece"]
                    and metrics["fp_per_hour"] <= config["gates"]["maximum_fp_per_hour"])
                record = {"boundary_mode": mode, "entropy": entropy, "acceptance": float(acceptance),
                          "nms": locked["nms"], "promotion_eligible": bool(eligible),
                          "calibration_ece": ece, **metrics}
                search.append(record)
                key = (int(eligible), metrics["f1"], -metrics["fp_per_hour"])
                if mode_best is None or key > mode_best:
                    mode_best, mode_report = key, {**report, "entropy": entropy, "acceptance": float(acceptance),
                        "calibration": calibrator.to_json(), "calibration_ece": ece,
                        "locked_coarse_calibration": locked["calibration"], "passed": bool(eligible)}
                if eligible and (best is None or key > best):
                    best = key
                    point = {**locked, **metrics, "boundary_mode": mode, "boundary_enabled": True,
                        "boundary_entropy": entropy, "acceptance": float(acceptance),
                        "calibration": calibrator.to_json(), "calibration_ece": ece,
                        "promotion_eligible": True}
            reports.append(mode_report)
    point["coarse_operating_point"] = locked
    point["score_formula"] = event_score_formula(config)
    point["boundary_search_resources"] = monitor.check(force=True)
    return point, reports, search
