from __future__ import annotations

import numpy as np

from bme_eating.joint_evaluation import candidate_coverage, metrics_with_exposure


def stratified_diagnostics(truth, predictions, candidates, ignore, windows, split):
    def evaluate(subjects):
        selected_truth = truth[truth.subject_key.isin(subjects)]
        selected_windows = windows[windows.subject_key.isin(subjects)]
        pool = [item for item in candidates if item.subject_key in subjects]
        prediction = [item for item in predictions if item.subject_key in subjects]
        metrics, _ = metrics_with_exposure(selected_truth, prediction,
            ignore[ignore.subject_key.isin(subjects)], selected_windows)
        return {"support_events": len(selected_truth), "support_subjects": len(subjects),
                "metrics": metrics, "coverage": candidate_coverage(pool, selected_truth, selected_windows)}
    subjects = set(split["development_subjects"])
    missing = {subject for subject in subjects if split["subject_statistics"][subject]["gyro_all_missing"]}
    report = {"overall": evaluate(subjects), "gyro_all_missing": evaluate(missing),
              "gyro_available": evaluate(subjects - missing)}
    _, matches = metrics_with_exposure(truth, predictions, ignore, windows)
    match_keys = set(zip(matches.subject_key.astype(str), matches.session_id.astype(str), matches.event_id.astype(str))) if len(matches) else set()
    durations = (truth.end_ms - truth.start_ms) / 1000
    masks = {"same_hand": truth.hand_relation.eq("same"), "different_hand": truth.hand_relation.eq("different"),
             "unknown_hand": ~truth.hand_relation.isin(["same", "different"])}
    bins = np.searchsorted(split["duration_edges_seconds"], durations.to_numpy(), side="right")
    masks.update({f"duration_bin_{index}": bins == index for index in range(5)})
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        target = truth[mask]
        tp = sum((str(row.subject_key), str(row.session_id), str(row.event_id)) in match_keys
                 for row in target.itertuples(index=False))
        covered = candidate_coverage(candidates, target, windows)
        eligible_sessions = []
        for key, group in truth.groupby(["subject_key", "session_id"]):
            if mask[truth.index.get_indexer(group.index)].all():
                eligible_sessions.append(tuple(map(str, key)))
        keys = set(eligible_sessions)
        def filter_sessions(frame, keys=keys):
            selected = np.asarray([tuple(map(str, key)) in keys for key in
                frame[["subject_key", "session_id"]].itertuples(index=False, name=None)], dtype=bool)
            return frame.loc[selected]
        session_truth = filter_sessions(truth)
        session_ignore = filter_sessions(ignore)
        session_windows = filter_sessions(windows)
        metrics, _ = metrics_with_exposure(session_truth, [item for item in predictions if (item.subject_key, item.session_id) in keys], session_ignore, session_windows)
        report[name] = {"support_events": len(target), "true_positive": tp, "false_negative": len(target) - tp,
            "recall": tp / len(target) if len(target) else None, "coverage": covered,
            "homogeneous_event_session_metrics": metrics,
            "metric_scope": "Only sessions whose evaluable events all belong to this stratum; background-only sessions have no truth hand/duration label"}
    return report


def locked_subject_bootstrap(truth, predictions, ignore, windows, subjects, replicates=1000, seed=2026):
    rows = []
    for subject in sorted(subjects):
        metrics, matches = metrics_with_exposure(truth[truth.subject_key.eq(subject)],
            [item for item in predictions if item.subject_key == subject],
            ignore[ignore.subject_key.eq(subject)], windows[windows.subject_key.eq(subject)])
        rows.append([metrics["true_positive"], metrics["false_positive"], metrics["false_negative"],
            metrics["observed_hours"],
            float(matches.start_absolute_error_ms.sum()) / 1000 if len(matches) else 0,
            float(matches.end_absolute_error_ms.sum()) / 1000 if len(matches) else 0])
    rng = np.random.default_rng(seed)
    totals = np.asarray(rows)[rng.integers(0, len(rows), (replicates, len(rows)))].sum(axis=1)
    tp, fp, fn, hours, start, end = totals.T
    values = {"f1": np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=2 * tp + fp + fn > 0),
        "fp_per_hour": np.divide(fp, hours, out=np.full_like(fp, np.nan), where=hours > 0),
        "start_mae_seconds": np.divide(start, tp, out=np.full_like(tp, np.nan), where=tp > 0),
        "end_mae_seconds": np.divide(end, tp, out=np.full_like(tp, np.nan), where=tp > 0)}
    return {"method": "subject_cluster_bootstrap_locked_point", "seed": seed, "replicates": replicates,
        "support_subjects": len(rows), "independent_evidence": False,
        "selection_optimism_corrected": False,
        "intervals_95": {name: np.nanquantile(value, [.025, .975]).tolist()
                         if np.isfinite(value).any() else None for name, value in values.items()}}


def operating_point_stability(records, point):
    eligible = [record for record in records if record["promotion_eligible"]]
    near = [record for record in eligible if record["f1"] >= point["f1"] - .003]
    return {"registered_points": len(records), "qualified_points": len(eligible),
        "near_optimal_delta": .003, "near_optimal_points": len(near),
        "acceptance_range": [min(row["acceptance"] for row in near), max(row["acceptance"] for row in near)] if near else None,
        "settings": sorted({row["setting"] for row in near}),
        "budgets": sorted({row["budget"] for row in near}), "nms": sorted({row["nms"] for row in near}),
        "independent_validation": False, "used_to_change_selection": False}
