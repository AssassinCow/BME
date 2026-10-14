from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bme_eating.evaluation import candidate_coverage, metrics_with_exposure


def endpoint_diagnostics(windows, truth, ignore, candidates):
    report = {}
    hours = float((windows.valid > 0).sum() * 3 / 3600)
    for name, field, threshold, support in (("onset", "start_ms", .30, 30_000),
                                            ("offset", "end_ms", .50, 60_000)):
        hits, eligible, background, peaks, maximum = 0, 0, 0, 0, 0.0
        usable_hits, usable_peaks, usable_background = 0, 0, 0
        for key, timeline in windows.groupby(["subject_key", "session_id"], sort=False):
            timeline = timeline.sort_values("timestamp_ms")
            times = timeline.timestamp_ms.to_numpy(np.int64)
            scores = timeline[name + "_probability"].to_numpy()
            valid = timeline.valid.to_numpy() > 0
            targets = truth[(truth.subject_key.astype(str) == str(key[0]))
                            & (truth.session_id.astype(str) == str(key[1]))]
            excluded = ignore[(ignore.subject_key.astype(str) == str(key[0]))
                              & (ignore.session_id.astype(str) == str(key[1]))]
            for interval in excluded.itertuples(index=False):
                valid &= (times < interval.start_ms) | (times > interval.end_ms)
            endpoints = targets[field].to_numpy(np.int64)
            eligible_endpoints = endpoints[(endpoints > times[0] - 3000) & (endpoints <= times[-1])]
            from bme_eating.endpoint_peaks import CausalOnsetPeaks
            detector = CausalOnsetPeaks(threshold)
            available = []
            if name == "onset":
                for timestamp, score, observed in zip(times, scores, timeline.valid):
                    peak = detector.observe(int(timestamp), float(score), float(observed))
                    if peak is not None and valid[np.searchsorted(times, peak[0])]:
                        available.append(peak[0])
            elif "state_probability" in timeline:
                from bme_eating.candidates import StreamingProposals
                stream = StreamingProposals(str(key[0]), str(key[1]), "diagnostic",
                    {"high_threshold": .10, "low_ratio": .90, "ema_half_life_seconds": 6,
                     "gap_merge_seconds": 15, "jitter_seconds": [],
                     "backtrack_seconds": 600, "start_expansion_seconds": 600})
                for row in timeline.itertuples(index=False):
                    previous = {source: entry["end"] for source, entry in stream.active.items()}
                    stream.observe(int(row.timestamp_ms), {"state": float(row.state_probability)},
                                   float(row.valid), float(row.offset_probability))
                    if row.offset_probability >= .50:
                        available.extend(end for source, end in previous.items() if source not in stream.active)
            available = np.asarray(sorted(set(available)), dtype=np.int64)
            usable_peaks += len(available)
            usable_background += sum(not ((timestamp >= endpoints) & (timestamp <= endpoints + support)).any()
                                     for timestamp in available)
            for endpoint in eligible_endpoints:
                selected = valid & (times >= endpoint) & (times <= endpoint + support)
                eligible += int(selected.any())
                hits += int((scores[selected] >= threshold).any())
                usable_hits += int(selected.any() and ((available >= endpoint) & (available <= endpoint + support)).any())
            maximum = max(maximum, float(scores[valid].max()) if valid.any() else 0.0)
            previous = -10**18
            for index in np.flatnonzero(valid & (scores >= threshold)):
                neighborhood = np.flatnonzero(valid & (np.abs(times - times[index]) <= 9000))
                best = neighborhood[np.argmax(scores[neighborhood])]
                if best != index or times[index] - previous < 15_000:
                    continue
                previous = int(times[index])
                peaks += 1
                background += int(not ((previous >= endpoints) & (previous <= endpoints + support)).any())
        report[name] = {"threshold": threshold, "maximum_probability": maximum,
                        "eligible_endpoints": eligible, "endpoint_hits": hits,
                        "endpoint_hit_rate": hits / max(eligible, 1), "peaks": peaks,
                        "background_peaks": background, "background_peaks_per_hour": background / max(hours, 1e-9)}
        report[name].update({"usable_endpoint_hits": usable_hits,
            "usable_endpoint_hit_rate": usable_hits / max(eligible, 1), "usable_peaks": usable_peaks,
            "usable_background_peaks_per_hour": usable_background / max(hours, 1e-9),
            "usable_trigger_scope": "causal_onset_peaks" if name == "onset" else "fixed_state_decoder_offset_closure"})
    errors = {"start": [], "end": []}
    from bme_eating.candidates import candidate_targets
    parents = {}
    for item in candidates:
        parents.setdefault(item.ancestor_id, item)
    for item in parents.values():
        positive, _, _, start, end = candidate_targets(item, truth)
        if positive:
            errors["start"].append((item.coarse_start_ms - start) / 1000)
            errors["end"].append((item.coarse_end_ms - end) / 1000)
    report["coarse_parent_bias"] = {name: {"matched_parents": len(values),
        "mae_seconds": float(np.abs(values).mean()) if values else None,
        "signed_error_seconds": float(np.mean(values)) if values else None}
        for name, values in errors.items()}
    report["exposure_hours"] = hours
    return report


def classifier_diagnostics(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    report = {"candidates": len(labels), "positives": int(labels.sum()),
              "negatives": int((labels == 0).sum()), "roc_auc": None, "average_precision": None}
    if len(np.unique(labels)) == 2:
        report.update({"roc_auc": float(roc_auc_score(labels, logits)),
                       "average_precision": float(average_precision_score(labels, logits))})
    for name, target in (("positive", 1), ("negative", 0)):
        values = logits[labels == target]
        report[name + "_logits"] = {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "quantiles": np.quantile(values, [0, .1, .5, .9, 1]).tolist() if len(values) else []}
    return report


def source_coverage(candidates, truth, windows):
    from bme_eating.evaluation import candidate_coverage
    groups = {}
    for item in candidates:
        parent = item.generation_parameters.get("parent_source", item.source)
        groups.setdefault(f"{parent}:{item.scale_seconds}", []).append(item)
    return {name: candidate_coverage(pool, truth, windows) for name, pool in sorted(groups.items())}


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
