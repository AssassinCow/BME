from __future__ import annotations

import numpy as np


def endpoint_diagnostics(windows, truth, ignore, candidates):
    report = {}
    hours = float((windows.valid > 0).sum() * 3 / 3600)
    for name, field, threshold, support in (("onset", "start_ms", .30, 30_000),
                                            ("offset", "end_ms", .50, 60_000)):
        hits, eligible, background, peaks, maximum = 0, 0, 0, 0, 0.0
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
            for endpoint in eligible_endpoints:
                selected = valid & (times >= endpoint) & (times <= endpoint + support)
                eligible += int(selected.any())
                hits += int((scores[selected] >= threshold).any())
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
    errors = {"start": [], "end": []}
    from bme_eating.joint_stream import candidate_targets
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
