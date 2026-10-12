from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bme_eating.hierarchical_artifacts import sha256_file
from bme_eating.joint_capabilities import supported
from bme_eating.joint_protocol import digest

SPLIT_ALGORITHM = "subject_constrained_multistart_swap_v1"
DURATION_EDGES = (60, 300, 900, 1800)


def subject_statistics(inputs):
    subjects = set(inputs.subject_folds)
    _, _, events, segments = inputs.frames(subjects, labels=True)
    truth = events[events.evaluable.astype(bool)].copy()
    durations = (truth.end_ms - truth.start_ms).to_numpy() / 1000
    truth["duration_bin"] = np.searchsorted(DURATION_EDGES, durations, side="right")
    records = {}
    for subject in sorted(subjects):
        sessions = segments[segments.subject_key.astype(str).eq(subject)]
        target = truth[truth.subject_key.astype(str).eq(subject)]
        records[subject] = {"sessions": int(sessions.session_id.nunique()), "events": len(target),
            "same": int(target.hand_relation.eq("same").sum()),
            "different": int(target.hand_relation.eq("different").sum()),
            "unknown": int((~target.hand_relation.isin(["same", "different"])).sum()),
            "duration_bins": [int(target.duration_bin.eq(index).sum()) for index in range(5)],
            "gyro_all_missing": bool(len(sessions) and sessions.gyro_valid_fraction.eq(0).all())}
    return records


def cohort_statistics(records, subjects):
    rows = [records[subject] for subject in sorted(subjects)]
    return {"subjects": len(rows), **{name: sum(row[name] for row in rows)
            for name in ("sessions", "events", "same", "different", "unknown")},
            "duration_bins": [sum(row["duration_bins"][index] for row in rows) for index in range(5)],
            "gyro_missing_subjects": [subject for subject in sorted(subjects) if records[subject]["gyro_all_missing"]],
            "gyro_missing_events": sum(records[subject]["events"] for subject in subjects
                                       if records[subject]["gyro_all_missing"])}


def generate_split(inputs, *, seed=2026):
    records = subject_statistics(inputs)
    subjects = sorted(records)
    if len(subjects) != 39:
        raise RuntimeError("v414 requires the audited 39-subject cohort")
    missing = [subject for subject in subjects if records[subject]["gyro_all_missing"]]
    if len(missing) != 2:
        raise RuntimeError("Expected exactly two natural GYRO-missing subjects")
    original_outer = {subject for subject in subjects if inputs.subject_folds[subject] == 0}
    required = min(missing, key=lambda subject: (abs(records[subject]["events"] - 4), subject))
    excluded = original_outer | (set(missing) - {required})
    if required in excluded:
        raise RuntimeError("Missing-domain assignment conflicts with the original fold0 cohort")
    eligible = [subject for subject in subjects if subject not in excluded]
    columns = ("sessions", "events", "same", "different", "unknown")
    matrix = np.asarray([[records[subject][name] for name in columns]
                         + records[subject]["duration_bins"] for subject in subjects], dtype=float)
    total = matrix.sum(axis=0)
    target = np.asarray([352, 45, 18, 27, total[4] * 45 / total[1],
                         *(total[5:] * 45 / total[1])])
    scales = np.asarray([352, 45, 18, 27, max(target[4], 1), *(np.maximum(target[5:], 1))])
    weights = np.asarray([8, 16, 4, 4, 1, 1, 1, 1, 1, 1], dtype=float)
    indices = {subject: index for index, subject in enumerate(subjects)}
    required_index = indices[required]
    pool = np.asarray([indices[subject] for subject in eligible if subject != required])
    def objective(selected):
        aggregate = matrix[list(selected)].sum(axis=0)
        return float(np.sum(weights * np.square((aggregate - target) / scales)))
    def key(selected):
        return objective(selected), tuple(subjects[index] for index in sorted(selected))
    rng = np.random.default_rng(seed)
    starts = {tuple(sorted([required_index, *rng.choice(pool, 10, replace=False).tolist()]))
              for _ in range(4096)}
    best = None
    for start in sorted(starts, key=key)[:16]:
        selected = set(start)
        for _ in range(100):
            alternatives = [tuple(sorted((selected - {old}) | {new}))
                            for old in sorted(selected - {required_index}) for new in pool if new not in selected]
            winner = min([tuple(sorted(selected)), *alternatives], key=key)
            if key(winner) >= key(tuple(sorted(selected))):
                break
            selected = set(winner)
        if best is None or key(selected) < key(best):
            best = selected
    development = [subjects[index] for index in sorted(best)]
    training = sorted(set(subjects) - set(development))
    actual = matrix[sorted(best)].sum(axis=0)
    return {"schema": "v414", "algorithm": SPLIT_ALGORITHM, "seed": seed,
        "data_identity": inputs.identity, "data_sha256": digest(inputs.identity),
        "training_subjects": training, "development_subjects": development, "all_subjects": subjects,
        "original_fold0_in_training": sorted(original_outer), "subject_statistics": records,
        "training_statistics": cohort_statistics(records, training),
        "development_statistics": cohort_statistics(records, development),
        "duration_edges_seconds": list(DURATION_EDGES),
        "search": {"starts": 4096, "refined_starts": 16, "maximum_swaps": 100,
                   "feature_names": [*columns, *(f"duration_bin_{index}" for index in range(5))],
                   "target": target.tolist(), "weights": weights.tolist(), "scales": scales.tolist(),
                   "objective": objective(best), "deviations": (actual - target).tolist(),
                   "tie_break": "sorted_stable_subject_ids", "uses_model_results": False}}


def validate_split(split, inputs=None):
    train, dev, all_subjects = (split.get(name, []) for name in
                              ("training_subjects", "development_subjects", "all_subjects"))
    if (split.get("schema") != "v414" or split.get("seed") != 2026
            or split.get("algorithm") != SPLIT_ALGORITHM
            or len(train) != 28 or len(dev) != 11 or len(all_subjects) != 39
            or len(set(all_subjects)) != 39 or len(set(train)) != 28 or len(set(dev)) != 11
            or set(train) & set(dev) or set(train) | set(dev) != set(all_subjects)
            or split.get("data_sha256") != digest(split.get("data_identity"))):
        raise RuntimeError("Invalid frozen v414 subject partition or data identity")
    records = split["subject_statistics"]
    for name, subjects in (("training", train), ("development", dev)):
        statistics = cohort_statistics(records, subjects)
        if statistics != split[name + "_statistics"] or len(statistics["gyro_missing_subjects"]) != 1:
            raise RuntimeError("Frozen split statistics or missing-domain constraint differs")
    if not set(split.get("original_fold0_in_training", [])) <= set(train):
        raise RuntimeError("Original fold0 subjects must enter development training")
    if inputs is not None:
        if (split["data_identity"] != inputs.identity or set(all_subjects) != set(inputs.subject_folds)
                or records != subject_statistics(inputs)):
            raise RuntimeError("Frozen split differs from current audited inputs")
        _, _, _, segments = inputs.frames(set(all_subjects), labels=False)
        if segments.groupby("session_id").subject_key.nunique().gt(1).any():
            raise RuntimeError("Session identity spans multiple subjects")
    return split


def load_split(config, inputs=None):
    root = Path(config["_config_path"]).resolve().parents[1]
    path = (root / config["development"]["split_path"]).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != config["development"]["split_sha256"]:
        raise RuntimeError("Frozen v414 split hash changed; create a new run")
    return validate_split(json.loads(path.read_text(encoding="utf-8")), inputs)


def validate_development_calibration(inputs, subjects, config, preprocessing, evidence, *, training):
    if training or not supported(config, "full_data_workflow") or preprocessing is None:
        raise RuntimeError("Seen-subject inference is restricted to final v414 development calibration")
    split = load_split(config)
    if (evidence.get("kind") != "development_on_final_model"
            or evidence.get("split_sha256") != config["development"]["split_sha256"]
            or evidence.get("data_sha256") != digest(inputs.identity)
            or set(subjects) != set(split["development_subjects"])
            or set(preprocessing["training_subjects"]) != set(split["all_subjects"])
            or set(preprocessing["scaler"]["training_subjects"]) != set(split["all_subjects"])):
        raise RuntimeError("Final development calibration cohort or identity differs")
