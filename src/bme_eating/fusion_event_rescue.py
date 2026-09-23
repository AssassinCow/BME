"""Protocol v5: fixed event rescue, with no label access in the inference path.

Development replay is deliberately separate from clean-Git experiment registration.
The inherited legacy decoder (including its frozen boundary channels and max-child
merge score) is unchanged. No additional boundary model is fitted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.splits import load_subject_folds
from bme_eating.dtp_postprocess import _event_diagnostics, fit_score_threshold
from bme_eating.fusion import _event_parameters, align_prediction_frames, json_safe, sha256_file
from bme_eating.fusion_v4 import (
    evaluate_v4_gate,
    paired_subject_bootstrap,
    summarize_event_predictions,
)
from bme_eating.metrics import partition_evaluation_events
from bme_eating.postprocess import probabilities_to_events
from bme_eating.reproducibility import require_clean_git_worktree

ROOT = Path(__file__).resolve().parents[2]
EVENT_COLUMNS = ["subject_key", "session_id", "start_ms", "end_ms", "score"]
ORDER = EVENT_COLUMNS[:4]
GENERATOR = {
    "detector_mode": "dual_ema",
    "fast_ema_half_life_seconds": 12,
    "slow_ema_half_life_seconds": 36,
    "fast_high_threshold": 0.55,
    "slow_high_threshold": 0.30,
    "exit_threshold_ratio": 0.25,
    "off_duration_seconds": 60,
    "minimum_event_seconds": 30,
    "merge_gap_seconds": 60,
    "boundary_lookback_seconds": 60,
}
GATE = {
    "minimum_f1_improvement": 0.02,
    "minimum_different_sensitivity_improvement": 0.03,
    "maximum_fp_per_hour_ratio": 1.2,
    "maximum_boundary_mae_ratio": 1.1,
    "maximum_partition_f1_drop": 0.01,
    "maximum_partition_strict_f1_drop": 0.01,
}
SETTINGS = {
    "protocol_version": 5,
    "score_quantile": 0.90,
    "exclusion_seconds": 120,
    "generator": GENERATOR,
    "bootstrap_replicates": 1000,
    "bootstrap_seed": 2026,
    "stress_folds": [2, 3, 4],
    "minimum_nondegraded_folds": 2,
    "maximum_fold_f1_drop": 0.03,
    "gate": GATE,
}
# Approved development replay, not targets for tuning. Counts disambiguate F1.
REPLAY = {
    0: {
        "baseline": (69, 62, 60),
        "candidate": (79, 75, 50),
        "different_sensitivity": 0.5,
        "strict_no_ignore_f1": 0.4817073170731707,
        "start_mae_seconds": 262.356,
        "end_mae_seconds": 82.29843037974683,
    },
    1: {
        "baseline": (69, 88, 60),
        "candidate": (80, 103, 49),
        "different_sensitivity": 0.5205479452054794,
        "strict_no_ignore_f1": 0.4444444444444444,
        "start_mae_seconds": 271.0872625,
        "end_mae_seconds": 106.8641625,
    },
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object: {path.name}")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True).encode()).hexdigest()


def _json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", value):
        raise ValueError("Run names must contain only ASCII letters, digits, _ and -")
    return value


def validate_protocol(config: dict[str, Any]) -> None:
    if config.get("event_rescue") != SETTINGS or config["fusion"]["protocol_version"] != 5:
        raise ValueError("Protocol v5 is fixed; q, buffer, decoder and gates cannot be changed")
    if config["data"]["output_step_seconds"] != 3:
        raise ValueError("Protocol v5 requires 3-second output steps")


def _identity(config: dict[str, Any], *, clean: bool = True) -> dict[str, Any]:
    commit = (
        require_clean_git_worktree()
        if clean
        else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    paths = sorted(
        [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("configs/*.yaml")]
    )
    return {
        "git_commit": commit,
        "code_sha256": _digest({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}),
        "config_sha256": _digest({k: v for k, v in config.items() if not k.startswith("_")}),
        "protocol": SETTINGS,
    }


@contextmanager
def _lock(path: Path):
    """OS lock is released even after a killed process; file is not a status flag."""
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _validate_events(frame: pd.DataFrame, name: str) -> None:
    if set(EVENT_COLUMNS) - set(frame.columns):
        raise ValueError(f"{name}: missing event columns")
    if frame[EVENT_COLUMNS].isna().any().any():
        raise ValueError(f"{name}: null event values")
    numeric = frame[["start_ms", "end_ms", "score"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 1] <= numeric[:, 0]).any():
        raise ValueError(f"{name}: invalid event duration or nonfinite values")
    if (numeric[:, :2] != np.floor(numeric[:, :2])).any():
        raise ValueError(f"{name}: timestamps must be integer milliseconds")
    if ((numeric[:, 2] < 0) | (numeric[:, 2] > 1)).any():
        raise ValueError(f"{name}: invalid probability")
    if frame.duplicated(ORDER).any():
        raise ValueError(f"{name}: duplicate events")


def append_rescue_events(
    baseline: pd.DataFrame, dtp_events: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """O(B+D) scan after stable sorting; no labels and no XGBoost edits."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A finite train-only threshold in [0, 1] is required")
    _validate_events(baseline, "baseline")
    _validate_events(dtp_events, "DTP")
    base = baseline[EVENT_COLUMNS].sort_values(ORDER, kind="stable").reset_index(drop=True)
    candidates = dtp_events[EVENT_COLUMNS].sort_values(ORDER, kind="stable").reset_index(drop=True)
    grouped = {key: group for key, group in base.groupby(ORDER[:2], sort=False)}
    accepted, reasons = [], []
    for key, group in candidates.groupby(ORDER[:2], sort=False):
        reference = grouped.get(key, base.iloc[:0])
        starts = reference.start_ms.to_numpy(dtype=np.int64)
        # Prefix max handles even overlapping/nested baseline intervals without altering them.
        ends = np.maximum.accumulate(reference.end_ms.to_numpy(dtype=np.int64))
        cursor = 0
        for index, event in group.iterrows():
            reason = "below_score_threshold"
            if event.score >= threshold:
                while cursor < len(ends) and ends[cursor] < event.start_ms - 120_000:
                    cursor += 1
                duplicate = cursor < len(starts) and starts[cursor] <= event.end_ms + 120_000
                reason = "baseline_within_120s" if duplicate else "accepted"
                if not duplicate:
                    accepted.append(index)
            reasons.append({**event.to_dict(), "reason": reason, "score_threshold": threshold})
    extra = candidates.loc[accepted]
    if extra.empty:
        combined = base.copy()
    elif base.empty:
        combined = extra.copy()
    else:
        combined = pd.concat([base, extra], ignore_index=True)
    combined = combined.sort_values(ORDER, kind="stable").reset_index(drop=True)
    # Multiset identity, including the exact float score; no approximate comparison.
    recovered = combined.merge(base[ORDER], on=ORDER, how="inner", validate="one_to_one")
    pd.testing.assert_frame_equal(
        recovered[EVENT_COLUMNS].sort_values(ORDER).reset_index(drop=True),
        base,
        check_dtype=False,
        check_exact=True,
    )
    audit = pd.DataFrame(reasons, columns=[*EVENT_COLUMNS, "reason", "score_threshold"])
    return combined, audit


def apply_frozen_event_rescue(
    baseline: pd.DataFrame, dtp: pd.DataFrame, selection: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deployment API. Only frozen numbers and predictions, never labels/search."""
    if selection.get("protocol") != SETTINGS:
        raise ValueError("Unexpected frozen event rescue protocol")
    baseline, dtp = align_prediction_frames(baseline, dtp)
    _validate_timeline(dtp)
    base_events = probabilities_to_events(
        baseline, **_event_parameters(selection["baseline_postprocess"])
    )
    generated = probabilities_to_events(dtp, **GENERATOR)
    combined, audit = append_rescue_events(base_events, generated, selection["score_threshold"])
    return combined, base_events, audit


def _validate_timeline(predictions):
    if predictions[["subject_key", "session_id", "timestamp_ms"]].isna().any().any():
        raise ValueError("Null timeline identity")
    for _, group in predictions.groupby(["subject_key", "session_id"], sort=False):
        times = group.timestamp_ms.sort_values().to_numpy()
        if (times != np.floor(times)).any():
            raise ValueError("Timestamps must use integer milliseconds")
        # v5 deliberately retains the frozen decoder's session policy. Re-splitting
        # here would change both baseline events and the approved development replay.


def _metrics(predictions, predicted, labels, postprocess):
    truth, ignore = partition_evaluation_events(labels, set(predictions.subject_key.astype(str)))
    return summarize_event_predictions(
        predictions,
        predicted,
        truth,
        ignore,
        iou_threshold=postprocess["iou_threshold"],
        matching_method=postprocess["matching_method"],
    )


def rescue_gate(candidate, baseline, partitions=None):
    gate = evaluate_v4_gate(candidate, baseline, GATE, partitions)
    gate["checks"].update(
        {
            "baseline_event_identity_preserved": True,  # asserted by append_rescue_events
            "tp_not_lower": candidate["true_positive"] >= baseline["true_positive"],
            "recall_not_lower": candidate["sensitivity"] >= baseline["sensitivity"],
        }
    )
    gate["passed"] = all(gate["checks"].values())
    return gate


def crossfit_event_rescue(baseline, dtp, labels, postprocess):
    baseline, dtp = align_prediction_frames(baseline, dtp)
    _validate_timeline(dtp)
    assignments = dtp[["subject_key", "calibration_fold"]].drop_duplicates()
    if assignments.subject_key.duplicated().any() or sorted(
        assignments.calibration_fold.unique()
    ) != [0, 1, 2]:
        raise ValueError("Exactly three subject-disjoint calibration folds are required")
    print("Decoding fixed XGBoost and legacy DTP events once...", flush=True)
    base_events = probabilities_to_events(baseline, **_event_parameters(postprocess))
    generated = probabilities_to_events(dtp, **GENERATOR)
    pieces, audits, partitions = [], [], []
    for heldout in range(3):
        train = set(assignments.loc[assignments.calibration_fold != heldout, "subject_key"])
        validation = set(assignments.loc[assignments.calibration_fold == heldout, "subject_key"])
        if train & validation or not train or not validation:
            raise ValueError("Meta train/heldout subjects overlap or are empty")
        training_events = generated[generated.subject_key.isin(train)]
        if training_events.empty:
            raise ValueError("No training candidate events; cannot fit score threshold")
        threshold = float(fit_score_threshold(training_events, 0.90))
        b = base_events[base_events.subject_key.isin(validation)]
        d = generated[generated.subject_key.isin(validation)]
        combined, audit = append_rescue_events(b, d, threshold)
        p = dtp[dtp.subject_key.isin(validation)]
        bm, cm = _metrics(p, b, labels, postprocess), _metrics(p, combined, labels, postprocess)
        if cm["true_positive"] < bm["true_positive"] or cm["sensitivity"] < bm["sensitivity"]:
            raise AssertionError("Append-only rescue decreased baseline TP/recall")
        partitions.append(
            {
                "heldout_fold": heldout,
                "training_subjects": sorted(train),
                "validation_subjects": sorted(validation),
                "score_threshold": threshold,
                "baseline_metrics": bm,
                "candidate_metrics": cm,
            }
        )
        audit["heldout_fold"] = heldout
        pieces.append(combined)
        audits.append(audit)
        print(f"Meta split {heldout + 1}/3 complete (threshold={threshold:.6f})", flush=True)
    combined = pd.concat(pieces, ignore_index=True).sort_values(ORDER).reset_index(drop=True)
    bm = _metrics(dtp, base_events, labels, postprocess)
    cm = _metrics(dtp, combined, labels, postprocess)
    record = {
        "protocol": SETTINGS,
        "baseline_postprocess": postprocess,
        "score_threshold": float(fit_score_threshold(generated, 0.90)),
        "threshold_training_subjects": sorted(assignments.subject_key.tolist()),
        "score_reference": generated[["subject_key", "score"]].to_dict("records"),
        "partitions": partitions,
        "baseline_metrics": bm,
        "candidate_metrics": cm,
        "meta_oof_gate": rescue_gate(cm, bm, partitions),
        "outer_predictions_read": False,
        "timeline_diagnostics": {
            "session_policy": "preserve_frozen_session_ids_legacy_decoder_no_new_gap_reset",
            "within_session_gaps_over_6s": int(
                (
                    dtp.sort_values(["subject_key", "session_id", "timestamp_ms"])
                    .groupby(["subject_key", "session_id"])
                    .timestamp_ms.diff()
                    > 6000
                ).sum()
            ),
        },
        "limitations": [
            "Protocol chosen after observing development folds 0/1; not independent validation.",
            "Frozen XGBoost decoder was selected on outer-train OOF, not nested meta-train.",
            "Legacy DTP decoder retains frozen boundary channels and max-child merge scores.",
        ],
    }
    return record, combined, base_events, pd.concat(audits, ignore_index=True)


def assert_development_replay(record: dict[str, Any], fold: int) -> None:
    expected = REPLAY[fold]
    for role in ("baseline", "candidate"):
        metrics = record[f"{role}_metrics"]
        counts = tuple(metrics[k] for k in ("true_positive", "false_positive", "false_negative"))
        if counts != expected[role]:
            raise RuntimeError(f"Development replay differs for fold {fold} {role}: {counts}")
    for name, value in expected.items():
        if name not in ("baseline", "candidate") and not math.isclose(
            record["candidate_metrics"][name], value, rel_tol=0, abs_tol=1e-8
        ):
            raise RuntimeError(f"Development replay differs for fold {fold}: {name}")


def _verify_artifacts(directory: Path, manifest: dict[str, Any]) -> None:
    hashes = manifest.get("artifact_hashes", {})
    if not hashes:
        raise RuntimeError("Missing artifact hashes")
    for name, expected in hashes.items():
        if Path(name).name != name or sha256_file(directory / name) != expected:
            raise RuntimeError(f"Frozen artifact changed: {name}")


def _manifest(directory, identity, provenance):
    _json(
        directory / "run_manifest.json",
        {
            "identity": identity,
            "provenance": provenance,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "random_seeds": {"bootstrap": 2026},
            "artifact_hashes": {
                p.name: sha256_file(p)
                for p in sorted(directory.iterdir())
                if p.is_file()
                and p.suffix not in (".lock", ".tmp")
                and p.name != "run_manifest.json"
            },
        },
    )


def _inputs(root, config, source_run, fold, *, outer=False):
    """Verify only the prediction scope being consumed: fitting never opens test files."""
    baseline_dir = (
        root / "experiments" / _name(config["experiment"]["baseline_name"]) / f"fold_{fold}"
    )
    source_dir = root / "experiments" / _name(source_run) / f"fold_{fold}"
    hashes = {}
    for role, directory, names in (
        (
            "baseline",
            baseline_dir,
            [
                "test_predictions.parquet" if outer else "validation_predictions.parquet",
                "selected_postprocess.json",
            ],
        ),
        (
            "dtp",
            source_dir,
            ["dtp_test_predictions.parquet" if outer else "dtp_oof_predictions.parquet"],
        ),
    ):
        manifest = _read(directory / "run_manifest.json")
        if manifest.get("experiment") != {"name": directory.parent.name, "fold": fold}:
            raise RuntimeError(f"{role} manifest identity differs")
        if role == "dtp" and manifest.get("git", {}).get("dirty") is not False:
            raise RuntimeError("DTP predictions require clean source provenance")
        if role == "baseline":
            claimed = manifest.get("artifact_provenance", {}).get(
                "claimed_source_commit", manifest.get("git", {}).get("commit")
            )
            if claimed != config["experiment"]["baseline_source_commit"]:
                raise RuntimeError("Baseline source commit differs")
        for name in names:
            actual = sha256_file(directory / name)
            if actual != manifest.get("artifact_hashes", {}).get(name):
                raise RuntimeError(f"Frozen {role} artifact changed: {name}")
            hashes[f"{role}/{name}"] = actual
        hashes[f"{role}/run_manifest.json"] = sha256_file(directory / "run_manifest.json")
        for key, name in (
            ("events", "events.parquet"),
            ("subject_folds", "subject_folds.json"),
            ("subject_folds_manifest", "subject_folds.manifest.json"),
        ):
            actual = sha256_file(root / "indices" / name)
            if actual != manifest.get("hashes", {}).get(key):
                raise RuntimeError(f"{role} input fingerprint differs: {key}")
            hashes[f"input/{name}"] = actual
    bpath = baseline_dir / (
        "test_predictions.parquet" if outer else "validation_predictions.parquet"
    )
    dpath = source_dir / (
        "dtp_test_predictions.parquet" if outer else "dtp_oof_predictions.parquet"
    )
    baseline, dtp = align_prediction_frames(pd.read_parquet(bpath), pd.read_parquet(dpath))
    fold_map = load_subject_folds(root / "indices" / "subject_folds.json")
    expected = {s for s, f in fold_map.items() if (f == fold) == outer}
    if set(dtp.subject_key) != expected:
        raise RuntimeError("Prediction subjects disagree with outer fold map")
    source_manifest = _read(source_dir / "run_manifest.json")
    outer_train = {s for s, f in fold_map.items() if f != fold}
    validation_sets = []
    for partition in range(3):
        name = f"crossfit_{partition}/metadata.json"
        actual = sha256_file(source_dir / name)
        if source_manifest.get("artifact_hashes", {}).get(name) != actual:
            raise RuntimeError("DTP crossfit metadata fingerprint differs")
        metadata = _read(source_dir / name)
        train = set(metadata["train_subjects"])
        valid = set(metadata["validation_subjects"])
        if (
            metadata["outer_fold"] != fold
            or metadata["inner_validation_partition"] != partition
            or train & valid
            or train | valid != outer_train
        ):
            raise RuntimeError("DTP model training/validation scope leaks or differs")
        if not outer and valid != set(dtp.loc[dtp.calibration_fold == partition, "subject_key"]):
            raise RuntimeError("DTP OOF calibration_fold disagrees with its model metadata")
        validation_sets.append(valid)
        hashes[f"dtp/{name}"] = actual
    if set.union(*validation_sets) != outer_train or sum(map(len, validation_sets)) != len(
        outer_train
    ):
        raise RuntimeError("DTP validation partitions overlap or omit subjects")
    postprocess = _read(baseline_dir / "selected_postprocess.json")
    if (
        postprocess["iou_threshold"] != 0.25
        or postprocess["matching_method"] != "max_cardinality_iou"
    ):
        raise ValueError("Protocol v5 requires strict IoU > 0.25 maximum cardinality matching")
    labels = pd.read_parquet(
        root / "indices" / "events.parquet", filters=[("subject_key", "in", sorted(expected))]
    )
    # Schema validation also catches missing labels/coverage before computation.
    partition_evaluation_events(labels, expected)
    return baseline, dtp, labels, postprocess, hashes


def _verify_input_hashes(root, config, source_run, fold, hashes):
    paths = {
        "input": root / "indices",
        "baseline": root / "experiments" / config["experiment"]["baseline_name"] / f"fold_{fold}",
        "dtp": root / "experiments" / source_run / f"fold_{fold}",
    }
    for key, expected in hashes.items():
        role, name = key.split("/", 1)
        if sha256_file(paths[role] / name) != expected:
            raise RuntimeError(f"Input changed during execution: {key}")


def _write_evidence(directory, predictions, combined, base_events, audit, labels, postprocess):
    combined.to_parquet(directory / "events.parquet", index=False)
    base_events.to_parquet(directory / "baseline_events.parquet", index=False)
    audit.to_csv(directory / "dtp_decisions.csv", index=False)
    audit[audit.reason == "accepted"].to_csv(directory / "accepted_dtp_events.csv", index=False)
    audit[audit.reason != "accepted"].to_csv(directory / "rejected_dtp_events.csv", index=False)
    truth, ignore = partition_evaluation_events(labels, set(predictions.subject_key))
    iou, method = postprocess["iou_threshold"], postprocess["matching_method"]
    for name, events in (("candidate", combined), ("baseline", base_events)):
        failures, subjects, hands = _event_diagnostics(
            predictions, events, truth, ignore, iou, method
        )
        failures.to_csv(directory / f"{name}_failure_cases.csv", index=False)
        subjects.to_csv(directory / f"{name}_per_subject.csv", index=False)
        _json(directory / f"{name}_hand_metrics.json", hands)
    print("Computing 1000 paired subject bootstrap replicates...", flush=True)
    bootstrap = paired_subject_bootstrap(
        predictions,
        combined,
        predictions,
        base_events,
        truth,
        ignore,
        iou_threshold=iou,
        matching_method=method,
        replicates=1000,
        seed=2026,
    )
    _json(directory / "bootstrap.json", {"seed": 2026, "replicates": 1000, "deltas": bootstrap})
    # Compact timeline preserves the exact exposure denominator for pooled aggregation.
    bounds = predictions.groupby(["subject_key", "session_id"]).timestamp_ms.agg(["min", "max"])
    timeline = pd.concat(
        [
            bounds["min"].rename("timestamp_ms").reset_index(),
            bounds["max"].rename("timestamp_ms").reset_index(),
        ]
    ).drop_duplicates()
    timeline.to_parquet(directory / "exposure.parquet", index=False)
    labels.to_parquet(directory / "evaluation_labels.parquet", index=False)


def _context(args):
    config = load_config(args.config)
    validate_protocol(config)
    identity = _identity(config)
    _, root = resolve_roots(config)
    run = root / "experiments" / _name(args.run_name)
    return config, identity, root, run


def _load_fit(run, fold, identity):
    directory = run / f"fold_{fold}"
    manifest = _read(directory / "run_manifest.json")
    if manifest["identity"] != identity:
        raise RuntimeError("Frozen code, Git or config identity changed")
    _verify_artifacts(directory, manifest)
    selection = _read(directory / "selected_event_rescue.json")
    if selection["run_name"] != run.name or selection["fold"] != fold:
        raise RuntimeError("Selection identity differs")
    return selection, manifest


def fit_event_rescue(args):
    config, identity, root, run = _context(args)
    fold = int(args.fold)
    if fold >= 2:
        require_registration(root, args.run_name, config, args.source_run)
    baseline, dtp, labels, postprocess, hashes = _inputs(root, config, args.source_run, fold)
    signature = _digest(
        {"identity": identity, "inputs": hashes, "fold": fold, "source_run": args.source_run}
    )
    directory = run / f"fold_{fold}"
    if args.fresh:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise FileNotFoundError("--resume requires an existing v5 fit")
    with _lock(directory / "fit.lock"):
        checkpoint = directory / "checkpoint.json"
        if checkpoint.exists() and _read(checkpoint)["signature"] != signature:
            raise RuntimeError(
                "Cannot resume after prediction, label, split, config or code changes"
            )
        if (directory / "run_manifest.json").exists():
            _load_fit(run, fold, identity)
            print(f"Verified completed fit: {directory}", flush=True)
            return directory
        _json(checkpoint, {"signature": signature})
        print("[1/3] Fixed decoding and train-only threshold crossfit...", flush=True)
        record, combined, base_events, audit = crossfit_event_rescue(
            baseline, dtp, labels, postprocess
        )
        if fold in REPLAY:
            assert_development_replay(record, fold)
        record.update(
            {
                "run_name": run.name,
                "fold": fold,
                "source_run": args.source_run,
                "input_hashes": hashes,
                "identity": identity,
                "scope": "development" if fold < 2 else "locked_stress_test",
            }
        )
        print("[2/3] Saving metrics, decisions and uncertainty...", flush=True)
        _write_evidence(directory, dtp, combined, base_events, audit, labels, postprocess)
        combined.to_csv(directory / "meta_oof_events.csv", index=False)
        _json(
            directory / "meta_oof_metrics.json",
            {
                "baseline": record["baseline_metrics"],
                "candidate": record["candidate_metrics"],
                "gate": record["meta_oof_gate"],
                "partitions": record["partitions"],
            },
        )
        _json(directory / "selected_event_rescue.json", record)
        # Detect code/input edits while a long computation was in progress.
        if _identity(config) != identity:
            raise RuntimeError("Code changed during fitting")
        _verify_input_hashes(root, config, args.source_run, fold, hashes)
        _manifest(directory, identity, {"inputs": hashes, "scope": record["scope"]})
        print(f"[3/3] Fit complete, no outer predictions read: {directory}", flush=True)
    return directory


def register_event_rescue(args):
    config, identity, root, run = _context(args)
    records = [_load_fit(run, fold, identity)[0] for fold in (0, 1)]
    if len({r["source_run"] for r in records}) != 1:
        raise RuntimeError("Development folds use different prediction source runs")
    for fold, record in enumerate(records):
        assert_development_replay(record, fold)
        _verify_input_hashes(root, config, record["source_run"], fold, record["input_hashes"])
    payload = {
        "identity": identity,
        "run_name": run.name,
        "source_run": records[0]["source_run"],
        "development_manifests": {
            str(f): sha256_file(run / f"fold_{f}" / "run_manifest.json") for f in (0, 1)
        },
        "status": "locked_stress_test_authorized_not_promoted",
        "known_development_failure": "fold 0 FP/hour ratio 1.209677 exceeds 1.2",
        "stress_folds": [2, 3, 4],
    }
    with _lock(run / "registration.lock"):
        path = run / "protocol_registration.json"
        if path.exists():
            if _read(path) != payload:
                raise RuntimeError("Immutable protocol registration differs")
        else:
            _json(path, payload)
    print(path, flush=True)
    return path


def require_registration(root, run_name, config, source_run):
    validate_protocol(config)
    run = root / "experiments" / _name(run_name)
    registered = _read(run / "protocol_registration.json")
    if registered["identity"] != _identity(config) or registered["run_name"] != run_name:
        raise RuntimeError("Registered protocol code/config/Git changed")
    if registered["source_run"] != source_run:
        raise RuntimeError("Registered DTP source run differs")
    for fold in (0, 1):
        if (
            sha256_file(run / f"fold_{fold}" / "run_manifest.json")
            != registered["development_manifests"][str(fold)]
        ):
            raise RuntimeError("Registered development evidence changed")
        _load_fit(run, fold, registered["identity"])
    return registered


def evaluate_event_rescue(args):
    config, identity, root, run = _context(args)
    fold = int(args.fold)
    if fold not in (2, 3, 4):
        raise ValueError("Outer v5 evaluation is restricted to locked stress folds 2-4")
    selection, fit_manifest = _load_fit(run, fold, identity)
    registration = require_registration(root, run.name, config, selection["source_run"])
    # A failed development gate does not become a pass; this is the explicitly
    # registered all-three-fold stress experiment, with promotion decided only at the end.
    directory = run / f"fold_{fold}"
    with _lock(directory / "evaluation.lock"):
        final = directory / "outer"
        if final.exists():
            raise FileExistsError("Outer fold already evaluated; parameters cannot be switched")
        pending = directory / "outer.pending"
        if pending.exists():
            raise RuntimeError("Interrupted outer evaluation exists; preserve it for audit")
        _verify_input_hashes(root, config, selection["source_run"], fold, selection["input_hashes"])
        pending.mkdir()
        _json(
            pending / "attempt.json", {"identity": identity, "selection_sha256": _digest(selection)}
        )
        baseline, dtp, labels, postprocess, hashes = _inputs(
            root, config, selection["source_run"], fold, outer=True
        )
        if set(dtp.subject_key) & set(selection["threshold_training_subjects"]):
            raise RuntimeError("Outer subjects appeared in threshold training")
        print("[1/2] Applying frozen event rescue once...", flush=True)
        combined, base_events, audit = apply_frozen_event_rescue(baseline, dtp, selection)
        bm, cm = (
            _metrics(dtp, base_events, labels, postprocess),
            _metrics(dtp, combined, labels, postprocess),
        )
        report = {
            "fold": fold,
            "baseline": bm,
            "candidate": cm,
            "gate_diagnostic": rescue_gate(cm, bm),
            "promotion_status": "pending_all_stress_folds",
            "registration_sha256": _digest(registration),
        }
        print("[2/2] Saving immutable outer evidence...", flush=True)
        _write_evidence(pending, dtp, combined, base_events, audit, labels, postprocess)
        _json(pending / "metrics.json", report)
        _manifest(
            pending,
            identity,
            {
                "inputs": hashes,
                "fit_manifest_sha256": _digest(fit_manifest),
                "registration_sha256": _digest(registration),
            },
        )
        if _identity(config) != identity:
            raise RuntimeError("Code changed during evaluation")
        _verify_input_hashes(root, config, selection["source_run"], fold, hashes)
        pending.rename(final)
    print(final, flush=True)
    return final


def summarize_event_rescue(args):
    config, identity, root, run = _context(args)
    registered = _read(run / "protocol_registration.json")
    require_registration(root, run.name, config, registered["source_run"])
    predictions, candidates, baselines, labels, reports, partitions = [], [], [], [], [], []
    hashes, seen = {}, set()
    for fold in (2, 3, 4):
        selection, fit_manifest = _load_fit(run, fold, identity)
        partitions.extend(selection["partitions"])
        directory = run / f"fold_{fold}" / "outer"
        manifest = _read(directory / "run_manifest.json")
        if manifest["identity"] != identity:
            raise RuntimeError("Stress folds have different frozen identities")
        if manifest["provenance"]["fit_manifest_sha256"] != _digest(fit_manifest) or manifest[
            "provenance"
        ]["registration_sha256"] != _digest(registered):
            raise RuntimeError("Outer evidence belongs to a different fit or registration")
        _verify_artifacts(directory, manifest)
        p = pd.read_parquet(directory / "exposure.parquet")
        if seen & set(p.subject_key):
            raise RuntimeError("Stress outer folds overlap subjects")
        seen.update(p.subject_key)
        predictions.append(p)
        candidates.append(pd.read_parquet(directory / "events.parquet"))
        baselines.append(pd.read_parquet(directory / "baseline_events.parquet"))
        labels.append(pd.read_parquet(directory / "evaluation_labels.parquet"))
        report = _read(directory / "metrics.json")
        if report["fold"] != fold or report["registration_sha256"] != _digest(registered):
            raise RuntimeError("Outer report identity differs")
        reports.append(report)
        hashes[str(fold)] = sha256_file(directory / "run_manifest.json")
    p, c, b, y = [
        pd.concat(items, ignore_index=True)
        for items in (predictions, candidates, baselines, labels)
    ]
    post = {"iou_threshold": 0.25, "matching_method": "max_cardinality_iou"}
    bm, cm = _metrics(p, b, y, post), _metrics(p, c, y, post)
    gate = rescue_gate(cm, bm, partitions)
    deltas = [r["candidate"]["f1"] - r["baseline"]["f1"] for r in reports]
    gate["checks"].update(
        {
            "two_folds_not_degraded": sum(d >= 0 for d in deltas) >= 2,
            "no_fold_drop_over_003": min(deltas) >= -0.03,
            "each_fold_tp_not_lower": all(
                r["candidate"]["true_positive"] >= r["baseline"]["true_positive"] for r in reports
            ),
        }
    )
    gate["passed"] = all(gate["checks"].values())
    truth, ignore = partition_evaluation_events(y, seen)
    report = {
        "baseline": bm,
        "candidate": cm,
        "gate": gate,
        "folds": reports,
        "outer_manifest_hashes": hashes,
        "submission_candidate": "event_rescue_v5" if gate["passed"] else "xgboost",
        "bootstrap": paired_subject_bootstrap(
            p,
            c,
            p,
            b,
            truth,
            ignore,
            iou_threshold=0.25,
            matching_method="max_cardinality_iou",
            replicates=1000,
            seed=2026,
        ),
    }
    with _lock(run / "summary.lock"):
        path = run / "stress_summary.json"
        if path.exists() and _read(path) != json_safe(report):
            raise RuntimeError("Immutable stress summary differs")
        _json(path, report)
        _json(
            run / "stress_summary.manifest.json",
            {
                "identity": identity,
                "artifact_hashes": {path.name: sha256_file(path)},
                "outer_manifest_hashes": hashes,
                "registration_sha256": sha256_file(run / "protocol_registration.json"),
            },
        )
    print(json.dumps(json_safe(gate)), flush=True)
    print(path, flush=True)
    return path
