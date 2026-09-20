from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_INPUT_FILES = {
    "quality_report": "quality_report.json",
    "quality_expectations": "quality_expectations.json",
    "subject_folds": "subject_folds.json",
    "subject_folds_manifest": "subject_folds.manifest.json",
    "events": "events.parquet",
    "anchors": "anchors.parquet",
    "segments": "segments.parquet",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _validate_hashes(directory: Path, hashes: dict[str, Any], *, label: str) -> None:
    for name, expected in hashes.items():
        path = directory / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} declared by manifest: {path}")
        if _sha256(path) != str(expected):
            raise RuntimeError(f"{label.capitalize()} changed after manifesting: {path}")


def _load_folds(root: Path) -> list[dict[str, Any]]:
    rows = []
    seen_subjects: dict[str, int] = {}
    indices_root = root.parent.parent / "indices"
    for fold in range(5):
        path = root / f"fold_{fold}" / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold metrics: {path}")
        payload = _read_object(path)
        method = str(payload.get("primary_method", "max_cardinality_iou"))
        manifest_path = root / f"fold_{fold}" / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing fold manifest: {manifest_path}")
        manifest = _read_object(manifest_path)
        if int(manifest.get("version", 0)) < 2:
            raise RuntimeError(f"Fold {fold} manifest predates the strict comparison schema")
        git_record = manifest.get("git", {})
        if git_record.get("dirty") is not False:
            raise RuntimeError(f"Fold {fold} was produced or backfilled from a dirty working tree")
        commit = str(git_record.get("commit") or "")
        config_hash = str(manifest.get("resolved_config_sha256") or "")
        if not commit or not config_hash:
            raise RuntimeError(f"Fold {fold} manifest lacks commit or resolved-config identity")
        experiment = manifest.get("experiment", {})
        if experiment.get("name") != root.name or experiment.get("fold") != fold:
            raise RuntimeError(f"Fold {fold} manifest belongs to a different experiment or fold")
        artifact_hashes = manifest.get("artifact_hashes", {})
        if not isinstance(artifact_hashes, dict):
            raise TypeError(f"Fold {fold} artifact_hashes must be a JSON object")
        _validate_hashes(path.parent, artifact_hashes, label="artifact")
        model_hashes = manifest.get("model_hashes", {})
        if not isinstance(model_hashes, dict):
            raise TypeError(f"Fold {fold} model_hashes must be a JSON object")
        _validate_hashes(path.parent, model_hashes, label="model artifact")
        expected_metrics_hash = artifact_hashes.get("test_metrics.json")
        if not expected_metrics_hash or _sha256(path) != expected_metrics_hash:
            raise RuntimeError(f"Fold {fold} test_metrics.json is missing or changed after manifesting")
        selected_fusion_path = root / f"fold_{fold}" / "selected_fusion.json"
        selection_run_name = None
        internal_fusion_gate_passed = None
        if selected_fusion_path.is_file():
            expected_selection_hash = artifact_hashes.get("selected_fusion.json")
            if not expected_selection_hash or _sha256(selected_fusion_path) != expected_selection_hash:
                raise RuntimeError(
                    f"Fold {fold} selected_fusion.json is missing or changed after manifesting"
                )
            selection = _read_object(selected_fusion_path)
            selection_run_name = str(selection.get("run_name") or "")
            if selection_run_name != root.name:
                raise RuntimeError(f"Fold {fold} selected fusion belongs to another run")
            if int(selection.get("version", 0)) < 3 or int(
                selection.get("crossfit_partitions", 0)
            ) != 3:
                raise RuntimeError(
                    f"Fold {fold} selected fusion predates the three-way cross-fit protocol"
                )
            required_crossfit_artifacts = {
                "dtp_oof_predictions.parquet",
                "dtp_test_predictions.parquet",
                *{
                    f"crossfit_{partition}/{name}"
                    for partition in range(3)
                    for name in (
                        "best.pt",
                        "metadata.json",
                        "normalization.json",
                        "best_validation_predictions.parquet",
                        "dtp_test_predictions.parquet",
                    )
                },
            }
            missing_crossfit_artifacts = sorted(
                required_crossfit_artifacts - set(artifact_hashes)
            )
            if missing_crossfit_artifacts:
                raise RuntimeError(
                    f"Fold {fold} manifest lacks cross-fit artifacts: "
                    f"{missing_crossfit_artifacts}"
                )
            internal_fusion_gate_passed = selection.get("internal_gate", {}).get("passed")
            if not isinstance(internal_fusion_gate_passed, bool):
                raise RuntimeError(f"Fold {fold} lacks a boolean internal fusion gate result")
        input_hashes = manifest.get("hashes", {})
        missing_inputs = sorted(set(REQUIRED_INPUT_FILES) - set(input_hashes))
        if missing_inputs:
            raise RuntimeError(f"Fold {fold} manifest lacks input fingerprints: {missing_inputs}")
        for name, filename in REQUIRED_INPUT_FILES.items():
            input_path = indices_root / filename
            if not input_path.is_file() or _sha256(input_path) != str(input_hashes[name]):
                raise RuntimeError(f"Fold {fold} input changed after manifesting: {name}")
        subjects = {str(subject) for subject in payload.get("by_subject", {})}
        if not subjects:
            raise RuntimeError(f"Fold {fold} metrics do not contain per-subject evidence")
        duplicates = sorted(subject for subject in subjects if subject in seen_subjects)
        if duplicates:
            raise RuntimeError(
                f"Subjects appear in more than one fold: {duplicates}; first fold map is invalid"
            )
        seen_subjects.update({subject: fold for subject in subjects})
        rows.append(
            {
                "fold": fold,
                "metrics": payload[method],
                "different": payload.get("hand_relation", {}).get("different", {}),
                "same": payload.get("hand_relation", {}).get("same", {}),
                "strict": payload.get("strict_no_ignore", {}),
                "by_subject": payload.get("by_subject", {}),
                "subjects": subjects,
                "internal_fusion_gate_passed": internal_fusion_gate_passed,
                "provenance": {
                    "commit": commit,
                    "resolved_config_sha256": config_hash,
                    "selection_run_name": selection_run_name,
                },
                "fingerprints": {
                    key: value
                    for key, value in input_hashes.items()
                    if key
                    in {
                        "quality_report",
                        "quality_expectations",
                        "subject_folds",
                        "subject_folds_manifest",
                        "events",
                    }
                },
            }
        )
    commits = {row["provenance"]["commit"] for row in rows}
    config_hashes = {row["provenance"]["resolved_config_sha256"] for row in rows}
    selection_names = {row["provenance"]["selection_run_name"] for row in rows}
    if len(commits) != 1:
        raise RuntimeError(f"Experiment folds mix Git commits: {sorted(commits)}")
    if len(config_hashes) != 1:
        raise RuntimeError("Experiment folds mix resolved configurations")
    if selection_names != {None} and selection_names != {root.name}:
        raise RuntimeError("Experiment folds mix fusion and non-fusion identities")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    true_positive = sum(float(row["metrics"]["true_positive"]) for row in rows)
    false_positive = sum(float(row["metrics"]["false_positive"]) for row in rows)
    false_negative = sum(float(row["metrics"]["false_negative"]) for row in rows)
    precision = true_positive / max(true_positive + false_positive, 1.0)
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    different_truth = sum(float(row["different"].get("truth_events") or 0) for row in rows)
    different_hits = sum(float(row["different"].get("matched_events") or 0) for row in rows)
    same_truth = sum(float(row["same"].get("truth_events") or 0) for row in rows)
    same_hits = sum(float(row["same"].get("matched_events") or 0) for row in rows)
    strict_true_positive = sum(float(row["strict"].get("true_positive") or 0) for row in rows)
    strict_false_positive = sum(float(row["strict"].get("false_positive") or 0) for row in rows)
    strict_false_negative = sum(float(row["strict"].get("false_negative") or 0) for row in rows)
    strict_precision = strict_true_positive / max(
        strict_true_positive + strict_false_positive, 1.0
    )
    strict_sensitivity = strict_true_positive / max(
        strict_true_positive + strict_false_negative, 1.0
    )
    strict_f1 = 2 * strict_precision * strict_sensitivity / max(
        strict_precision + strict_sensitivity, 1e-12
    )
    matched = max(true_positive, 1.0)
    return {
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "different_sensitivity": different_hits / max(different_truth, 1.0),
        "same_sensitivity": same_hits / max(same_truth, 1.0),
        "strict_no_ignore_f1": strict_f1,
        "weighted_start_mae_seconds": sum(
            float(row["metrics"].get("start_mae_seconds") or 0)
            * float(row["metrics"]["true_positive"])
            for row in rows
        )
        / matched,
        "weighted_end_mae_seconds": sum(
            float(row["metrics"].get("end_mae_seconds") or 0)
            * float(row["metrics"]["true_positive"])
            for row in rows
        )
        / matched,
        "mean_false_positives_per_observed_hour": sum(
            float(row["metrics"]["false_positives_per_observed_hour"]) for row in rows
        )
        / len(rows),
    }


def _f1_from_counts(true_positive: float, false_positive: float, false_negative: float) -> float:
    precision = true_positive / max(true_positive + false_positive, 1.0)
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    return 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)


def _subject_counts(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    output: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        for subject, metrics in row.get("by_subject", {}).items():
            output[str(subject)] = (
                float(metrics.get("true_positive") or 0),
                float(metrics.get("false_positive") or 0),
                float(metrics.get("false_negative") or 0),
            )
    return output


def paired_bootstrap_probability(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    iterations: int = 2000,
    seed: int = 2026,
) -> float:
    baseline = _subject_counts(baseline_rows)
    candidate = _subject_counts(candidate_rows)
    subjects = sorted(set(baseline) & set(candidate))
    if not subjects or set(baseline) != set(candidate):
        return float("nan")
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        baseline_counts = np.asarray([baseline[str(subject)] for subject in sampled]).sum(axis=0)
        candidate_counts = np.asarray([candidate[str(subject)] for subject in sampled]).sum(axis=0)
        baseline_f1 = _f1_from_counts(*baseline_counts)
        candidate_f1 = _f1_from_counts(*candidate_counts)
        wins += candidate_f1 > baseline_f1
    return wins / iterations


def _fingerprints_unchanged(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> bool:
    baseline = [row.get("fingerprints", {}) for row in baseline_rows]
    candidate = [row.get("fingerprints", {}) for row in candidate_rows]
    return bool(baseline[0]) and all(item == baseline[0] for item in baseline + candidate)


def _subject_coverage_matches(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> bool:
    return all(
        set(baseline_rows[index].get("by_subject", {}))
        == set(candidate_rows[index].get("by_subject", {}))
        for index in range(5)
    )


def evaluate_promotion(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline_rows) != 5 or len(candidate_rows) != 5:
        raise ValueError("Promotion requires exactly five baseline and five candidate folds")
    baseline = _aggregate(baseline_rows)
    candidate = _aggregate(candidate_rows)
    nondegraded_folds = sum(
        float(candidate_rows[index]["metrics"]["f1"])
        >= float(baseline_rows[index]["metrics"]["f1"]) - 0.01
        for index in range(5)
    )
    bootstrap_probability = paired_bootstrap_probability(baseline_rows, candidate_rows)
    candidate_uses_fusion = any(
        row.get("provenance", {}).get("selection_run_name") for row in candidate_rows
    )
    checks = {
        "all_baseline_folds_have_recall": all(
            float(row["metrics"]["sensitivity"]) > 0 for row in baseline_rows
        ),
        "all_candidate_folds_have_recall": all(
            float(row["metrics"]["sensitivity"]) > 0 for row in candidate_rows
        ),
        "all_internal_fusion_gates_passed": (
            not candidate_uses_fusion
            or all(row.get("internal_fusion_gate_passed") is True for row in candidate_rows)
        ),
        "micro_f1_at_least_0p490": candidate["f1"] >= 0.490,
        "at_least_three_folds_within_0p01": nondegraded_folds >= 3,
        "different_sensitivity_at_least_0p184": candidate["different_sensitivity"] >= 0.184,
        "same_sensitivity_at_least_0p623": candidate["same_sensitivity"] >= 0.623,
        "strict_no_ignore_f1_at_least_0p434": candidate["strict_no_ignore_f1"] >= 0.434,
        "false_positives_per_hour_at_most_0p02844": (
            candidate["mean_false_positives_per_observed_hour"] <= 0.02844
        ),
        "weighted_start_mae_at_most_256_seconds": (
            candidate["weighted_start_mae_seconds"] <= 256.0
        ),
        "weighted_end_mae_at_most_102_seconds": (
            candidate["weighted_end_mae_seconds"] <= 102.0
        ),
        "paired_bootstrap_probability_at_least_0p80": bootstrap_probability >= 0.80,
        "data_and_fold_fingerprints_unchanged": _fingerprints_unchanged(
            baseline_rows, candidate_rows
        ),
        "per_fold_subject_coverage_unchanged": _subject_coverage_matches(
            baseline_rows, candidate_rows
        ),
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "provenance": {
            "baseline": baseline_rows[0].get("provenance"),
            "candidate": candidate_rows[0].get("provenance"),
            "subjects": len(_subject_counts(baseline_rows)),
        },
        "nondegraded_folds": nondegraded_folds,
        "paired_bootstrap_iterations": 2000,
        "paired_bootstrap_probability": bootstrap_probability,
        "checks": checks,
        "promote": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the five-fold model promotion gate.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline experiment root.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate experiment root.")
    parser.add_argument("--output", type=Path, required=True, help="JSON decision output.")
    args = parser.parse_args()
    baseline_rows = _load_folds(args.baseline)
    candidate_rows = _load_folds(args.candidate)
    output = evaluate_promotion(baseline_rows, candidate_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["promote"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
