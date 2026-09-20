from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import _bootstrap  # noqa: F401

from bme_eating.cli import _evaluate_prediction_file
from bme_eating.config import load_config, resolve_roots
from bme_eating.data.splits import load_subject_folds
from bme_eating.metrics import partition_evaluation_events


def _evaluate_experiment(
    output_root: Path,
    events: pd.DataFrame,
    experiment: str,
    outer_fold: int,
    allowed_subjects: set[str],
) -> dict[str, Any]:
    fold_dir = output_root / "experiments" / experiment / f"fold_{outer_fold}"
    prediction_path = fold_dir / "validation_predictions.parquet"
    postprocess_path = fold_dir / "selected_postprocess.json"
    manifest_path = fold_dir / "run_manifest.json"
    if not prediction_path.exists() or not postprocess_path.exists() or not manifest_path.exists():
        return {"experiment": experiment, "status": "missing"}
    predictions = pd.read_parquet(prediction_path)
    prediction_subjects = set(predictions["subject_key"].astype(str).unique())
    if not prediction_subjects <= allowed_subjects:
        raise RuntimeError(
            f"{experiment} validation predictions contain outer-fold subjects"
        )
    truth, ignore = partition_evaluation_events(events, prediction_subjects)
    postprocess = json.loads(postprocess_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprints = {
        key: value
        for key, value in manifest.get("hashes", {}).items()
        if key
        in {
            "quality_report",
            "quality_expectations",
            "subject_folds",
            "subject_folds_manifest",
            "events",
        }
    }
    _, metrics, _ = _evaluate_prediction_file(predictions, truth, postprocess, ignore)
    primary = metrics[str(metrics["primary_method"])]
    different = metrics.get("hand_relation", {}).get("different", {})
    return {
        "experiment": experiment,
        "status": "complete",
        "subjects": len(prediction_subjects),
        "f1": float(primary["f1"]),
        "different_sensitivity": float(different.get("sensitivity") or 0.0),
        "false_positives_per_observed_hour": float(
            primary["false_positives_per_observed_hour"]
        ),
        "postprocess_at_search_boundary": bool(
            postprocess.get("threshold_at_search_boundary", False)
        ),
        "data_and_fold_fingerprints": fingerprints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen dyadic candidates using fold-0 outer-training OOF predictions only."
    )
    parser.add_argument("--config", default="configs/baseline_dyadic_lite.yaml")
    parser.add_argument("--outer-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--baseline", default="baseline_boundary")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["baseline_dyadic_lite"],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    subject_folds = load_subject_folds(output_root / "indices" / "subject_folds.json")
    allowed_subjects = {
        str(subject) for subject, fold in subject_folds.items() if fold != args.outer_fold
    }
    baseline = _evaluate_experiment(
        output_root, events, args.baseline, args.outer_fold, allowed_subjects
    )
    if baseline["status"] != "complete":
        raise FileNotFoundError("The local-feature OOF baseline is required before dyadic screening")
    if baseline["postprocess_at_search_boundary"]:
        raise RuntimeError("The local-feature OOF baseline has a boundary-selected parameter")
    if not baseline["data_and_fold_fingerprints"]:
        raise RuntimeError("The local-feature OOF baseline has no data/fold fingerprints")
    candidates = [
        _evaluate_experiment(output_root, events, name, args.outer_fold, allowed_subjects)
        for name in args.candidates
    ]
    for candidate in candidates:
        if candidate["status"] != "complete":
            continue
        candidate["checks"] = {
            "f1_gain_at_least_0p015": candidate["f1"] - baseline["f1"] >= 0.015,
            "different_sensitivity_gain_at_least_0p03": (
                candidate["different_sensitivity"]
                - baseline["different_sensitivity"]
                >= 0.03
            ),
            "false_positives_per_hour_at_most_1p2x": (
                candidate["false_positives_per_observed_hour"]
                <= baseline["false_positives_per_observed_hour"] * 1.2
            ),
            "postprocess_parameters_are_interior": not candidate[
                "postprocess_at_search_boundary"
            ],
            "data_and_fold_fingerprints_unchanged": candidate[
                "data_and_fold_fingerprints"
            ]
            == baseline["data_and_fold_fingerprints"],
        }
        candidate["promote"] = all(candidate["checks"].values())
    result = {
        "version": 1,
        "selection_scope": "outer_training_oof_only",
        "outer_fold": args.outer_fold,
        "baseline": baseline,
        "candidates": candidates,
        "run_full_five_fold": bool(
            candidates
            and candidates[0].get("experiment") == "baseline_dyadic_lite"
            and candidates[0].get("promote", False)
        ),
    }
    output = args.output or output_root / "experiments" / "dyadic_oof_screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["run_full_five_fold"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
