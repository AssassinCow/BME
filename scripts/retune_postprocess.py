from __future__ import annotations

import argparse
import json
import shutil

import pandas as pd

import _bootstrap  # noqa: F401

from bme_eating.cli import _evaluate_prediction_file, _tune_and_save_postprocess
from bme_eating.config import load_config, resolve_roots
from bme_eating.data.quality import validate_quality_gate
from bme_eating.metrics import partition_evaluation_events
from bme_eating.reproducibility import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retune event postprocessing from frozen OOF predictions without retraining."
    )
    parser.add_argument("--config", default="configs/baseline_fastslow.yaml")
    parser.add_argument("--source-experiment", default="baseline")
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    args = parser.parse_args()

    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    validate_quality_gate(output_root)
    target_name = str(config.get("experiment", {}).get("name", "baseline_fastslow"))
    source = output_root / "experiments" / args.source_experiment / f"fold_{args.fold}"
    target = output_root / "experiments" / target_name / f"fold_{args.fold}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "model.json", target / "model.json")
    write_run_manifest(target, config, output_root)
    validation = pd.read_parquet(source / "validation_predictions.parquet")
    if "calibration_fold" not in validation.columns:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        inner_partitions = {
            str(subject): int(fold)
            for subject, fold in metadata.get("inner_partitions", {}).items()
        }
        validation["calibration_fold"] = validation["subject_key"].astype(str).map(
            inner_partitions
        )
        if validation["calibration_fold"].isna().any():
            raise RuntimeError("Cannot recover calibration_fold for every OOF subject")
        validation["calibration_fold"] = validation["calibration_fold"].astype(int)
    events = pd.read_parquet(output_root / "indices" / "events.parquet")
    validation_subjects = set(validation["subject_key"].astype(str).unique())
    validation_truth, validation_ignore = partition_evaluation_events(
        events, validation_subjects
    )
    selected = _tune_and_save_postprocess(
        validation, validation_truth, config, target, validation_ignore
    )
    test = pd.read_parquet(source / "test_predictions.parquet")
    test_subjects = set(test["subject_key"].astype(str).unique())
    if validation_subjects & test_subjects:
        raise RuntimeError("Source OOF predictions overlap held-out test subjects")
    truth, ignore = partition_evaluation_events(events, test_subjects)
    predicted_events, metrics, failures = _evaluate_prediction_file(
        test, truth, selected, ignore
    )
    validation.to_parquet(target / "validation_predictions.parquet", index=False)
    test.to_parquet(target / "test_predictions.parquet", index=False)
    predicted_events.to_csv(target / "test_events.csv", index=False)
    failures.to_csv(target / "test_failure_cases.csv", index=False)
    (target / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "metadata.json").write_text(
        json.dumps(
            {
                "outer_fold": args.fold,
                "source_experiment": args.source_experiment,
                "postprocess_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_run_manifest(target, config, output_root)
    print(target)


if __name__ == "__main__":
    main()
