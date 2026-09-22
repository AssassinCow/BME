from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from bme_eating.config import resolve_roots
from bme_eating.dtp_postprocess import (
    _baseline,
    _event_diagnostics,
    _git_identity,
    _metrics,
    _run_name,
    _save_json,
    _source,
    _training_inputs,
    fit_score_threshold,
)
from bme_eating.fusion import evaluate_fusion_predictions, sha256_file
from bme_eating.fusion_v4 import paired_subject_bootstrap
from bme_eating.metrics import evaluate_events, partition_evaluation_events
from bme_eating.postprocess import probabilities_to_events
from bme_eating.reproducibility import require_clean_git_worktree


def frozen_events(
    predictions: pd.DataFrame, generator: dict[str, Any], score_threshold: float
) -> pd.DataFrame:
    if not 0 <= score_threshold <= 1:
        raise ValueError("Frozen event threshold must be in [0, 1]")
    events = probabilities_to_events(predictions, **generator)
    return events[events.score >= score_threshold].reset_index(drop=True)


def _partition_metrics(
    predictions: pd.DataFrame,
    predicted: pd.DataFrame,
    events: pd.DataFrame,
    partitions: list[int],
    iou: float,
    method: str,
) -> list[dict[str, float]]:
    metrics = []
    for partition in partitions:
        subset = predictions[predictions.calibration_fold == partition]
        subjects = set(subset.subject_key.astype(str))
        truth, ignore = partition_evaluation_events(events, subjects)
        metrics.append(
            _metrics(
                subset,
                predicted[predicted.subject_key.astype(str).isin(subjects)],
                truth,
                ignore,
                iou,
                method,
            )
        )
    return metrics


def _rank(rows: list[dict[str, float]], quantile: float) -> tuple[float, ...]:
    return (
        min(row["f1"] for row in rows),
        statistics.mean(row["f1"] for row in rows),
        statistics.mean(row["strict_no_ignore_f1"] for row in rows),
        -statistics.mean(row["false_positives_per_observed_hour"] for row in rows),
        -quantile,
    )


def crossfit_legacy_generator(
    oof: pd.DataFrame,
    events: pd.DataFrame,
    generator: dict[str, Any],
    quantiles: list[float],
    iou: float,
    method: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[float, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    assignments = oof[["subject_key", "calibration_fold"]].drop_duplicates()
    if assignments.subject_key.duplicated().any() or sorted(
        assignments.calibration_fold.unique().tolist()
    ) != [0, 1, 2]:
        raise ValueError("Three subject-disjoint calibration folds are required")
    if (
        not quantiles
        or len(quantiles) != len(set(quantiles))
        or any(not 0 <= quantile <= 1 for quantile in quantiles)
    ):
        raise ValueError("Score quantiles must be unique and in [0, 1]")
    if generator.get("detector_mode") != "dual_ema":
        raise ValueError("Protocol 2.1 requires the frozen legacy dual EMA generator")

    generated = probabilities_to_events(oof, **generator)
    if generated.empty:
        raise ValueError("Legacy generator produced no training events")
    scopes: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for heldout in range(3):
        train = oof[oof.calibration_fold != heldout]
        validation = oof[oof.calibration_fold == heldout]
        train_subjects = set(train.subject_key.astype(str))
        heldout_subjects = set(validation.subject_key.astype(str))
        if not train_subjects or not heldout_subjects or train_subjects & heldout_subjects:
            raise RuntimeError("Meta training and heldout subjects overlap or are empty")
        train_events = generated[generated.subject_key.astype(str).isin(train_subjects)]
        validation_events = generated[generated.subject_key.astype(str).isin(heldout_subjects)]
        if train_events.empty:
            raise ValueError("Meta training has no candidate event scores")
        train_partitions = [number for number in range(3) if number != heldout]
        options = []
        for quantile in quantiles:
            threshold = fit_score_threshold(train_events, quantile)
            retained_train = train_events[train_events.score >= threshold]
            inner = _partition_metrics(train, retained_train, events, train_partitions, iou, method)
            options.append({"quantile": quantile, "threshold": threshold, "train_folds": inner})
        winner = max(options, key=lambda option: _rank(option["train_folds"], option["quantile"]))
        scopes.append(
            {
                "heldout_fold": heldout,
                "train_subjects": sorted(train_subjects),
                "validation_subjects": sorted(heldout_subjects),
                "selected_quantile": winner["quantile"],
                "options": options,
                "candidate_events": validation_events,
            }
        )
        for option in options:
            filtered = validation_events[validation_events.score >= option["threshold"]]
            truth, ignore = partition_evaluation_events(events, heldout_subjects)
            measured = _metrics(validation, filtered, truth, ignore, iou, method)
            trials.append(
                {
                    "heldout_fold": heldout,
                    "quantile": option["quantile"],
                    "train_only_threshold": option["threshold"],
                    "train_min_f1": min(row["f1"] for row in option["train_folds"]),
                    "heldout_f1": measured["f1"],
                    "heldout_fp_per_hour": measured["false_positives_per_observed_hour"],
                    "selected_within_train": option is winner,
                }
            )

    selected_parts: list[pd.DataFrame] = []
    fixed_parts: dict[float, list[pd.DataFrame]] = {quantile: [] for quantile in quantiles}
    for scope in scopes:
        validation_events = scope.pop("candidate_events")
        options = {option["quantile"]: option for option in scope["options"]}
        selected = options[scope["selected_quantile"]]
        selected_parts.append(validation_events[validation_events.score >= selected["threshold"]])
        scope["selected_threshold"] = selected["threshold"]
        scope["candidate_thresholds"] = {
            str(quantile): options[quantile]["threshold"] for quantile in quantiles
        }
        for quantile in quantiles:
            fixed_parts[quantile].append(
                validation_events[validation_events.score >= options[quantile]["threshold"]]
            )
    selected_events = pd.concat(selected_parts, ignore_index=True)
    fixed_candidates = {
        quantile: pd.concat(parts, ignore_index=True) for quantile, parts in fixed_parts.items()
    }
    record = {
        "protocol_version": "2.1",
        "generator": generator,
        "scopes": scopes,
        "score_semantics": "mean causal EMA before merge; legacy merge uses maximum child score",
        "limitation": "Legacy generator was fixed using an earlier outer-train OOF diagnostic; its selection is not independently cross-fitted. Final threshold is refit on all outer-train OOF. No outer holdout was read.",
    }
    return record, selected_events, fixed_candidates, pd.DataFrame(trials), generated


def _manifest(directory: Path, config: dict[str, Any], provenance: dict[str, Any]) -> None:
    _save_json(
        directory / "run_manifest.json",
        {
            "protocol_version": "2.1",
            "experiment": {"name": directory.parent.name, "fold": int(directory.name[5:])},
            "git": _git_identity(),
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "provenance": provenance,
            "random_seeds": {
                "split": config["data"]["split_seed"],
                "bootstrap": config["dtp_postprocess"]["bootstrap_seed"],
            },
            "artifact_hashes": {
                path.name: sha256_file(path)
                for path in sorted(directory.iterdir())
                if path.is_file() and path.name != "run_manifest.json"
            },
        },
    )


def _comparison(output_root: Path, run: str, source_hashes: dict[str, str]) -> tuple[Path, str]:
    directory = output_root / "experiments" / _run_name(run) / "fold_0"
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    path = directory / "selected_dtp_postprocess.json"
    selected = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != {"name": run, "fold": 0}:
        raise RuntimeError("Frozen v2 comparison identity changed")
    if manifest.get("git", {}).get("dirty") is not False:
        raise RuntimeError("Frozen v2 comparison came from a dirty worktree")
    if manifest["artifact_hashes"].get(path.name) != sha256_file(path):
        raise RuntimeError("Frozen v2 selection hash changed")
    if selected.get("protocol_version") != 2 or selected.get("source_hashes") != source_hashes:
        raise RuntimeError("Frozen v2 comparison source differs from this prediction source")
    events = directory / "meta_robust_f1_events.parquet"
    if manifest["artifact_hashes"].get(events.name) != sha256_file(events):
        raise RuntimeError("Frozen v2 meta events hash changed")
    return events, sha256_file(path)


def complementarity(
    truth: pd.DataFrame,
    ignore: pd.DataFrame,
    baseline_events: pd.DataFrame,
    candidate_events: pd.DataFrame,
    iou: float,
    method: str,
) -> dict[str, float]:
    baseline_metrics, baseline_matches = evaluate_events(
        truth, baseline_events, iou, method, ignore=ignore
    )
    candidate_metrics, candidate_matches = evaluate_events(
        truth, candidate_events, iou, method, ignore=ignore
    )

    def matched_truth(matches: pd.DataFrame) -> set[tuple[str, int, int]]:
        return set(
            zip(
                matches.get("subject_key", []),
                matches.get("truth_start_ms", []),
                matches.get("truth_end_ms", []),
                strict=True,
            )
        )

    baseline_hits = matched_truth(baseline_matches)
    candidate_hits = matched_truth(candidate_matches)
    rescued = candidate_hits - baseline_hits
    relations = {
        (str(row.subject_key), int(row.start_ms), int(row.end_ms)): str(row.hand_relation)
        for row in truth.itertuples(index=False)
    }
    false_positives = float(candidate_metrics["false_positive"])
    return {
        "common_true_positives": float(len(baseline_hits & candidate_hits)),
        "dtp_unique_true_positives": float(len(rescued)),
        "xgboost_unique_true_positives": float(len(baseline_hits - candidate_hits)),
        "union_true_positives_upper_bound": float(len(baseline_hits | candidate_hits)),
        "rescued_xgboost_false_negatives": float(len(rescued)),
        "rescued_different_hand_events": float(
            sum(relations[key] == "different" for key in rescued)
        ),
        "standalone_dtp_false_positives": false_positives,
        "standalone_false_positives_per_rescue": false_positives / max(1, len(rescued)),
        "xgboost_false_negatives": float(baseline_metrics["false_negative"]),
    }


def _component_screen(
    diagnostics: dict[str, float],
    partition_diagnostics: list[dict[str, float]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "unique_rescues": diagnostics["rescued_xgboost_false_negatives"]
        >= settings["minimum_unique_rescues"],
        "different_hand_rescues": diagnostics["rescued_different_hand_events"]
        >= settings["minimum_different_hand_rescues"],
        "false_positives_per_rescue": diagnostics["standalone_false_positives_per_rescue"]
        <= settings["maximum_standalone_false_positives_per_rescue"],
        "each_partition_rescues": all(
            row["rescued_xgboost_false_negatives"] >= settings["minimum_partition_rescues"]
            for row in partition_diagnostics
        ),
    }
    return {
        "passed_for_fusion_ablation": all(checks.values()),
        "checks": checks,
        "not_a_standalone_or_fusion_promotion_gate": True,
    }


def tune_legacy_crossfit(args: Any, config: dict[str, Any]) -> Path:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    require_clean_git_worktree()
    _, root = resolve_roots(config)
    if int(args.fold) != 0:
        raise ValueError("Protocol 2.1 selection is restricted to development fold 0")
    settings = config["dtp_postprocess"]
    name = _run_name(args.run_name)
    directory = root / "experiments" / name / "fold_0"
    if directory.exists():
        raise FileExistsError("Protocol 2.1 requires a new selection run; no overwrites")
    source_dir, source_hashes = _source(root, args.source_run, 0)
    baseline_dir, baseline_info = _baseline(root, config, 0)
    comparison_path, comparison_hash = _comparison(root, settings["comparison_run"], source_hashes)
    oof, baseline, _, events = _training_inputs(root, source_dir, baseline_dir, 0)
    postprocess = json.loads((baseline_dir / "selected_postprocess.json").read_text())
    iou = float(postprocess["iou_threshold"])
    method = str(postprocess["matching_method"])
    print("[1/3] Generating legacy EMA candidates once; cross-fitting scores...", flush=True)
    record, selected_events, fixed_candidates, trials, generated = crossfit_legacy_generator(
        oof, events, settings["legacy_generator"], settings["legacy_score_quantiles"], iou, method
    )
    subjects = set(oof.subject_key.astype(str))
    truth, ignore = partition_evaluation_events(events, subjects)
    xgb_metrics, xgb_events = evaluate_fusion_predictions(baseline, truth, ignore, postprocess)
    v2_events = pd.read_parquet(comparison_path)
    v2_metrics = _metrics(oof, v2_events, truth, ignore, iou, method)
    selected_metrics = _metrics(oof, selected_events, truth, ignore, iou, method)
    deployment_trials = []
    for quantile, candidate_events in fixed_candidates.items():
        metrics = _metrics(oof, candidate_events, truth, ignore, iou, method)
        folds = _partition_metrics(oof, candidate_events, events, [0, 1, 2], iou, method)
        diagnostic = complementarity(truth, ignore, xgb_events, candidate_events, iou, method)
        partition_diagnostics = []
        for partition in range(3):
            partition_subjects = set(
                oof.loc[oof.calibration_fold == partition, "subject_key"].astype(str)
            )
            partition_truth, partition_ignore = partition_evaluation_events(
                events, partition_subjects
            )
            partition_diagnostics.append(
                complementarity(
                    partition_truth,
                    partition_ignore,
                    xgb_events[xgb_events.subject_key.astype(str).isin(partition_subjects)],
                    candidate_events[
                        candidate_events.subject_key.astype(str).isin(partition_subjects)
                    ],
                    iou,
                    method,
                )
            )
        screen = _component_screen(diagnostic, partition_diagnostics, settings)
        deployment_trials.append(
            {
                "quantile": quantile,
                "metrics": metrics,
                "fold_metrics": folds,
                "complementarity": diagnostic,
                "partition_complementarity": partition_diagnostics,
                "component_screen": screen,
            }
        )
    passing = [
        trial
        for trial in deployment_trials
        if trial["component_screen"]["passed_for_fusion_ablation"]
    ]
    if not passing:
        raise RuntimeError("No DTP component meets the predeclared fusion-ablation screen")
    rescue = max(
        passing,
        key=lambda trial: (
            trial["complementarity"]["rescued_different_hand_events"],
            trial["complementarity"]["rescued_xgboost_false_negatives"],
            -trial["complementarity"]["standalone_dtp_false_positives"],
        ),
    )
    balanced = min(
        passing,
        key=lambda trial: (
            trial["complementarity"]["standalone_false_positives_per_rescue"],
            -trial["complementarity"]["rescued_different_hand_events"],
            trial["complementarity"]["standalone_dtp_false_positives"],
        ),
    )
    winner = rescue
    chosen_quantile = float(winner["quantile"])
    fixed_events = fixed_candidates[chosen_quantile]
    fixed_metrics = winner["metrics"]
    fixed_folds = winner["fold_metrics"]
    record["final_quantile"] = chosen_quantile
    record["final_score_threshold"] = fit_score_threshold(generated, chosen_quantile)
    record["deployment_trials"] = deployment_trials
    record["fusion_gate_candidates"] = {
        name: {
            "quantile": float(trial["quantile"]),
            "full_oof_score_threshold": fit_score_threshold(generated, float(trial["quantile"])),
            "complementarity": trial["complementarity"],
            "partition_complementarity": trial["partition_complementarity"],
            "component_screen": trial["component_screen"],
        }
        for name, trial in (("rescue", rescue), ("balanced", balanced))
    }
    bootstrap = paired_subject_bootstrap(
        oof,
        fixed_events,
        oof,
        v2_events,
        truth,
        ignore,
        iou_threshold=iou,
        matching_method=method,
        replicates=int(settings["bootstrap_replicates"]),
        seed=int(settings["bootstrap_seed"]),
    )
    record.update(
        {
            "selection_run": name,
            "source_run": args.source_run,
            "comparison_run": settings["comparison_run"],
            "source_hashes": source_hashes,
            "comparison_selection_sha256": comparison_hash,
            "baseline_hashes": baseline_info["artifact_hashes"],
            "input_hashes": baseline_info["input_hashes"],
            "metrics": {
                "selected_per_fold": selected_metrics,
                "frozen_quantile": fixed_metrics,
                "v2": v2_metrics,
                "xgboost": xgb_metrics,
            },
            "frozen_fold_metrics": fixed_folds,
            "component_screen": winner["component_screen"],
            "fusion_promotion_status": "not_evaluated",
            "bootstrap_vs_v2": bootstrap,
            "development_only": True,
        }
    )
    print("[2/3] Writing meta-OOF evidence and the immutable choice...", flush=True)
    directory.mkdir(parents=True)
    trials.to_csv(directory / "threshold_trials.csv", index=False)
    selected_events.to_parquet(directory / "meta_selected_events.parquet", index=False)
    fixed_events.to_parquet(directory / "meta_frozen_quantile_events.parquet", index=False)
    for candidate_name, candidate in record["fusion_gate_candidates"].items():
        fixed_candidates[candidate["quantile"]].to_parquet(
            directory / f"meta_{candidate_name}_events.parquet", index=False
        )
    fixed_events.to_csv(directory / "meta_oof_events.csv", index=False)
    failures, per_subject, by_hand = _event_diagnostics(
        oof, fixed_events, truth, ignore, iou, method
    )
    failures.to_csv(directory / "meta_oof_failure_cases.csv", index=False)
    per_subject.to_csv(directory / "meta_oof_per_subject_metrics.csv", index=False)
    _save_json(directory / "meta_oof_hand_relation_metrics.json", by_hand)
    _save_json(
        directory / "meta_oof_metrics.json",
        {
            "candidate": fixed_metrics,
            "v2": v2_metrics,
            "xgboost": xgb_metrics,
            "component_screen": winner["component_screen"],
            "fusion_promotion_status": "not_evaluated",
        },
    )
    _save_json(directory / "selected_dtp_postprocess.json", record)
    _manifest(
        directory,
        config,
        {
            "source_hashes": source_hashes,
            "comparison_selection_sha256": comparison_hash,
            "input_hashes": baseline_info["input_hashes"],
        },
    )
    print(
        "[3/3] Finished without reading outer holdout; search is never run in inference.",
        flush=True,
    )
    print(
        json.dumps(
            {
                "component_candidates": {
                    name: item["quantile"]
                    for name, item in record["fusion_gate_candidates"].items()
                },
                "fusion_promotion_status": "not_evaluated",
                "output": str(directory),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return directory


def _load_selection(directory: Path, name: str) -> dict[str, Any]:
    path = directory / "selected_dtp_postprocess.json"
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(path.read_text(encoding="utf-8"))
    if manifest["experiment"] != {"name": name, "fold": 0} or manifest["git"] != _git_identity():
        raise RuntimeError("Frozen v2.1 selection Git or identity changed")
    for artifact, expected in manifest["artifact_hashes"].items():
        if sha256_file(directory / artifact) != expected:
            raise RuntimeError(f"Frozen v2.1 artifact hash changed: {artifact}")
    if manifest["artifact_hashes"].get(path.name) != sha256_file(path):
        raise RuntimeError("Frozen v2.1 selection hash changed")
    if selection.get("protocol_version") != "2.1" or selection.get("selection_run") != name:
        raise RuntimeError("Frozen v2.1 protocol identity changed")
    return selection


def evaluate_legacy_frozen(args: Any, config: dict[str, Any]) -> Path:
    raise RuntimeError(
        "Protocol 2.1 is a DTP component diagnostic, not a standalone promotion. "
        "Run fusion meta-crossfit ablations and freeze the winning fusion before reading outer data."
    )
