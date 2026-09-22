from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.config import load_config, resolve_roots
from bme_eating.dtp_postprocess import _git_identity, event_gate, fit_score_threshold
from bme_eating.dtp_postprocess_21 import frozen_events
from bme_eating.fusion import align_prediction_frames, sha256_file
from bme_eating.fusion_event_ablation import _apply_variant, _rescue_variants, _variant_rank
from bme_eating.fusion_v4 import (
    evaluate_v4_gate,
    paired_subject_bootstrap,
    summarize_event_predictions,
    write_json_atomic,
)
from bme_eating.metrics import (
    evaluate_events,
    interval_iou_matrix,
    match_events,
    partition_evaluation_events,
)
from bme_eating.postprocess import probabilities_to_events
from bme_eating.reproducibility import require_clean_git_worktree


def preserve_matched_ends(
    original: pd.DataFrame,
    rescued: pd.DataFrame,
    *,
    iou_threshold: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"subject_key", "session_id", "start_ms", "end_ms", "score"}
    for name, frame in (("original", original), ("rescued", rescued)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} events are missing {sorted(missing)}")
        if not frame.empty and (frame.end_ms <= frame.start_ms).any():
            raise ValueError(f"{name} events contain invalid intervals")
    result = rescued.copy().reset_index(drop=True)
    decisions: list[dict[str, Any]] = []
    if result.empty:
        return result, pd.DataFrame(columns=["rescued_index", "original_index", "iou", "reason"])
    grouped_original = {
        key: group.sort_values(["start_ms", "end_ms"], kind="stable")
        for key, group in original.reset_index(drop=True).groupby(
            ["subject_key", "session_id"], sort=True
        )
    }
    for key, group in result.groupby(["subject_key", "session_id"], sort=True):
        ordered = group.sort_values(["start_ms", "end_ms"], kind="stable")
        source = grouped_original.get(key)
        candidate_iou = (
            interval_iou_matrix(
                source[["start_ms", "end_ms"]].to_numpy(),
                ordered[["start_ms", "end_ms"]].to_numpy(),
            )
            if source is not None
            else np.zeros((0, len(ordered)))
        )
        assignments = {
            match.prediction_index: match
            for match in match_events(
                source[["start_ms", "end_ms"]].to_numpy()
                if source is not None
                else np.empty((0, 2)),
                ordered[["start_ms", "end_ms"]].to_numpy(),
                iou_threshold=iou_threshold,
            )
        }
        indices = ordered.index.to_list()
        for position, rescued_index in enumerate(indices):
            event = result.loc[rescued_index]
            match = assignments.get(position)
            original_index = None
            iou = None
            reason = "no_iou_match"
            if (
                match is None
                and candidate_iou.size
                and (candidate_iou[:, position] > iou_threshold).any()
            ):
                reason = "assignment_conflict"
            if match is not None:
                original_index = int(source.index[match.truth_index])
                iou = float(match.iou)
                new_end = int(source.loc[original_index, "end_ms"])
                old_end = int(event.end_ms)
                previous_end = (
                    int(result.loc[indices[position - 1], "end_ms"]) if position else None
                )
                next_start = (
                    int(result.loc[indices[position + 1], "start_ms"])
                    if position + 1 < len(indices)
                    else None
                )
                if new_end <= int(event.start_ms):
                    reason = "invalid_duration"
                elif (previous_end is not None and previous_end > int(event.start_ms)) or (
                    next_start is not None and new_end > next_start
                ):
                    reason = "would_overlap"
                elif new_end == old_end:
                    reason = "unchanged"
                else:
                    result.loc[rescued_index, "end_ms"] = new_end
                    reason = "end_copied"
            decisions.append(
                {
                    "subject_key": key[0],
                    "session_id": key[1],
                    "rescued_index": int(rescued_index),
                    "original_index": original_index,
                    "iou": iou,
                    "old_end_ms": int(event.end_ms),
                    "new_end_ms": int(result.loc[rescued_index, "end_ms"]),
                    "reason": reason,
                }
            )
    return result, pd.DataFrame(decisions)


def _assert_replay(actual: dict[str, float], frozen: dict[str, float], label: str) -> None:
    for name in (
        "true_positive",
        "false_positive",
        "false_negative",
        "f1",
        "start_mae_seconds",
        "end_mae_seconds",
        "strict_no_ignore_f1",
    ):
        if not math.isclose(float(actual[name]), float(frozen[name]), rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(
                f"Frozen {label} replay disagrees on {name}: {actual[name]} vs {frozen[name]}"
            )


def _paired_errors(
    truth: pd.DataFrame,
    original: pd.DataFrame,
    rescue: pd.DataFrame,
    preserved: pd.DataFrame,
    iou_threshold: float,
    method: str,
    heldout: int,
) -> pd.DataFrame:
    matches = {}
    for name, events in (
        ("original_v4", original),
        ("additive_rescue", rescue),
        ("preserved_end", preserved),
    ):
        _, paired = evaluate_events(truth, events, iou_threshold, method)
        paired = paired.copy()
        if paired.duplicated(["subject_key", "event_id"]).any():
            raise RuntimeError("Truth event IDs are not unique within subjects")
        matches[name] = paired.set_index(["subject_key", "event_id"])
    rows = []
    for _, event in truth.iterrows():
        key = (event.subject_key, event.event_id)
        original_match = matches["original_v4"]
        rescue_match = matches["additive_rescue"]
        preserved_match = matches["preserved_end"]
        category = (
            "common"
            if key in original_match.index and key in rescue_match.index
            else "new"
            if key in rescue_match.index
            else "lost"
            if key in original_match.index
            else "not_detected"
        )
        if category == "not_detected":
            continue
        row = {
            "heldout_calibration_fold": heldout,
            "subject_key": key[0],
            "event_id": key[1],
            "hand_relation": event.get("hand_relation", "unknown"),
            "category": category,
        }
        for name, frame in (
            ("original_v4", original_match),
            ("additive_rescue", rescue_match),
            ("preserved_end", preserved_match),
        ):
            row[f"{name}_end_error_ms"] = (
                int(frame.at[key, "end_absolute_error_ms"]) if key in frame.index else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def replay_end_preservation(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    original_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    selection: dict[str, Any],
    component: dict[str, Any],
    previous: dict[str, Any],
    fusion_config: dict[str, Any],
    baseline_postprocess: dict[str, Any],
) -> dict[str, Any]:
    end_config = fusion_config["end_preservation"]
    if float(end_config["iou_threshold"]) != float(baseline_postprocess["iou_threshold"]):
        raise ValueError("End alignment must use the frozen baseline IoU threshold")
    baseline, dtp = align_prediction_frames(baseline, dtp)
    keys = ["subject_key", "session_id", "timestamp_ms"]
    original_predictions = original_predictions.sort_values(keys).reset_index(drop=True)
    if not baseline[keys].equals(original_predictions[keys]):
        raise RuntimeError("Frozen v4 timeline differs from source predictions")
    iou = float(baseline_postprocess["iou_threshold"])
    method = str(baseline_postprocess["matching_method"])
    parts: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ("original_v4", "additive_rescue", "preserved_end")
    }
    paired = []
    alignment = []
    partition_rows = []
    rescued_predictions = []
    for scope, frozen in zip(
        selection["meta_folds"],
        previous["comparison"]["nested_additive_rescue"]["partitions"],
        strict=True,
    ):
        heldout = int(scope["heldout_calibration_fold"])
        if heldout != int(frozen["heldout_calibration_fold"]):
            raise RuntimeError("Frozen meta-fold order changed")
        print(f"Replaying heldout {heldout}...", flush=True)
        validation = dtp[dtp.calibration_fold == heldout].reset_index(drop=True)
        training = dtp[dtp.calibration_fold != heldout].reset_index(drop=True)
        validation_subjects = set(validation.subject_key.astype(str))
        if validation_subjects & set(training.subject_key.astype(str)):
            raise RuntimeError("Meta-train and heldout subjects overlap")
        heldout_baseline = baseline[
            baseline.subject_key.astype(str).isin(validation_subjects)
        ].reset_index(drop=True)
        heldout_original = original_predictions[
            original_predictions.subject_key.astype(str).isin(validation_subjects)
        ]
        variant = frozen["selected_variant"]
        threshold = frozen["train_only_score_threshold"]
        gate = None
        if variant["quantile"] is not None:
            generated = probabilities_to_events(training, **component["generator"])
            reference = generated
            accepted = frozen_events(validation, component["generator"], float(threshold))
            gate = event_gate(validation, accepted, reference)
        rescued = _apply_variant(heldout_baseline, validation, scope, fusion_config, variant, gate)
        rescued["calibration_fold"] = heldout
        rescued_predictions.append(rescued)
        truth, ignore = partition_evaluation_events(labels, validation_subjects)
        original_events = probabilities_to_events(
            heldout_original, **_event_parameters(scope["postprocess"])
        )
        rescue_events = probabilities_to_events(rescued, **_event_parameters(scope["postprocess"]))
        original_metrics = summarize_event_predictions(
            heldout_original,
            original_events,
            truth,
            ignore,
            iou_threshold=iou,
            matching_method=method,
        )
        rescue_metrics = summarize_event_predictions(
            rescued, rescue_events, truth, ignore, iou_threshold=iou, matching_method=method
        )
        _assert_replay(original_metrics, scope["candidate_metrics"], f"v4 heldout {heldout}")
        _assert_replay(rescue_metrics, frozen["candidate_metrics"], f"rescue heldout {heldout}")
        preserved_events, decisions = preserve_matched_ends(
            original_events, rescue_events, iou_threshold=iou
        )
        preserved_metrics = summarize_event_predictions(
            rescued, preserved_events, truth, ignore, iou_threshold=iou, matching_method=method
        )
        decisions["heldout_calibration_fold"] = heldout
        alignment.append(decisions)
        paired.append(
            _paired_errors(
                truth, original_events, rescue_events, preserved_events, iou, method, heldout
            )
        )
        for name, events in (
            ("original_v4", original_events),
            ("additive_rescue", rescue_events),
            ("preserved_end", preserved_events),
        ):
            events = events.copy()
            events["heldout_calibration_fold"] = heldout
            parts[name].append(events)
        partition_rows.append(
            {
                "heldout_calibration_fold": heldout,
                "selected_variant": variant,
                "train_only_score_threshold": threshold,
                "original_metrics": original_metrics,
                "rescue_metrics": rescue_metrics,
                "candidate_metrics": preserved_metrics,
                "baseline_metrics": scope["baseline_metrics"],
            }
        )
    predictions = pd.concat(rescued_predictions).sort_values(keys).reset_index(drop=True)
    truth, ignore = partition_evaluation_events(labels, set(dtp.subject_key.astype(str)))
    metrics = {
        name: summarize_event_predictions(
            original_predictions if name == "original_v4" else predictions,
            pd.concat(frames, ignore_index=True),
            truth,
            ignore,
            iou_threshold=iou,
            matching_method=method,
        )
        for name, frames in parts.items()
    }
    _assert_replay(metrics["original_v4"], selection["meta_oof_metrics"], "v4 pooled")
    _assert_replay(
        metrics["additive_rescue"],
        previous["comparison"]["nested_additive_rescue"]["metrics"],
        "rescue pooled",
    )
    before = metrics["additive_rescue"]
    after = metrics["preserved_end"]
    checks = {
        "pooled_end_mae_improves_5_percent": after["end_mae_seconds"]
        <= (1 - float(end_config["minimum_end_mae_improvement"])) * before["end_mae_seconds"],
        "every_partition_end_mae_not_worse": all(
            row["candidate_metrics"]["end_mae_seconds"]
            <= row["rescue_metrics"]["end_mae_seconds"] + 1e-9
            for row in partition_rows
        ),
        "f1_not_lower": after["f1"] >= before["f1"] - 1e-12,
        "strict_f1_not_lower": after["strict_no_ignore_f1"]
        >= before["strict_no_ignore_f1"] - 1e-12,
    }
    module_gate = {"checks": checks, "passed": all(checks.values())}
    fusion_gate = evaluate_v4_gate(
        after,
        selection["meta_oof_baseline_metrics"],
        fusion_config["promotion_gate"],
        partition_rows,
    )
    baseline_events = probabilities_to_events(baseline, **_event_parameters(baseline_postprocess))
    bootstrap = paired_subject_bootstrap(
        predictions,
        pd.concat(parts["preserved_end"], ignore_index=True),
        baseline,
        baseline_events,
        truth,
        ignore,
        iou_threshold=iou,
        matching_method=method,
        replicates=1000,
        seed=2026,
    )
    return {
        "events": {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()},
        "predictions": predictions,
        "metrics": metrics,
        "partitions": partition_rows,
        "paired_errors": pd.concat(paired, ignore_index=True),
        "paired_summary": (
            pd.concat(paired, ignore_index=True)
            .groupby(["heldout_calibration_fold", "hand_relation", "category"], dropna=False)
            .agg(
                count=("event_id", "size"),
                original_end_mae_ms=("original_v4_end_error_ms", "mean"),
                rescue_end_mae_ms=("additive_rescue_end_error_ms", "mean"),
                preserved_end_mae_ms=("preserved_end_end_error_ms", "mean"),
            )
            .reset_index()
        ),
        "alignment": pd.concat(alignment, ignore_index=True),
        "module_gate": module_gate,
        "fusion_gate": fusion_gate,
        "bootstrap_vs_baseline": bootstrap,
    }


def apply_frozen_end_preservation(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    selection: dict[str, Any],
    component_generator: dict[str, Any],
    fusion_config: dict[str, Any],
    *,
    inference_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not selection.get("promotion_allowed") or selection.get("deployment") is None:
        raise RuntimeError("End preservation has not passed both development gates")
    if inference_scope not in {"development_benchmark", "outer"}:
        raise ValueError("Explicit inference scope is required")
    deployment = selection["deployment"]
    if inference_scope == "outer" and set(dtp.subject_key.astype(str)) & set(
        deployment["training_subjects"]
    ):
        raise RuntimeError("Outer subjects overlap the fitted calibration and threshold subjects")
    baseline, dtp = align_prediction_frames(baseline, dtp)
    scope = {"calibrator": deployment["calibrator"], "parameters": deployment["parameters"]}
    original_variant = {"alpha": 0.0}
    original_predictions = _apply_variant(
        baseline, dtp, scope, fusion_config, original_variant, None
    )
    original_events = probabilities_to_events(
        original_predictions, **_event_parameters(deployment["postprocess"])
    )
    gate = None
    variant = deployment["variant"]
    if variant["quantile"] is not None:
        candidates = frozen_events(dtp, component_generator, float(deployment["score_threshold"]))
        reference = pd.DataFrame(deployment["score_reference"], columns=["subject_key", "score"])
        gate = event_gate(dtp, candidates, reference)
    rescued_predictions = _apply_variant(baseline, dtp, scope, fusion_config, variant, gate)
    rescued_events = probabilities_to_events(
        rescued_predictions, **_event_parameters(deployment["postprocess"])
    )
    preserved, _ = preserve_matched_ends(
        original_events, rescued_events, iou_threshold=float(selection["matching_iou_threshold"])
    )
    return original_events, rescued_events, preserved


def fit_deployment(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    labels: pd.DataFrame,
    selection: dict[str, Any],
    component: dict[str, Any],
    fusion_config: dict[str, Any],
    postprocess: dict[str, Any],
) -> dict[str, Any]:
    scope = {
        "calibrator": selection["final_calibrator"],
        "parameters": selection["final_parameters"],
    }
    event_parameters = selection["final_postprocess"]
    subjects = set(dtp.subject_key.astype(str))
    truth, ignore = partition_evaluation_events(labels, subjects)
    generated = probabilities_to_events(dtp, **component["generator"])
    original = _apply_variant(baseline, dtp, scope, fusion_config, {"alpha": 0.0}, None)
    original_events = probabilities_to_events(original, **_event_parameters(event_parameters))
    original_metrics = summarize_event_predictions(
        original,
        original_events,
        truth,
        ignore,
        iou_threshold=float(postprocess["iou_threshold"]),
        matching_method=str(postprocess["matching_method"]),
    )
    choices = []
    variants = list(_rescue_variants(fusion_config))
    for index, variant in enumerate(variants, start=1):
        print(f"Freezing complete-OOF rescue variant {index}/{len(variants)}...", flush=True)
        if variant["quantile"] is None:
            threshold = None
            fused = original
            metrics = original_metrics
        else:
            threshold = fit_score_threshold(generated, float(variant["quantile"]))
            gate = event_gate(dtp, generated[generated.score >= threshold], generated)
            fused = _apply_variant(baseline, dtp, scope, fusion_config, variant, gate)
            events = probabilities_to_events(fused, **_event_parameters(event_parameters))
            metrics = summarize_event_predictions(
                fused,
                events,
                truth,
                ignore,
                iou_threshold=float(postprocess["iou_threshold"]),
                matching_method=str(postprocess["matching_method"]),
            )
        eligible = (
            metrics["f1"] >= original_metrics["f1"]
            and metrics["different_sensitivity"] >= original_metrics["different_sensitivity"]
            and metrics["strict_no_ignore_f1"] >= original_metrics["strict_no_ignore_f1"] - 0.005
        )
        choices.append((eligible, metrics, variant, threshold))
    eligible = [choice for choice in choices if choice[0]]
    selected = max(eligible, key=lambda choice: _variant_rank(choice[1], choice[2]["alpha"]))
    _, metrics, variant, threshold = selected
    return {
        "calibrator": scope["calibrator"],
        "parameters": scope["parameters"],
        "postprocess": event_parameters,
        "variant": variant,
        "score_threshold": threshold,
        "score_reference": generated[["subject_key", "score"]].to_dict("records")
        if threshold is not None
        else [],
        "training_subjects": sorted(subjects),
        "training_metrics": metrics,
        "training_original_metrics": original_metrics,
    }


def _event_parameters(postprocess: dict[str, Any]) -> dict[str, Any]:
    from bme_eating.fusion import _event_parameters as parameters

    return parameters(postprocess)


def _load_frozen_component(directory: Path, name: str) -> dict[str, Any]:
    path = directory / "selected_dtp_postprocess.json"
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["experiment"] != {"name": name, "fold": 0} or manifest["git"]["dirty"]:
        raise RuntimeError("Frozen component provenance is not clean or mismatches the run")
    head = _git_identity()["commit"]
    previous_head = manifest["git"]["commit"]
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", previous_head, head],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("Frozen component commit is not an ancestor of current code")
    for artifact, digest in manifest["artifact_hashes"].items():
        if sha256_file(directory / artifact) != digest:
            raise RuntimeError(f"Frozen component artifact changed: {artifact}")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("protocol_version") != "2.1" or selection.get("selection_run") != name:
        raise RuntimeError("Frozen component protocol identity changed")
    return selection


def run_end_preservation(args: Any) -> Path:
    require_clean_git_worktree()
    if int(args.fold) != 0:
        raise ValueError("Only development fold 0 may be replayed")
    if not re.fullmatch(r"fusion_end_preservation_[A-Za-z0-9_-]{1,64}", args.run_name):
        raise ValueError("Use a new fusion_end_preservation_... run name")
    config = load_config(args.config)
    _, root = resolve_roots(config)

    def fold_dir(run: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run):
            raise ValueError("Invalid run name")
        return root / "experiments" / run / "fold_0"

    output = fold_dir(args.run_name)
    if output.exists():
        raise FileExistsError("Use a fresh run name; frozen outputs remain read-only")
    fusion_dir = fold_dir(args.fusion_run)
    component_dir = fold_dir(args.component_run)
    ablation_dir = fold_dir(args.ablation_run)
    selection_path = fusion_dir / "selected_fusion.json"
    prediction_path = fusion_dir / "validation_predictions.parquet"
    report_path = ablation_dir / "fusion_event_gate_ablation.json"

    def verified(path: Path, manifest: dict[str, Any]) -> str:
        digest = sha256_file(path)
        if manifest["artifact_hashes"].get(path.name) != digest:
            raise RuntimeError(f"Frozen artifact hash mismatch: {path.name}")
        return digest

    fusion_manifest = json.loads((fusion_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ablation_manifest = json.loads((ablation_dir / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = {
        "fusion_selection": verified(selection_path, fusion_manifest),
        "fusion_predictions": verified(prediction_path, fusion_manifest),
        "previous_ablation": verified(report_path, ablation_manifest),
    }
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        selection.get("protocol_version") != 4
        or selection.get("fold") != 0
        or not previous["development_only"]
    ):
        raise ValueError("Expected frozen development v4 selection and ablation")
    if previous["source_hashes"]["fusion"] != hashes["fusion_selection"]:
        raise RuntimeError("Ablation used a different frozen v4 selection")
    component = _load_frozen_component(component_dir, args.component_run)
    if (
        component["source_hashes"] != selection["source_artifact_hashes"]
        or component["baseline_hashes"] != selection["baseline_artifact_hashes"]
        or component["input_hashes"] != selection["input_hashes"]
    ):
        raise RuntimeError("Source prediction, fold or label hashes differ")
    source_dir = fold_dir(component["source_run"])
    baseline_dir = fold_dir(selection["baseline_experiment"])
    paths = {
        "component_selection": component_dir / "selected_dtp_postprocess.json",
        "baseline_predictions": baseline_dir / "validation_predictions.parquet",
        "dtp_predictions": source_dir / "dtp_oof_predictions.parquet",
        "labels": root / "indices" / "events.parquet",
        "baseline_postprocess": baseline_dir / "selected_postprocess.json",
        "config": Path(config["_config_path"]),
        "parent_config": Path(config["_config_path"]).parent / "dtp_fusion_event_ablation.yaml",
        "code": Path(__file__),
    }
    hashes.update({name: sha256_file(path) for name, path in paths.items()})
    for name, digest in (
        ("baseline_predictions", component["baseline_hashes"]["validation_predictions.parquet"]),
        ("dtp_predictions", component["source_hashes"]["dtp_oof_predictions.parquet"]),
        ("labels", component["input_hashes"]["events"]),
        (
            "baseline_postprocess",
            selection["baseline_artifact_hashes"]["selected_postprocess.json"],
        ),
        ("parent_config", previous["source_hashes"]["config"]),
        ("component_selection", previous["source_hashes"]["component"]),
    ):
        if hashes[name] != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    baseline = pd.read_parquet(paths["baseline_predictions"])
    dtp = pd.read_parquet(paths["dtp_predictions"])
    original = pd.read_parquet(prediction_path)
    labels = pd.read_parquet(
        paths["labels"], filters=[("subject_key", "in", sorted(set(dtp.subject_key.astype(str))))]
    )
    postprocess = json.loads(paths["baseline_postprocess"].read_text(encoding="utf-8"))
    result = replay_end_preservation(
        baseline,
        dtp,
        original,
        labels,
        selection,
        component,
        previous,
        config["fusion"],
        postprocess,
    )
    result["promotion_allowed"] = bool(
        result["module_gate"]["passed"] and result["fusion_gate"]["passed"]
    )
    if result["module_gate"]["passed"] and result["fusion_gate"]["passed"]:
        print("Development gates passed; fitting the one full-OOF deployment choice...", flush=True)
        result["deployment"] = fit_deployment(
            baseline, dtp, labels, selection, component, config["fusion"], postprocess
        )
    else:
        result["deployment"] = None
    result["matching_iou_threshold"] = float(postprocess["iou_threshold"])
    if result["deployment"] is not None:
        benchmark_count = 339652
        if len(dtp) < benchmark_count:
            raise RuntimeError("Frozen OOF does not contain the declared benchmark window count")
        aligned_baseline, aligned_dtp = align_prediction_frames(baseline, dtp)
        begin = time.perf_counter()
        original_events, rescued_events, preserved_events = apply_frozen_end_preservation(
            aligned_baseline.iloc[:benchmark_count].copy(),
            aligned_dtp.iloc[:benchmark_count].copy(),
            result,
            component["generator"],
            config["fusion"],
            inference_scope="development_benchmark",
        )
        result["inference_benchmark"] = {
            "windows": benchmark_count,
            "subset": "first_339652_aligned_oof_windows",
            "seconds": time.perf_counter() - begin,
            "original_events": len(original_events),
            "rescued_events": len(rescued_events),
            "preserved_events": len(preserved_events),
            "excludes_model_inference": True,
            "excludes_search_and_labels": True,
        }
    output.mkdir(parents=True)
    artifacts = {}
    for name, frame in {
        **{f"{name}_events": frame for name, frame in result.pop("events").items()},
        "paired_errors": result.pop("paired_errors"),
        "paired_summary": result.pop("paired_summary"),
        "alignment": result.pop("alignment"),
    }.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts[path.name] = sha256_file(path)
    result.pop("predictions")
    result.update(
        {
            "development_only": True,
            "outer_predictions_read": False,
            "promotion_allowed": bool(
                result["module_gate"]["passed"] and result["fusion_gate"]["passed"]
            ),
            "frozen_numeric_parameters": [
                {
                    "heldout_calibration_fold": row["heldout_calibration_fold"],
                    "variant": row["selected_variant"],
                    "train_only_score_threshold": row["train_only_score_threshold"],
                }
                for row in result["partitions"]
            ],
            "input_hashes": hashes,
        }
    )
    write_json_atomic(output / "end_preservation_report.json", result)
    artifacts["end_preservation_report.json"] = sha256_file(output / "end_preservation_report.json")
    write_json_atomic(
        output / "run_manifest.json",
        {
            "experiment": {"name": args.run_name, "fold": 0},
            "git": _git_identity(),
            "input_hashes": hashes,
            "artifact_hashes": artifacts,
            "selection_scope": "development_only_outer_train_meta_oof",
        },
    )
    return output
