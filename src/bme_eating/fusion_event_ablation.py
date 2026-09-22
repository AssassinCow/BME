from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bme_eating.config import load_config, resolve_roots
from bme_eating.dtp_postprocess import _git_identity, event_gate, fit_score_threshold
from bme_eating.dtp_postprocess_21 import _load_selection, frozen_events
from bme_eating.fusion import align_prediction_frames, evaluate_fusion_predictions, sha256_file
from bme_eating.fusion_v4 import (
    PlattCalibrator,
    evaluate_v4_gate,
    fuse_gated_prediction_frames,
    summarize_event_predictions,
)
from bme_eating.metrics import partition_evaluation_events
from bme_eating.postprocess import probabilities_to_events
from bme_eating.reproducibility import require_clean_git_worktree


def compare_event_gates(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    events: pd.DataFrame,
    v4: dict[str, Any],
    component: dict[str, Any],
    fusion_config: dict[str, Any],
    baseline_postprocess: dict[str, Any],
    original_predictions: pd.DataFrame,
) -> dict[str, Any]:
    baseline, dtp = align_prediction_frames(baseline, dtp)
    expected = baseline[["subject_key", "session_id", "timestamp_ms"]]
    if not original_predictions[["subject_key", "session_id", "timestamp_ms"]].equals(expected):
        original_predictions = original_predictions.sort_values(
            ["subject_key", "session_id", "timestamp_ms"]
        ).reset_index(drop=True)
        if not original_predictions[["subject_key", "session_id", "timestamp_ms"]].equals(expected):
            raise ValueError("Frozen v4 meta predictions differ from the DTP/baseline timeline")
    generated = probabilities_to_events(dtp, **component["generator"])
    if generated.empty:
        raise ValueError("DTP generator produced no component events")
    all_subjects = set(dtp.subject_key.astype(str))
    truth, ignore = partition_evaluation_events(events, all_subjects)
    iou = float(baseline_postprocess["iou_threshold"])
    method = str(baseline_postprocess["matching_method"])
    comparisons: dict[str, Any] = {}
    original_reconstructed = []
    for name in ("rescue", "balanced"):
        print(f"Comparing multiplicative gate: {name}...", flush=True)
        choice = component["fusion_gate_candidates"][name]
        if not choice["component_screen"]["passed_for_fusion_ablation"]:
            raise RuntimeError(f"DTP {name} gate did not pass its component screen")
        heldout_predictions = []
        heldout_events = []
        partition_rows = []
        for scope in v4["meta_folds"]:
            heldout = int(scope["heldout_calibration_fold"])
            validation = dtp[dtp.calibration_fold == heldout]
            training = dtp[dtp.calibration_fold != heldout]
            train_subjects = set(training.subject_key.astype(str))
            validation_subjects = set(validation.subject_key.astype(str))
            if train_subjects & validation_subjects:
                raise RuntimeError("Meta training and heldout subjects overlap")
            train_events = generated[generated.subject_key.astype(str).isin(train_subjects)]
            threshold = fit_score_threshold(train_events, float(choice["quantile"]))
            train_reference = train_events[train_events.score >= threshold]
            candidate_events = frozen_events(validation, component["generator"], float(threshold))
            gate = event_gate(validation, candidate_events, train_reference)
            heldout_baseline = (
                baseline[baseline.calibration_fold == heldout]
                if ("calibration_fold" in baseline)
                else baseline[baseline.subject_key.astype(str).isin(validation_subjects)]
            )
            fused = fuse_gated_prediction_frames(
                heldout_baseline,
                validation,
                PlattCalibrator.from_dict(scope["calibrator"]),
                scope["parameters"],
                residual_clip=float(fusion_config["residual_clip"]),
                epsilon=float(fusion_config["probability_epsilon"]),
                gate_ema_half_life_seconds=float(
                    fusion_config["gated_residual"]["gate_ema_half_life_seconds"]
                ),
                event_gate=gate,
            )
            if name == "rescue":
                original_reconstructed.append(
                    fuse_gated_prediction_frames(
                        heldout_baseline,
                        validation,
                        PlattCalibrator.from_dict(scope["calibrator"]),
                        scope["parameters"],
                        residual_clip=float(fusion_config["residual_clip"]),
                        epsilon=float(fusion_config["probability_epsilon"]),
                        gate_ema_half_life_seconds=float(
                            fusion_config["gated_residual"]["gate_ema_half_life_seconds"]
                        ),
                    )
                )
            fused["calibration_fold"] = heldout
            partition_truth, partition_ignore = partition_evaluation_events(
                events, validation_subjects
            )
            metrics, predicted = evaluate_fusion_predictions(
                fused, partition_truth, partition_ignore, scope["postprocess"]
            )
            heldout_predictions.append(fused)
            heldout_events.append(predicted)
            partition_rows.append(
                {
                    "heldout_fold": heldout,
                    "threshold_from_train_only": threshold,
                    "training_subjects": len(train_subjects),
                    "validation_subjects": len(validation_subjects),
                    "candidate_metrics": metrics,
                    "baseline_metrics": scope["baseline_metrics"],
                }
            )
        predictions = (
            pd.concat(heldout_predictions, ignore_index=True)
            .sort_values(["subject_key", "session_id", "timestamp_ms"])
            .reset_index(drop=True)
        )
        predicted = pd.concat(heldout_events, ignore_index=True)
        metrics = summarize_event_predictions(
            predictions,
            predicted,
            truth,
            ignore,
            iou_threshold=iou,
            matching_method=method,
        )
        gate = evaluate_v4_gate(
            metrics,
            v4["meta_oof_baseline_metrics"],
            fusion_config["promotion_gate"],
            partition_rows,
        )
        comparisons[name] = {
            "quantile": choice["quantile"],
            "metrics": metrics,
            "fusion_gate": gate,
            "delta_f1_vs_original_v4": metrics["f1"] - v4["meta_oof_metrics"]["f1"],
            "delta_different_sensitivity_vs_original_v4": metrics["different_sensitivity"]
            - v4["meta_oof_metrics"]["different_sensitivity"],
            "partitions": partition_rows,
        }
    comparisons["original_v4"] = {
        "metrics": v4["meta_oof_metrics"],
        "fusion_gate": v4["meta_oof_gate"],
    }
    reconstructed = (
        pd.concat(original_reconstructed, ignore_index=True)
        .sort_values(["subject_key", "session_id", "timestamp_ms"])
        .reset_index(drop=True)
    )
    if not np.allclose(
        reconstructed.state_probability.to_numpy(dtype=float),
        original_predictions.state_probability.to_numpy(dtype=float),
        rtol=0,
        atol=1e-6,
    ):
        raise RuntimeError("Unmodified v4 predictions could not be reproduced")
    comparisons["nested_additive_rescue"] = nested_additive_rescue(
        baseline,
        dtp,
        events,
        v4,
        component,
        fusion_config,
        baseline_postprocess,
    )
    return comparisons


def _variant_rank(metrics: dict[str, float], alpha: float) -> tuple[float, ...]:
    return (
        float(metrics["f1"]),
        float(metrics["different_sensitivity"]),
        float(metrics["strict_no_ignore_f1"]),
        -float(metrics["false_positives_per_observed_hour"]),
        -float(alpha),
    )


def _rescue_variants(fusion_config: dict[str, Any]):
    yield {"name": "original_v4", "quantile": None, "alpha": 0.0}
    rescue = fusion_config["event_rescue"]
    for quantile in rescue["quantile_candidates"]:
        for alpha in rescue["alpha_candidates"]:
            yield {
                "name": f"q_{float(quantile):g}_alpha_{float(alpha):g}",
                "quantile": float(quantile),
                "alpha": float(alpha),
            }


def _apply_variant(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    scope: dict[str, Any],
    fusion_config: dict[str, Any],
    variant: dict[str, Any],
    gate: pd.DataFrame | None,
) -> pd.DataFrame:
    return fuse_gated_prediction_frames(
        baseline,
        dtp,
        PlattCalibrator.from_dict(scope["calibrator"]),
        scope["parameters"],
        residual_clip=float(fusion_config["residual_clip"]),
        epsilon=float(fusion_config["probability_epsilon"]),
        gate_ema_half_life_seconds=float(
            fusion_config["gated_residual"]["gate_ema_half_life_seconds"]
        ),
        event_gate=gate,
        restrict_positive_with_event_gate=False,
        event_rescue_alpha=float(variant["alpha"]),
        event_rescue_baseline_max=float(fusion_config["event_rescue"]["baseline_max"]),
    )


def nested_additive_rescue(
    baseline: pd.DataFrame,
    dtp: pd.DataFrame,
    events: pd.DataFrame,
    v4: dict[str, Any],
    component: dict[str, Any],
    fusion_config: dict[str, Any],
    baseline_postprocess: dict[str, Any],
) -> dict[str, Any]:
    variants = list(_rescue_variants(fusion_config))
    predictions = []
    predicted_events = []
    partition_rows = []
    for scope in v4["meta_folds"]:
        heldout = int(scope["heldout_calibration_fold"])
        print(f"Nested additive rescue: heldout {heldout}...", flush=True)
        validation = dtp[dtp.calibration_fold == heldout].reset_index(drop=True)
        training = dtp[dtp.calibration_fold != heldout].reset_index(drop=True)
        validation_subjects = set(validation.subject_key.astype(str))
        training_subjects = set(training.subject_key.astype(str))
        train_baseline = baseline[
            baseline.subject_key.astype(str).isin(training_subjects)
        ].reset_index(drop=True)
        validation_baseline = baseline[
            baseline.subject_key.astype(str).isin(validation_subjects)
        ].reset_index(drop=True)
        train_truth, train_ignore = partition_evaluation_events(events, training_subjects)
        train_generated = probabilities_to_events(training, **component["generator"])
        fitted: dict[float, tuple[float, pd.DataFrame]] = {}
        training_rows = []
        original_metrics = None
        for variant in variants:
            gate = None
            if variant["quantile"] is not None:
                quantile = float(variant["quantile"])
                if quantile not in fitted:
                    threshold = float(fit_score_threshold(train_generated, quantile))
                    fitted[quantile] = (threshold, train_generated)
                threshold, reference = fitted[quantile]
                accepted = train_generated[train_generated.score >= threshold]
                gate = event_gate(training, accepted, reference)
            fused = _apply_variant(train_baseline, training, scope, fusion_config, variant, gate)
            metrics, _ = evaluate_fusion_predictions(
                fused, train_truth, train_ignore, scope["postprocess"]
            )
            if variant["name"] == "original_v4":
                original_metrics = metrics
            training_rows.append({"variant": variant, "metrics": metrics})
        assert original_metrics is not None
        eligible = [
            row
            for row in training_rows
            if row["metrics"]["f1"] >= original_metrics["f1"]
            and row["metrics"]["different_sensitivity"] >= original_metrics["different_sensitivity"]
            and row["metrics"]["strict_no_ignore_f1"]
            >= original_metrics["strict_no_ignore_f1"] - 0.005
        ]
        if not eligible:
            eligible = [training_rows[0]]
        selected = max(
            eligible,
            key=lambda row: _variant_rank(row["metrics"], row["variant"]["alpha"]),
        )
        variant = selected["variant"]
        validation_gate = None
        threshold = None
        if variant["quantile"] is not None:
            threshold, reference = fitted[float(variant["quantile"])]
            accepted = frozen_events(validation, component["generator"], threshold)
            validation_gate = event_gate(validation, accepted, reference)
        fused = _apply_variant(
            validation_baseline,
            validation,
            scope,
            fusion_config,
            variant,
            validation_gate,
        )
        fused["calibration_fold"] = heldout
        validation_truth, validation_ignore = partition_evaluation_events(
            events, validation_subjects
        )
        metrics, fold_events = evaluate_fusion_predictions(
            fused, validation_truth, validation_ignore, scope["postprocess"]
        )
        predictions.append(fused)
        predicted_events.append(fold_events)
        partition_rows.append(
            {
                "heldout_calibration_fold": heldout,
                "selected_variant": variant,
                "train_only_score_threshold": threshold,
                "training_candidates": training_rows,
                "candidate_metrics": metrics,
                "baseline_metrics": scope["baseline_metrics"],
            }
        )
    combined_predictions = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["subject_key", "session_id", "timestamp_ms"])
        .reset_index(drop=True)
    )
    combined_events = pd.concat(predicted_events, ignore_index=True)
    truth, ignore = partition_evaluation_events(events, set(dtp.subject_key.astype(str)))
    metrics = summarize_event_predictions(
        combined_predictions,
        combined_events,
        truth,
        ignore,
        iou_threshold=float(baseline_postprocess["iou_threshold"]),
        matching_method=str(baseline_postprocess["matching_method"]),
    )
    gate = evaluate_v4_gate(
        metrics,
        v4["meta_oof_baseline_metrics"],
        fusion_config["promotion_gate"],
        partition_rows,
    )
    return {
        "selection_scope": "train-only variant selection per heldout meta fold",
        "metrics": metrics,
        "fusion_gate": gate,
        "partitions": partition_rows,
    }


def run_ablation(args: Any) -> Path:
    require_clean_git_worktree()
    config = load_config(args.config)
    if int(args.fold) != 0:
        raise ValueError("Only the observed development fold 0 can be compared")
    _, output_root = resolve_roots(config)
    if not re.fullmatch(r"fusion_event_ablation_[A-Za-z0-9_-]{1,64}", args.run_name):
        raise ValueError("Use a new fusion_event_ablation_... run name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.fusion_run):
        raise ValueError("Invalid frozen v4 run name")
    if not re.fullmatch(r"dtp_postprocess_[A-Za-z0-9_-]{1,64}", args.component_run):
        raise ValueError("Invalid DTP component run name")
    fusion_dir = output_root / "experiments" / args.fusion_run / "fold_0"
    component_dir = output_root / "experiments" / args.component_run / "fold_0"
    output = output_root / "experiments" / args.run_name / "fold_0"
    if output.exists():
        raise FileExistsError("Fusion event-gate ablation must use a new run name")
    component = _load_selection(component_dir, args.component_run)
    fusion_path = fusion_dir / "selected_fusion.json"
    fusion_manifest = json.loads((fusion_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if fusion_manifest["artifact_hashes"].get(fusion_path.name) != sha256_file(fusion_path):
        raise RuntimeError("Frozen v4 selection hash changed")
    original_path = fusion_dir / "validation_predictions.parquet"
    if fusion_manifest["artifact_hashes"].get(original_path.name) != sha256_file(original_path):
        raise RuntimeError("Frozen v4 meta predictions hash changed")
    v4 = json.loads(fusion_path.read_text(encoding="utf-8"))
    if v4.get("protocol_version") != 4 or v4.get("fold") != 0:
        raise ValueError("Expected a frozen v4 fold-0 selection")
    if component["source_hashes"] != v4["source_artifact_hashes"]:
        raise RuntimeError("DTP component and v4 fusion use different prediction sources")
    if component["baseline_hashes"] != v4["baseline_artifact_hashes"]:
        raise RuntimeError("DTP component and v4 fusion use different frozen baselines")
    if component["input_hashes"] != v4["input_hashes"]:
        raise RuntimeError("DTP component and v4 fusion use different input labels or folds")
    source_dir = output_root / "experiments" / component["source_run"] / "fold_0"
    baseline_dir = output_root / "experiments" / v4["baseline_experiment"] / "fold_0"
    if component["source_hashes"]["dtp_oof_predictions.parquet"] != sha256_file(
        source_dir / "dtp_oof_predictions.parquet"
    ) or component["baseline_hashes"]["validation_predictions.parquet"] != sha256_file(
        baseline_dir / "validation_predictions.parquet"
    ):
        raise RuntimeError("Frozen DTP or baseline predictions changed")
    baseline = pd.read_parquet(baseline_dir / "validation_predictions.parquet")
    dtp = pd.read_parquet(source_dir / "dtp_oof_predictions.parquet")
    training_subjects = sorted(set(dtp.subject_key.astype(str)))
    events = pd.read_parquet(
        output_root / "indices" / "events.parquet",
        filters=[("subject_key", "in", training_subjects)],
    )
    original = pd.read_parquet(original_path)
    postprocess = json.loads((baseline_dir / "selected_postprocess.json").read_text())
    comparisons = compare_event_gates(
        baseline, dtp, events, v4, component, config["fusion"], postprocess, original
    )
    output.mkdir(parents=True)
    report = {
        "development_only": True,
        "outer_predictions_read": False,
        "selection_rule": "Component candidates were chosen on observed fold-0 OOF; fusion gate only is promotable",
        "comparison": comparisons,
        "source_hashes": {
            "component": sha256_file(component_dir / "selected_dtp_postprocess.json"),
            "fusion": sha256_file(fusion_path),
            "baseline_predictions": sha256_file(baseline_dir / "validation_predictions.parquet"),
            "dtp_predictions": sha256_file(source_dir / "dtp_oof_predictions.parquet"),
            "labels": sha256_file(output_root / "indices" / "events.parquet"),
            "config": sha256_file(Path(config["_config_path"])),
        },
    }
    from bme_eating.fusion_v4 import write_json_atomic

    write_json_atomic(output / "fusion_event_gate_ablation.json", report)
    write_json_atomic(
        output / "run_manifest.json",
        {
            "experiment": {"name": args.run_name, "fold": 0},
            "git": _git_identity(),
            "config_sha256": report["source_hashes"]["config"],
            "input_hashes": report["source_hashes"],
            "artifact_hashes": {
                "fusion_event_gate_ablation.json": sha256_file(
                    output / "fusion_event_gate_ablation.json"
                )
            },
            "selection_scope": "development_only_outer_train_oof",
        },
    )
    return output
