from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bme_eating.calibration_v4 import LogisticScoreCombiner, ProposalCalibrationV4
from bme_eating.hierarchical_artifacts import (
    HierarchicalRun,
    sha256_file,
    write_json_atomic,
    write_parquet_atomic,
)
from bme_eating.hierarchical_v4_artifacts import (
    _saved_resume_config_hash,
    current_v4_identity,
)
from bme_eating.metrics import evaluate_events
from bme_eating.models.endpoint_refiner import (
    augment_boundary_training_proposals,
    build_endpoint_features,
    select_boundary_range,
)
from bme_eating.models.event_verifier_v4 import build_proposal_features_v4, classify_proposals
from bme_eating.reproducibility import git_worktree_identity
from bme_eating.stats_features import STATS_FEATURE_COLUMNS
from bme_eating.training import hierarchical_v4_trainer as core
from bme_eating.v4_protocol import CODE_VERSION, PROTOCOL_VERSION

DIAGNOSTIC_PROTOCOL = "single_outer_fold_subject_crossfit_heads_v1"
DIAGNOSTIC_SOURCE_PATHS = (
    "scripts/train_hierarchical_v4_single_fold_heads.py",
    "src/bme_eating/training/hierarchical_v4_single_fold_heads.py",
    "tests/test_statsfusion_v4_single_fold_heads.py",
    "src/bme_eating/training/hierarchical_v4_trainer.py",
    "tests/test_statsfusion_v4.py",
)
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _diagnostic_source_identity(project_root: Path) -> dict[str, str]:
    paths = {relative: project_root / relative for relative in DIAGNOSTIC_SOURCE_PATHS}
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Single-fold diagnostic sources are missing: {missing}")
    return {relative: sha256_file(path) for relative, path in paths.items()}


def _verify_diagnostic_worktree(project_root: Path, expected_commit: str) -> dict[str, Any]:
    active = git_worktree_identity(project_root)
    return {
        **active,
        "parent_commit": expected_commit,
        "matches_parent_commit": active.get("commit") == expected_commit,
        "source_sha256": _diagnostic_source_identity(project_root),
    }


def record_single_fold_candidate_compatibility(run: HierarchicalRun) -> None:
    if run.stage != "PROPOSALS_COMPLETE":
        raise RuntimeError("Candidate compatibility may be recorded only after generation")
    run.payload["single_fold_diagnostic_compatibility"] = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "scope": "candidate_domain_metric_key_only",
        "parent_git_enforced": False,
        "diagnostic_worktree": _verify_diagnostic_worktree(
            _project_root(), str(run.payload["git"]["commit"])
        ),
    }
    write_json_atomic(run.manifest_path, run.payload)


def load_v4_run_for_single_fold_diagnostics(
    config: dict[str, Any],
    input_root: Path,
    output_root: Path,
    run_name: str,
    fold: int,
) -> HierarchicalRun:
    run_root = output_root / "experiments" / run_name / f"fold_{fold}"
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Parent v4 run manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("run_name") != run_name or int(payload.get("outer_fold", -1)) != fold:
        raise RuntimeError("Parent v4 run identity does not match the requested run and fold")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Single-fold diagnostics require the active v4 protocol")
    if payload.get("code_version") != CODE_VERSION:
        raise RuntimeError("Single-fold diagnostics require the active v4 code version")
    current = current_v4_identity(config, input_root)
    if _saved_resume_config_hash(run_root, payload) != current["resolved_config_sha256"]:
        raise RuntimeError("Single-fold diagnostic configuration differs from the parent run")
    if payload.get("input_hashes") != current["input_hashes"]:
        raise RuntimeError("Single-fold diagnostic inputs differ from the parent run")
    parent_git = payload.get("git")
    if not isinstance(parent_git, dict) or not isinstance(parent_git.get("commit"), str):
        raise TypeError("Parent v4 run has no valid Git identity")
    _verify_diagnostic_worktree(_project_root(), str(parent_git["commit"]))
    run = HierarchicalRun(run_root, manifest_path, payload)
    run.verify_artifacts()
    return run


def _parent_identity(run: Any, config: dict[str, Any]) -> dict[str, Any]:
    parents = {
        "resolved_config": run.root / "resolved_config.yaml",
        "oof_windows": run.root / "oof" / "window_predictions.parquet",
        "oof_proposals": run.root / "oof" / "proposals_labeled.parquet",
        "outer_windows": run.root / "outer" / "window_predictions.parquet",
        "outer_proposals": run.root / "outer" / "proposals.parquet",
    }
    missing = [name for name, path in parents.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Single-fold heads are missing parent artifacts: {missing}")
    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "code_version": CODE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_name": str(run.payload["run_name"]),
        "outer_fold": int(run.payload["outer_fold"]),
        "random_seed": int(config["training"]["random_seed"]),
        "verifier_seeds": [int(value) for value in config["verifier"]["seeds"]],
        "boundary_seeds": [int(value) for value in config["boundary"]["seeds"]],
        "diagnostic_worktree": _verify_diagnostic_worktree(
            _project_root(), str(run.payload["git"]["commit"])
        ),
        "parent_sha256": {name: sha256_file(path) for name, path in parents.items()},
    }


def _write_manifest(
    path: Path,
    identity: dict[str, Any],
    stage: str,
    artifacts: list[Path],
    root: Path,
) -> None:
    write_json_atomic(
        path,
        {
            **identity,
            "stage": stage,
            "evidence_class": "development_stress_diagnostic_only",
            "formal_pooled_oof": False,
            "limitations": [
                "Only one outer fold is available.",
                (
                    "Head type, calibration, thresholds, and entropy are tuned jointly on the "
                    "single-fold state-OOF subjects."
                ),
                "This artifact cannot be consumed by the formal final trainer or bundle exporter.",
            ],
            "artifact_sha256": {
                path.relative_to(root).as_posix(): sha256_file(path)
                for path in artifacts
                if path.is_file()
            },
        },
    )


def _load_or_initialize(
    run: Any,
    config: dict[str, Any],
    *,
    fresh: bool,
    resume: bool,
) -> tuple[Path, Path, dict[str, Any], str]:
    if fresh == resume:
        raise ValueError("Choose exactly one of fresh or resume")
    root = run.root / "diagnostics" / "single_fold_heads"
    manifest_path = root / "manifest.json"
    identity = _parent_identity(run, config)
    if manifest_path.is_file():
        if fresh:
            raise FileExistsError(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        semantic_keys = set(identity) - {"diagnostic_worktree"}
        saved_identity = {key: manifest.get(key) for key in semantic_keys}
        current_identity = {key: identity.get(key) for key in semantic_keys}
        if saved_identity != current_identity:
            raise RuntimeError("Single-fold diagnostic resume identity changed")
        previous_source = manifest.get("diagnostic_worktree")
        active_source = identity.get("diagnostic_worktree")
        history = list(manifest.get("source_identity_history", []))
        if previous_source != active_source:
            transition = {"previous": previous_source, "active": active_source}
            if not history or history[-1] != transition:
                history.append(transition)
        if history:
            identity["source_identity_history"] = history
        if previous_source != active_source:
            manifest.update(identity)
            write_json_atomic(manifest_path, manifest)
        return root, manifest_path, identity, str(manifest["stage"])
    if resume:
        raise FileNotFoundError("Single-fold diagnostic does not exist; start with --fresh")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Single-fold diagnostic directory exists without a manifest")
    root.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, identity, "CREATED", [], root)
    return root, manifest_path, identity, "CREATED"


def _subject_partitions(proposals: pd.DataFrame, config: dict[str, Any]) -> dict[str, int]:
    subjects = set(proposals["subject_key"].astype(str))
    count = min(int(config["hierarchical"]["verifier_crossfit_partitions"]), len(subjects))
    if count < 2:
        raise RuntimeError("Single-fold head crossfit requires at least two OOF subjects")
    return core.stacking_partitions(
        subjects, count, int(config["training"]["random_seed"]) + 510_000
    )


def _fit_verifiers(
    root: Path,
    run: Any,
    config: dict[str, Any],
    inputs: core.V4Inputs,
    *,
    resume: bool,
) -> tuple[list[Path], dict[str, Any]]:
    proposals = pd.read_parquet(run.root / "oof" / "proposals_labeled.parquet").reset_index(
        drop=True
    )
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    if proposals.empty:
        raise RuntimeError("Single-fold verifier has no OOF proposals")
    features = build_proposal_features_v4(
        proposals,
        windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        config["verifier"],
    )
    core._assert_proposal_feature_alignment(features, proposals, context="Single-fold heads")
    categories = classify_proposals(proposals)
    partitions = _subject_partitions(proposals, config)
    partition_values = proposals["subject_key"].astype(str).map(partitions).to_numpy(dtype=int)
    matrix = core._verifier_matrix(features)
    targets = np.asarray(features.event_target, dtype=np.int64)
    weights = np.asarray(features.sample_weight, dtype=np.float64)
    logistic_candidates = [
        float(value) for value in config["verifier"].get("logistic_c_values", [1.0])
    ]
    logistic_scores = {
        value: np.full(len(proposals), np.nan, dtype=np.float64) for value in logistic_candidates
    }
    event_logit = np.full(len(proposals), np.nan, dtype=np.float64)
    iou_logit = np.full(len(proposals), np.nan, dtype=np.float64)
    selected_epochs: dict[int, list[int]] = {int(seed): [] for seed in config["verifier"]["seeds"]}
    artifacts: list[Path] = []
    lineage: list[dict[str, Any]] = []
    parent_sha256 = {
        "oof_windows": sha256_file(run.root / "oof" / "window_predictions.parquet"),
        "oof_proposals": sha256_file(run.root / "oof" / "proposals_labeled.parquet"),
    }
    for partition in sorted(set(partition_values)):
        prediction_indices = np.flatnonzero(partition_values == partition)
        training_indices = np.flatnonzero(partition_values != partition)
        training_subjects = set(proposals.iloc[training_indices]["subject_key"].astype(str))
        prediction_subjects = set(proposals.iloc[prediction_indices]["subject_key"].astype(str))
        core.assert_disjoint_subjects(
            training_subjects=training_subjects,
            prediction_subjects=prediction_subjects,
        )
        train_features = core._slice_proposal_features(features, training_indices)
        prediction_features = core._slice_proposal_features(features, prediction_indices)
        for regularization_c in logistic_candidates:
            model = LogisticScoreCombiner.fit(
                matrix[training_indices],
                targets[training_indices],
                sample_weight=weights[training_indices],
                regularization_c=regularization_c,
            )
            logistic_scores[regularization_c][prediction_indices] = model.predict(
                matrix[prediction_indices]
            )
        fit_subjects, selector_subjects, selector_split = core._selector_split(
            training_subjects,
            float(config["training"]["selector_fraction"]),
            int(config["training"]["random_seed"]) + 520_000 + partition,
            events=inputs.events,
            anchors=inputs.anchors,
        )
        train_subject_values = proposals.iloc[training_indices]["subject_key"].astype(str)
        fit_local = np.flatnonzero(train_subject_values.isin(fit_subjects).to_numpy())
        selector_local = np.flatnonzero(train_subject_values.isin(selector_subjects).to_numpy())
        fit_features = core._slice_proposal_features(train_features, fit_local)
        selector_features = core._slice_proposal_features(train_features, selector_local)
        selector_proposals = (
            proposals.iloc[training_indices].iloc[selector_local].reset_index(drop=True)
        )
        selector_truth, selector_ignore = core.partition_evaluation_events(
            inputs.events, selector_subjects
        )
        selector_windows = windows[windows["subject_key"].astype(str).isin(selector_subjects)]
        seed_predictions: list[tuple[np.ndarray, np.ndarray]] = []
        selected_for_fold: dict[str, int] = {}
        for configured_seed in config["verifier"]["seeds"]:
            seed = int(configured_seed)
            checkpoint = root / "verifier" / f"partition_{partition}_seed_{seed}.pt"
            selector_identity = {
                "fit_subjects": sorted(fit_subjects),
                "selector_subjects": sorted(selector_subjects),
                "prediction_subjects": sorted(prediction_subjects),
                "parent_artifact_sha256": parent_sha256,
            }
            selected_epoch, history = core._select_verifier_epoch(
                fit_features,
                categories[training_indices][fit_local],
                selector_features,
                selector_proposals,
                selector_truth,
                selector_ignore,
                selector_windows,
                config,
                seed=seed + partition * 100,
                checkpoint_path=checkpoint.with_name(checkpoint.stem + "_selector_last.pt"),
                resume=resume,
                identity=selector_identity,
            )
            retrain_identity = {
                "training_subjects": sorted(training_subjects),
                "prediction_subjects": sorted(prediction_subjects),
                "parent_artifact_sha256": parent_sha256,
            }
            model = core._train_verifier_model_v4(
                train_features,
                categories[training_indices],
                config,
                seed=seed + partition * 100,
                epochs=selected_epoch,
                checkpoint_path=checkpoint.with_name(checkpoint.stem + "_last.pt"),
                resume=resume,
                identity=retrain_identity,
            )
            core._save_torch_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "sequence_dim": features.sequence.shape[-1],
                    "scalar_dim": features.scalar.shape[-1],
                    "config": config["verifier"],
                    "seed": seed,
                    "epochs": selected_epoch,
                    **selector_identity,
                    "training_subjects": sorted(training_subjects),
                },
            )
            selector_path = root / "verifier" / f"partition_{partition}_seed_{seed}.json"
            write_json_atomic(
                selector_path,
                {
                    "selected_epoch": selected_epoch,
                    "history": history,
                    "selector_split": selector_split,
                    **selector_identity,
                },
            )
            artifacts.extend((checkpoint, selector_path))
            selected_epochs[seed].append(selected_epoch)
            selected_for_fold[str(seed)] = selected_epoch
            seed_predictions.append(core._infer_verifier_model(model, prediction_features, config))
        event_logit[prediction_indices] = np.mean([value[0] for value in seed_predictions], axis=0)
        iou_logit[prediction_indices] = np.mean([value[1] for value in seed_predictions], axis=0)
        lineage.append(
            {
                "partition": partition,
                "training_subjects": sorted(training_subjects),
                "selector_subjects": sorted(selector_subjects),
                "prediction_subjects": sorted(prediction_subjects),
                "selected_epoch_by_seed": selected_for_fold,
            }
        )
    if not np.isfinite(event_logit).all() or not np.isfinite(iou_logit).all():
        raise RuntimeError("Single-fold Deep crossfit left unscored proposals")
    if any(not np.isfinite(values).all() for values in logistic_scores.values()):
        raise RuntimeError("Single-fold Logistic crossfit left unscored proposals")
    subjects = set(proposals["subject_key"].astype(str))
    truth, ignore = core.partition_evaluation_events(inputs.events, subjects)
    logistic_reports: list[dict[str, Any]] = []
    logistic_frames: dict[float, pd.DataFrame] = {}
    for regularization_c, values in logistic_scores.items():
        frame = proposals.copy()
        frame["logistic_score"] = values
        frame["final_score"] = values
        diagnostics, _ = core._pooled_head_diagnostics(
            frame, "final_score", truth, windows, ignore, config
        )
        logistic_reports.append(
            {
                "regularization_c": regularization_c,
                "f1": float(diagnostics["point"]["f1"]),
                "fp_per_hour": float(diagnostics["point"]["fp_per_hour"]),
            }
        )
        logistic_frames[regularization_c] = frame
    selected_logistic = max(
        logistic_reports,
        key=lambda value: (
            float(value["f1"]),
            -float(value["fp_per_hour"]),
            -float(value["regularization_c"]),
        ),
    )
    selected_c = float(selected_logistic["regularization_c"])
    logistic_frame = logistic_frames[selected_c]
    raw = proposals.copy()
    raw["stacking_partition"] = partition_values
    raw["event_logit"] = event_logit
    raw["iou_logit"] = iou_logit
    raw["predicted_iou"] = 1.0 / (1.0 + np.exp(-iou_logit))
    raw["state_score"] = features.scalar[:, 2]
    raw["sample_weight"] = weights
    calibrated_parts: list[pd.DataFrame] = []
    for partition in sorted(set(partition_values)):
        holdout = partition_values == partition
        calibration = ProposalCalibrationV4.fit(raw.loc[~holdout])
        calibrated_parts.append(calibration.apply(raw.loc[holdout]))
    deep_frame = pd.concat(calibrated_parts, ignore_index=True).sort_values(
        "proposal_id", kind="stable"
    )
    final_calibration = ProposalCalibrationV4.fit(raw)
    baseline_frame = proposals.copy()
    baseline_frame["final_score"] = baseline_frame["generator_score"]
    baseline_diagnostics, _ = core._pooled_head_diagnostics(
        baseline_frame, "final_score", truth, windows, ignore, config
    )
    logistic_diagnostics, _ = core._pooled_head_diagnostics(
        logistic_frame, "final_score", truth, windows, ignore, config
    )
    logistic_promotion = core._pooled_head_promotion(
        baseline_diagnostics,
        logistic_diagnostics,
        minimum_f1_gain=0.005,
        config=config,
    )
    current_name = "logistic" if logistic_promotion["passed"] else "state_only"
    current_diagnostics = (
        logistic_diagnostics if current_name == "logistic" else baseline_diagnostics
    )
    deep_diagnostics, _ = core._pooled_head_diagnostics(
        deep_frame, "final_score", truth, windows, ignore, config
    )
    deep_promotion = core._pooled_head_promotion(
        current_diagnostics,
        deep_diagnostics,
        minimum_f1_gain=float(config["promotion_gate"]["minimum_verifier_f1_improvement"]),
        config=config,
    )
    winner = "deep" if deep_promotion["passed"] else current_name
    selected_frame = (
        deep_frame
        if winner == "deep"
        else logistic_frame
        if winner == "logistic"
        else baseline_frame
    )
    selected_diagnostics = (
        deep_diagnostics
        if winner == "deep"
        else logistic_diagnostics
        if winner == "logistic"
        else baseline_diagnostics
    )
    deployment_logistic = LogisticScoreCombiner.fit(
        matrix,
        targets,
        sample_weight=weights,
        regularization_c=selected_c,
    )
    outer_proposals = pd.read_parquet(run.root / "outer" / "proposals.parquet").reset_index(
        drop=True
    )
    outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    outer_features = build_proposal_features_v4(
        outer_proposals,
        outer_windows,
        [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
        config["verifier"],
    )
    core._assert_proposal_feature_alignment(
        outer_features, outer_proposals, context="Single-fold outer heads"
    )
    outer_scored = outer_proposals.copy()
    outer_scored["logistic_score"] = deployment_logistic.predict(
        core._verifier_matrix(outer_features)
    )
    deployment_epochs = {seed: int(np.median(values)) for seed, values in selected_epochs.items()}
    deployment_models: list[torch.nn.Module] = []
    if winner == "deep":
        for seed, epochs in deployment_epochs.items():
            checkpoint = root / "verifier" / f"deployment_seed_{seed}.pt"
            identity = {
                "training_subjects": sorted(subjects),
                "prediction_subjects": sorted(set(run.payload["outer_test_subjects"])),
                "parent_artifact_sha256": parent_sha256,
            }
            model = core._train_verifier_model_v4(
                features,
                categories,
                config,
                seed=seed + 590_000,
                epochs=epochs,
                checkpoint_path=checkpoint.with_name(checkpoint.stem + "_last.pt"),
                resume=resume,
                identity=identity,
            )
            core._save_torch_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "sequence_dim": features.sequence.shape[-1],
                    "scalar_dim": features.scalar.shape[-1],
                    "config": config["verifier"],
                    "seed": seed,
                    "epochs": epochs,
                    **identity,
                },
            )
            artifacts.append(checkpoint)
            deployment_models.append(model)
        if len(outer_scored):
            predictions = [
                core._infer_verifier_model(model, outer_features, config)
                for model in deployment_models
            ]
            outer_scored["event_logit"] = np.mean([value[0] for value in predictions], axis=0)
            outer_scored["iou_logit"] = np.mean([value[1] for value in predictions], axis=0)
            outer_scored["state_score"] = outer_features.scalar[:, 2]
            outer_scored = final_calibration.apply(outer_scored)
    if winner == "logistic":
        outer_scored["final_score"] = outer_scored["logistic_score"]
    elif winner == "state_only":
        outer_scored["final_score"] = outer_scored["generator_score"]
    selected_frame = selected_frame.copy()
    if "logistic_score" not in selected_frame:
        selected_frame["logistic_score"] = logistic_frame["logistic_score"].to_numpy()
    oof_path = root / "oof_proposal_scores.parquet"
    outer_path = root / "outer_proposal_scores.parquet"
    logistic_path = root / "logistic_verifier.json"
    calibration_path = root / "proposal_calibration.json"
    report_path = root / "verifier_report.json"
    write_parquet_atomic(oof_path, selected_frame)
    write_parquet_atomic(outer_path, outer_scored)
    write_json_atomic(logistic_path, deployment_logistic.to_json())
    write_json_atomic(calibration_path, final_calibration.to_json())
    report = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "winner": winner,
        "score_column": "final_score",
        "selected_point": selected_diagnostics["point"],
        "state_only": baseline_diagnostics,
        "logistic": logistic_diagnostics,
        "deep": deep_diagnostics,
        "logistic_regularization_search": logistic_reports,
        "selected_logistic_c": selected_c,
        "logistic_promotion": logistic_promotion,
        "deep_promotion": deep_promotion,
        "lineage": lineage,
        "deployment_epoch_by_seed": {str(key): value for key, value in deployment_epochs.items()},
        "formal_pooled_oof": False,
    }
    write_json_atomic(report_path, report)
    artifacts.extend((oof_path, outer_path, logistic_path, calibration_path, report_path))
    return artifacts, report


def _fit_boundary(
    root: Path,
    run: Any,
    config: dict[str, Any],
    inputs: core.V4Inputs,
    verifier_report: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[Path], dict[str, Any]]:
    scores = pd.read_parquet(root / "oof_proposal_scores.parquet")
    windows = pd.read_parquet(run.root / "oof" / "window_predictions.parquet")
    point = verifier_report["selected_point"]
    accepted = core._accepted_from_point(scores, point, "final_score")
    positive = core._attach_truth_boundaries(scores[scores["max_iou"] > 0.25], inputs.events)
    independent_events = len(
        positive[["subject_key", "session_id", "matched_event_id"]].drop_duplicates()
    )
    report: dict[str, Any] = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "independent_positive_events": independent_events,
        "formal_minimum_events": int(config["boundary"]["minimum_independent_events"]),
        "formal_promotion_eligible": False,
        "diagnostic_enabled": False,
    }
    artifacts: list[Path] = []
    if accepted.empty or independent_events < 2:
        report["disabled_reason"] = "insufficient accepted or positive events"
        report_path = root / "boundary_report.json"
        write_json_atomic(report_path, report)
        return [report_path], report
    partitions = _subject_partitions(scores, config)
    partition_values = scores["subject_key"].astype(str).map(partitions).to_numpy(dtype=int)
    selected_epochs: dict[int, list[int]] = {int(seed): [] for seed in config["boundary"]["seeds"]}
    score_parts: list[pd.DataFrame] = []
    lineage: list[dict[str, Any]] = []
    parent_sha256 = {
        "oof_scores": sha256_file(root / "oof_proposal_scores.parquet"),
        "verifier_report": sha256_file(root / "verifier_report.json"),
    }
    try:
        for partition in sorted(set(partition_values)):
            prediction_subjects = {
                subject for subject, value in partitions.items() if value == partition
            }
            train = positive[~positive["subject_key"].astype(str).isin(prediction_subjects)]
            holdout = accepted[accepted["subject_key"].astype(str).isin(prediction_subjects)]
            if holdout.empty:
                continue
            training_subjects = set(train["subject_key"].astype(str))
            if len(training_subjects) < 2:
                raise RuntimeError(f"boundary partition {partition} has too few subjects")
            fit_subjects, selector_subjects, selector_split = core._selector_split(
                training_subjects,
                float(config["training"]["selector_fraction"]),
                int(config["training"]["random_seed"]) + 610_000 + partition,
                events=inputs.events,
                anchors=inputs.anchors,
            )
            fit_positive = train[train["subject_key"].astype(str).isin(fit_subjects)]
            selector_positive = train[train["subject_key"].astype(str).isin(selector_subjects)]
            if fit_positive.empty or selector_positive.empty:
                raise RuntimeError(f"boundary partition {partition} selector has no positives")
            boundary_range = select_boundary_range(
                (train["truth_start_ms"] - train["coarse_start_ms"]).to_numpy(dtype=float) / 1000.0,
                (train["truth_end_ms"] - train["coarse_end_ms"]).to_numpy(dtype=float) / 1000.0,
                quantile=float(config["boundary"]["residual_quantile"]),
                minimum_seconds=int(config["boundary"]["minimum_range_seconds"]),
                maximum_seconds=int(config["boundary"]["maximum_range_seconds"]),
            )
            selection_range = select_boundary_range(
                (fit_positive["truth_start_ms"] - fit_positive["coarse_start_ms"]).to_numpy(
                    dtype=float
                )
                / 1000.0,
                (fit_positive["truth_end_ms"] - fit_positive["coarse_end_ms"]).to_numpy(dtype=float)
                / 1000.0,
                quantile=float(config["boundary"]["residual_quantile"]),
                minimum_seconds=int(config["boundary"]["minimum_range_seconds"]),
                maximum_seconds=int(config["boundary"]["maximum_range_seconds"]),
            )

            def make_features(frame: pd.DataFrame, value_range: Any, augment: bool):
                source = (
                    augment_boundary_training_proposals(
                        frame,
                        maximum_jitters_per_event=int(
                            config["boundary"]["maximum_jitters_per_event"]
                        ),
                        jitter_seconds=int(config["boundary"]["jitter_seconds"]),
                    )
                    if augment
                    else frame
                )
                return source, build_endpoint_features(
                    source,
                    windows,
                    [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
                    value_range,
                    config["boundary"],
                )

            _, fit_features = make_features(fit_positive, selection_range, True)
            selector_augmented, selector_features = make_features(
                selector_positive, selection_range, True
            )
            train_augmented, train_features = make_features(train, boundary_range, True)
            _, holdout_features = make_features(holdout, boundary_range, False)
            selector_subject_map = selector_augmented.set_index("boundary_sample_id")[
                "subject_key"
            ].astype(str)
            selector_subject_keys = selector_subject_map.loc[
                selector_features.sample_ids.astype(str)
            ].to_numpy(dtype=str)
            seed_outputs: list[tuple[np.ndarray, ...]] = []
            selected_for_fold: dict[str, int] = {}
            for configured_seed in config["boundary"]["seeds"]:
                seed = int(configured_seed)
                checkpoint = root / "boundary" / f"partition_{partition}_seed_{seed}.pt"
                selector_identity = {
                    "fit_subjects": sorted(fit_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "prediction_subjects": sorted(prediction_subjects),
                    "range": selection_range.__dict__,
                    "parent_artifact_sha256": parent_sha256,
                }
                selected_epoch, history = core._select_boundary_epoch(
                    fit_features,
                    selector_features,
                    config,
                    seed=seed + partition * 100,
                    selector_subject_keys=selector_subject_keys,
                    checkpoint_path=checkpoint.with_name(checkpoint.stem + "_selector_last.pt"),
                    resume=resume,
                    identity=selector_identity,
                )
                retrain_identity = {
                    "training_subjects": sorted(training_subjects),
                    "prediction_subjects": sorted(prediction_subjects),
                    "range": boundary_range.__dict__,
                    "parent_artifact_sha256": parent_sha256,
                }
                model = core._train_endpoint_model(
                    train_features,
                    config,
                    seed=seed + partition * 100,
                    epochs=selected_epoch,
                    checkpoint_path=checkpoint.with_name(checkpoint.stem + "_last.pt"),
                    resume=resume,
                    identity=retrain_identity,
                )
                core._save_torch_atomic(
                    checkpoint,
                    {
                        "model": model.state_dict(),
                        "input_dim": train_features.start_sequence.shape[-1],
                        "config": config["boundary"],
                        "seed": seed,
                        "epochs": selected_epoch,
                        **selector_identity,
                        "training_subjects": sorted(training_subjects),
                    },
                )
                selector_path = root / "boundary" / f"partition_{partition}_seed_{seed}.json"
                write_json_atomic(
                    selector_path,
                    {
                        "selected_epoch": selected_epoch,
                        "history": history,
                        "selector_split": selector_split,
                        **selector_identity,
                    },
                )
                artifacts.extend((checkpoint, selector_path))
                selected_epochs[seed].append(selected_epoch)
                selected_for_fold[str(seed)] = selected_epoch
                seed_outputs.append(core._infer_endpoint_model(model, holdout_features, config))
            output = holdout[["proposal_id"]].copy().reset_index(drop=True)
            for index, name in enumerate(
                ("start_offset_seconds", "end_offset_seconds", "start_entropy", "end_entropy")
            ):
                output[name] = np.mean([value[index] for value in seed_outputs], axis=0)
            score_parts.append(output)
            lineage.append(
                {
                    "partition": partition,
                    "training_subjects": sorted(training_subjects),
                    "selector_subjects": sorted(selector_subjects),
                    "prediction_subjects": sorted(prediction_subjects),
                    "selected_epoch_by_seed": selected_for_fold,
                    "training_samples": len(train_augmented),
                }
            )
    except (RuntimeError, ValueError) as error:
        report["disabled_reason"] = f"{type(error).__name__}: {error}"
        report["lineage"] = lineage
        report_path = root / "boundary_report.json"
        write_json_atomic(report_path, report)
        artifacts.append(report_path)
        return artifacts, report
    boundary_scores = pd.concat(score_parts, ignore_index=True)
    if set(boundary_scores["proposal_id"].astype(str)) != set(accepted["proposal_id"].astype(str)):
        raise RuntimeError("Single-fold Boundary did not score every accepted proposal")
    subjects = set(scores["subject_key"].astype(str))
    truth, ignore = core.partition_evaluation_events(inputs.events, subjects)
    coarse_metrics, _ = evaluate_events(
        truth, core._prediction_events(accepted), method="max_cardinality_iou", ignore=ignore
    )
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    threshold_reports: list[dict[str, Any]] = []
    for threshold in config["boundary"]["entropy_thresholds"]:
        refined = core._refine_from_scores(
            accepted,
            boundary_scores,
            float(threshold),
            int(config["boundary"]["safety_gap_seconds"]),
        )
        predictions = refined.rename(
            columns={"refined_start_ms": "start_ms", "refined_end_ms": "end_ms"}
        )[["subject_key", "session_id", "start_ms", "end_ms", "proposal_id"]]
        metrics, _ = evaluate_events(
            truth, predictions, method="max_cardinality_iou", ignore=ignore
        )
        mae = float(np.nanmean([metrics["start_mae_seconds"], metrics["end_mae_seconds"]]))
        coarse_mae = float(
            np.nanmean([coarse_metrics["start_mae_seconds"], coarse_metrics["end_mae_seconds"]])
        )
        passed = bool(
            np.isfinite(mae)
            and np.isfinite(coarse_mae)
            and mae < coarse_mae
            and metrics["f1"]
            >= coarse_metrics["f1"] - float(config["promotion_gate"]["maximum_boundary_f1_drop"])
        )
        row = {"threshold": float(threshold), "passed": passed, "mae": mae, **metrics}
        threshold_reports.append(row)
        if passed:
            candidates.append((mae, float(threshold), row))
    boundary_path = root / "boundary_crossfit_scores.parquet"
    write_parquet_atomic(boundary_path, boundary_scores)
    artifacts.append(boundary_path)
    report.update(
        {
            "lineage": lineage,
            "thresholds": threshold_reports,
            "formal_sample_size_eligible": independent_events
            >= int(config["boundary"]["minimum_independent_events"]),
        }
    )
    if candidates:
        _, selected_threshold, selected = min(candidates, key=lambda value: (value[0], value[1]))
        report.update(
            {
                "diagnostic_enabled": True,
                "selected_entropy_threshold": selected_threshold,
                "selected": selected,
            }
        )
        all_subjects = set(positive["subject_key"].astype(str))
        final_range = select_boundary_range(
            (positive["truth_start_ms"] - positive["coarse_start_ms"]).to_numpy(dtype=float)
            / 1000.0,
            (positive["truth_end_ms"] - positive["coarse_end_ms"]).to_numpy(dtype=float) / 1000.0,
            quantile=float(config["boundary"]["residual_quantile"]),
            minimum_seconds=int(config["boundary"]["minimum_range_seconds"]),
            maximum_seconds=int(config["boundary"]["maximum_range_seconds"]),
        )
        final_augmented = augment_boundary_training_proposals(
            positive,
            maximum_jitters_per_event=int(config["boundary"]["maximum_jitters_per_event"]),
            jitter_seconds=int(config["boundary"]["jitter_seconds"]),
        )
        final_features = build_endpoint_features(
            final_augmented,
            windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            final_range,
            config["boundary"],
        )
        outer_scores = pd.read_parquet(root / "outer_proposal_scores.parquet")
        outer_accepted = core._accepted_from_point(outer_scores, point, "final_score")
        outer_windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
        outer_features = build_endpoint_features(
            outer_accepted,
            outer_windows,
            [f"stat_{name}" for name in STATS_FEATURE_COLUMNS],
            final_range,
            config["boundary"],
        )
        outputs: list[tuple[np.ndarray, ...]] = []
        for seed, values in selected_epochs.items():
            epochs = int(np.median(values))
            checkpoint = root / "boundary" / f"deployment_seed_{seed}.pt"
            identity = {
                "training_subjects": sorted(all_subjects),
                "prediction_subjects": sorted(set(run.payload["outer_test_subjects"])),
                "range": final_range.__dict__,
                "parent_artifact_sha256": parent_sha256,
            }
            model = core._train_endpoint_model(
                final_features,
                config,
                seed=seed + 690_000,
                epochs=epochs,
                checkpoint_path=checkpoint.with_name(checkpoint.stem + "_last.pt"),
                resume=resume,
                identity=identity,
            )
            core._save_torch_atomic(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "input_dim": final_features.start_sequence.shape[-1],
                    "config": config["boundary"],
                    "seed": seed,
                    "epochs": epochs,
                    **identity,
                },
            )
            artifacts.append(checkpoint)
            if len(outer_accepted):
                outputs.append(core._infer_endpoint_model(model, outer_features, config))
        outer_boundary = outer_accepted[["proposal_id"]].copy()
        for index, name in enumerate(
            ("start_offset_seconds", "end_offset_seconds", "start_entropy", "end_entropy")
        ):
            outer_boundary[name] = (
                np.mean([value[index] for value in outputs], axis=0)
                if outputs
                else np.empty(0, dtype=float)
            )
        outer_path = root / "outer_boundary_scores.parquet"
        range_path = root / "boundary_range.json"
        write_parquet_atomic(outer_path, outer_boundary)
        write_json_atomic(range_path, final_range.__dict__)
        artifacts.extend((outer_path, range_path))
    report_path = root / "boundary_report.json"
    write_json_atomic(report_path, report)
    artifacts.append(report_path)
    return artifacts, report


def _evaluate_outer(
    root: Path,
    run: Any,
    config: dict[str, Any],
    outer_inputs: core.V4Inputs,
    verifier_report: dict[str, Any],
    boundary_report: dict[str, Any],
) -> list[Path]:
    scores = pd.read_parquet(root / "outer_proposal_scores.parquet")
    accepted = core._accepted_from_point(scores, verifier_report["selected_point"], "final_score")
    boundary_applied = bool(boundary_report.get("diagnostic_enabled", False))
    if boundary_applied:
        boundary_scores = pd.read_parquet(root / "outer_boundary_scores.parquet")
        refined = core._refine_from_scores(
            accepted,
            boundary_scores,
            float(boundary_report["selected_entropy_threshold"]),
            int(config["boundary"]["safety_gap_seconds"]),
        )
        predictions = refined.rename(
            columns={"refined_start_ms": "start_ms", "refined_end_ms": "end_ms"}
        )[["subject_key", "session_id", "start_ms", "end_ms", "proposal_id", "final_score"]]
    else:
        predictions = accepted.rename(
            columns={"coarse_start_ms": "start_ms", "coarse_end_ms": "end_ms"}
        )[["subject_key", "session_id", "start_ms", "end_ms", "proposal_id", "final_score"]]
    subjects = set(run.payload["outer_test_subjects"])
    truth, ignore = core.partition_evaluation_events(outer_inputs.events, subjects)
    maximum, _ = evaluate_events(truth, predictions, method="max_cardinality_iou", ignore=ignore)
    greedy, _ = evaluate_events(truth, predictions, method="greedy", ignore=ignore)
    windows = pd.read_parquet(run.root / "outer" / "window_predictions.parquet")
    metrics = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "evidence_class": "development_stress_diagnostic_only",
        "head": verifier_report["winner"],
        "boundary_applied": boundary_applied,
        "boundary_formal_promotion_eligible": bool(
            boundary_report.get("formal_promotion_eligible", False)
        ),
        "max_cardinality_iou": maximum,
        "greedy": greedy,
        "hand": core._hand_metrics(truth, predictions, ignore),
        "fp_per_hour": maximum["false_positive"] / max(core._observed_hours(windows), 1e-9),
    }
    prediction_path = root / "outer_predictions.parquet"
    metrics_path = root / "outer_metrics.json"
    write_parquet_atomic(prediction_path, predictions)
    write_json_atomic(metrics_path, metrics)
    return [prediction_path, metrics_path]


def train_single_fold_heads_v4(
    run: Any,
    config: dict[str, Any],
    train_inputs: core.V4Inputs,
    input_root: Path,
    *,
    fresh: bool,
    resume: bool,
) -> Path:
    if run.stage not in {"PROPOSALS_COMPLETE", "SELECTED", "EVALUATED"}:
        raise RuntimeError("Single-fold heads require completed candidate generation")
    root, manifest_path, identity, stage = _load_or_initialize(
        run, config, fresh=fresh, resume=resume
    )
    artifacts: list[Path] = []
    if stage == "COMPLETE":
        return root
    if stage == "CREATED":
        verifier_artifacts, verifier_report = _fit_verifiers(
            root, run, config, train_inputs, resume=resume
        )
        artifacts.extend(verifier_artifacts)
        _write_manifest(manifest_path, identity, "VERIFIER_COMPLETE", artifacts, root)
        stage = "VERIFIER_COMPLETE"
    else:
        verifier_report = json.loads((root / "verifier_report.json").read_text(encoding="utf-8"))
        artifacts.extend(
            path for path in root.rglob("*") if path.is_file() and path != manifest_path
        )
    if stage == "VERIFIER_COMPLETE":
        boundary_artifacts, boundary_report = _fit_boundary(
            root, run, config, train_inputs, verifier_report, resume=resume
        )
        artifacts.extend(boundary_artifacts)
        _write_manifest(manifest_path, identity, "BOUNDARY_COMPLETE", artifacts, root)
        stage = "BOUNDARY_COMPLETE"
    else:
        boundary_report = json.loads((root / "boundary_report.json").read_text(encoding="utf-8"))
    if stage == "BOUNDARY_COMPLETE":
        outer_inputs = core.load_v4_inputs(
            config,
            input_root,
            fold=int(run.payload["outer_fold"]),
            event_role="outer_test",
            allow_outer_labels=True,
        )
        artifacts.extend(
            _evaluate_outer(root, run, config, outer_inputs, verifier_report, boundary_report)
        )
        _write_manifest(manifest_path, identity, "COMPLETE", artifacts, root)
    return root
