from __future__ import annotations

import json

import pandas as pd
import torch

from bme_eating.hierarchical_artifacts import sha256_file
from bme_eating.joint_protocol import (
    checkpoint_qualified,
    digest,
    model_run_identity,
    verify_artifact_hashes,
    verify_oof_lineage,
)


def audit_completed_joint(reference_root, inputs, *, folds=None):
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    identity = read(reference_root / "identity.json")
    from bme_eating.joint_resume_migration import active_resume_identity, migration_evidence
    identity = active_resume_identity(reference_root)
    if (identity["protocol"] not in {"statsfusion-joint-event-v1", "statsfusion-joint-event-v2", "statsfusion-joint-event-v3", "statsfusion-joint-event-v4"}
            or identity["data"] != inputs.identity
            or identity.get("seed") != 2026):
        raise RuntimeError("v410 reference protocol or data identity differs")
    evidence = {"identity.json": sha256_file(reference_root / "identity.json")}
    evidence.update(migration_evidence(reference_root))
    config_path = reference_root / "resolved_config.json"
    reference_config = read(config_path) if config_path.is_file() else {}
    from bme_eating.joint_protocol import config_identity
    if not reference_config or config_identity(reference_config) != identity["config_sha256"]:
        raise RuntimeError("Reference resolved configuration identity differs")
    single = reference_config.get("execution", {}).get("internal_mode") == "single"
    expected_partitions = (0,) if single else (0, 1, 2)
    selected_folds = list(range(5)) if folds is None else list(folds)
    if folds is None and reference_config.get("execution", {}).get("mode") == "fold0":
        raise RuntimeError("Single-fold reference cannot qualify a five-fold comparison")
    if config_path.is_file():
        evidence["resolved_config.json"] = sha256_file(config_path)
    migration = reference_root / "single_partition_migration.json"
    if migration.is_file():
        evidence[migration.name] = sha256_file(migration)
    for artifact in (reference_root / "resume_migrations").rglob("*"):
        if artifact.is_file():
            evidence[artifact.relative_to(reference_root).as_posix()] = sha256_file(artifact)
    predictions, windows, truths, ignores, candidates, lineages, calibrations, budgets = [], [], [], [], [], [], [], []
    covered = 0
    subjects = set(inputs.subject_folds)
    for fold in selected_folds:
        directory = reference_root / f"fold_{fold}"
        completion = read(directory / "completion.json")
        verify_artifact_hashes(directory, completion["evidence_sha256"])
        if completion["protocol"] != identity["protocol"] or completion["run_identity"] != model_run_identity(reference_root):
            raise RuntimeError("Reference fold completion identity mismatch")
        report = read(directory / "evaluation.json")
        holdout = {subject for subject, number in inputs.subject_folds.items() if number == fold}
        train = subjects - holdout
        if set(report["training_subjects"]) != train or set(report["prediction_subjects"]) != holdout:
            raise RuntimeError("Reference holdout/model exclusion mismatch")
        payload = torch.load(directory / "model" / "latest.pt", map_location="cpu", weights_only=True)
        if (not checkpoint_qualified(payload, report["identity"], train, 8)
                or set(payload["excluded_subjects"]) != holdout
                or payload["data_sha256"] != digest(inputs.identity)
                or payload["config_sha256"] != identity["config_sha256"]
                or payload["source_sha256"] != digest(identity["source"])
                or set(payload.get("preprocessing", {}).get("training_subjects", [])) != train):
            raise RuntimeError("Reference training checkpoint qualification failed")
        point = report["operating_point"]
        if not point["calibration"]["qualified"] or point["calibration"]["slope"] < 0:
            raise RuntimeError("Reference natural calibration is unqualified")
        internal = read(directory / "internal" / "internal_manifest.json")
        verify_artifact_hashes(directory / "internal", internal["evidence_sha256"])
        if read(directory / "internal" / "operating_point.json") != point:
            raise RuntimeError("Reference operating point differs from internal OOF lock")
        verify_oof_lineage(point["lineage"], train, holdout, identities={
            "data_sha256": payload["data_sha256"], "config_sha256": payload["config_sha256"],
            "source_sha256": payload["source_sha256"]}, expected_partitions=expected_partitions)
        for lineage in point["lineage"]:
            checkpoint = torch.load(directory / "internal" / f"partition_{lineage['partition']}" /
                                    "retrain" / "latest.pt", map_location="cpu", weights_only=True)
            if (not checkpoint_qualified(checkpoint, lineage["model_identity"],
                                          set(lineage["training_subjects"]), 8)
                    or set(checkpoint["excluded_subjects"]) != holdout | set(lineage["prediction_subjects"])
                    or set(checkpoint.get("preprocessing", {}).get("training_subjects", [])) != set(lineage["training_subjects"])
                    or checkpoint.get("epoch") != lineage["epoch"]
                    or any(checkpoint.get(name) != payload[name] for name in
                           ("source_sha256", "config_sha256", "data_sha256"))):
                raise RuntimeError("Reference internal checkpoint exclusion failed")
            selector_name = "selected.pt" if lineage.get("selector_promotion_eligible") else "diagnostic_selected.pt"
            selected = torch.load(directory / "internal" / f"partition_{lineage['partition']}" /
                                  "selection" / selector_name, map_location="cpu", weights_only=True)
            selector_fit = set(lineage["selector_fit_subjects"])
            selector = set(lineage["selector_subjects"])
            if (not checkpoint_qualified({**selected, "promotion_eligible": True}, lineage["model_identity"], selector_fit, 8)
                    or set(selected["excluded_subjects"]) != holdout | set(lineage["prediction_subjects"]) | selector
                    or set(selected.get("preprocessing", {}).get("training_subjects", [])) != selector_fit
                    or selected["epoch"] != lineage["epoch"]
                    or bool(selected.get("promotion_eligible")) != bool(lineage.get("selector_promotion_eligible"))
                    or any(selected.get(name) != payload[name] for name in
                           ("source_sha256", "config_sha256", "data_sha256"))):
                raise RuntimeError("Reference selector exclusion or qualification failed")
        for name, target in (("outer_predictions.parquet", predictions), ("outer_windows.parquet", windows),
                             ("truth.parquet", truths), ("ignore.parquet", ignores),
                             ("outer_candidates.parquet", candidates)):
            frame = pd.read_parquet(directory / name)
            if len(frame) and not set(frame.subject_key.astype(str)).issubset(holdout):
                raise RuntimeError("Reference prediction cohort leaks across folds")
            target.append(frame)
        from bme_eating.metrics import interval_iou_matrix
        for (subject, session), target in truths[-1].groupby(["subject_key", "session_id"]):
            pool = candidates[-1][candidates[-1].subject_key.astype(str).eq(str(subject))
                                   & candidates[-1].session_id.astype(str).eq(str(session))]
            covered += int((interval_iou_matrix(target[["start_ms", "end_ms"]].to_numpy(),
                                                pool[["start_ms", "end_ms"]].to_numpy()) > .25).any(1).sum())
        lineages.append({"fold": fold, "training_subjects": sorted(train), "prediction_subjects": sorted(holdout)})
        calibrations.append({**lineages[-1], "oof_lineages": point["lineage"]})
        budgets.append(point["budget"])
        evidence[f"fold_{fold}/completion.json"] = sha256_file(directory / "completion.json")
        evidence.update({f"fold_{fold}/{relative}": checksum for relative, checksum in completion["evidence_sha256"].items()})
        evidence.update({f"fold_{fold}/internal/{relative}": checksum for relative, checksum in internal["evidence_sha256"].items()})
    predictions, windows, truths, ignores, candidates = [pd.concat(frames, ignore_index=True)
                                                       for frames in (predictions, windows, truths, ignores, candidates)]
    if windows.duplicated(["subject_key", "session_id", "timestamp_ms"]).any():
        raise RuntimeError("Reference has duplicated outer exposure")
    expected_subjects = {subject for subject, fold in inputs.subject_folds.items() if fold in selected_folds}
    if set(windows.subject_key.astype(str)) != expected_subjects:
        raise RuntimeError("Reference is missing outer subject exposure")
    return {"qualified": True, "comparison_performed": False, "reasons": [],
        "root": str(reference_root.resolve()), "evidence_sha256": evidence,
        "derived_from_five_fold_completions": len(selected_folds) == 5,
        "folds": selected_folds,
        "payload": {"truth_sha256": digest(truths.to_dict("records")),
                    "ignore_sha256": digest(ignores.to_dict("records")),
                    "candidate_budget": budgets[0] if len(set(budgets)) == 1 else None,
                    "candidate_covered": covered,
                    "model_lineages": lineages, "calibration_lineage": calibrations}}
