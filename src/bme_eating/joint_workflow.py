from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path

import pandas as pd
import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_evaluation import (
    NonnegativePlatt,
    apply_online_budget,
    candidate_coverage,
    commit_candidates,
    event_frame,
    metrics_with_exposure,
    paired_subject_bootstrap,
)
from bme_eating.joint_protocol import (
    PROTOCOL,
    checkpoint_qualified,
    digest,
    protocol_identities,
    require_gate,
    save_torch_atomic,
    verify_artifact_hashes,
    verify_internal_manifest,
    verify_oof_lineage,
    write_failure,
)
from bme_eating.metrics import evaluate_events, partition_evaluation_events
from bme_eating.training.statsfusion_joint_trainer import (
    fit_source_mapping,
    load_trained,
    predict_joint_model,
    train_joint_model,
    train_partition_protocol,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_outer_completion(inputs, config, run, fold):
    directory = run / f"fold_{fold}"
    path = directory / "completion.json"
    if not path.is_file():
        raise RuntimeError(f"Outer fold {fold} completion evidence is missing")
    manifest = read_json(path)
    expected_identity = {"protocol": PROTOCOL, "run_identity": sha256_file(run / "identity.json"),
                         "fold": fold, **protocol_identities(config, inputs.identity)}
    if any(manifest.get(name) != value for name, value in expected_identity.items()):
        raise RuntimeError("Outer completion config/source/data/run identity mismatch")
    required = {"evaluation.json", "outer_evidence.pt", "outer_windows.parquet", "outer_predictions.parquet",
                "outer_candidates.parquet", "truth.parquet", "ignore.parquet", "model/latest.pt",
                "internal/internal_manifest.json", "budget_audit.json", "matches.csv"}
    if not required.issubset(manifest.get("evidence_sha256", {})):
        raise RuntimeError("Outer completion evidence is incomplete")
    verify_artifact_hashes(directory, manifest["evidence_sha256"])
    report = read_json(directory / "evaluation.json")
    outer = {subject for subject, index in inputs.subject_folds.items() if index == fold}
    subjects = set(inputs.subject_folds) - outer
    identity = digest({"run": expected_identity["run_identity"], "fold": fold,
                       "training": sorted(subjects), "epoch": report["epoch"]})
    if (report.get("fold") != fold or report.get("identity") != identity
            or set(report["prediction_subjects"]) != outer or set(report["training_subjects"]) != subjects):
        raise RuntimeError("Outer evaluation subject/model identity mismatch")
    point = verify_internal_manifest(directory / "internal", config, inputs.identity, subjects, outer,
                                     expected_identity["run_identity"])
    if report["operating_point"] != point:
        raise RuntimeError("Outer operating point differs from locked internal OOF")
    checkpoint = torch.load(directory / "model" / "latest.pt", map_location="cpu", weights_only=True)
    if (not checkpoint_qualified(checkpoint, identity, subjects, config["training"]["checkpoint_min_epoch"])
            or set(checkpoint.get("excluded_subjects", [])) != outer
            or set(checkpoint.get("preprocessing", {}).get("training_subjects", [])) != subjects
            or any(checkpoint.get(name) != value for name, value in protocol_identities(config, inputs.identity).items())
            or report["checkpoint_sha256"] != sha256_file(directory / "model" / "latest.pt")):
        raise RuntimeError("Outer checkpoint qualification or lineage failed")
    return report


def execute_outer(inputs, config, run, fold, resume):
    directory = run / f"fold_{fold}"
    complete = directory / "evaluation.json"
    if resume and (directory / "completion.json").exists():
        return verify_outer_completion(inputs, config, run, fold)
    outer = {subject for subject, index in inputs.subject_folds.items() if index == fold}
    subjects = set(inputs.subject_folds) - outer
    run_identity = sha256_file(run / "identity.json")
    epoch, point = train_partition_protocol(inputs, subjects, outer, config,
                                           directory / "internal", run_identity, resume)
    verify_oof_lineage(point["lineage"], subjects, outer, identities=protocol_identities(config, inputs.identity))
    identity = digest({"run": run_identity, "fold": fold, "training": sorted(subjects), "epoch": epoch})
    train_joint_model(inputs, subjects, config, directory / "model", identity, epochs=epoch,
                      resume=resume, outer_excluded=outer)
    model, payload = load_trained(directory / "model", config, identity, subjects, inputs.identity)
    mapping = fit_source_mapping(model, inputs, subjects, config, payload["preprocessing"],
                                  directory / "source_fit", identity)
    output = predict_joint_model(model, inputs, outer, config, payload["preprocessing"],
                                directory / "prediction", identity, payload["boundary"])
    candidates, budget_audit = apply_online_budget(output["coarse"][point["setting"]], output["windows"],
        {identity: mapping}, point["budget"], config["decoder"]["quotas"])
    refined_by_id = ({candidate.proposal_id: candidate for candidate in
                      output["refined"][str(point["boundary_entropy"])][point["setting"]]}
                     if point["boundary_enabled"] else {})
    final_candidates = [refined_by_id.get(candidate.proposal_id, candidate) for candidate in candidates]
    prediction, boundary_diagnostics = commit_candidates(final_candidates, NonnegativePlatt(**point["calibration"]),
                                    point["acceptance"], point["nms"], return_diagnostics=True)
    _, _, events, _ = inputs.frames(outer, labels=True)
    truth, ignore = partition_evaluation_events(events, outer)
    metrics, matches = metrics_with_exposure(truth, prediction, ignore, output["windows"])
    coverage = candidate_coverage(candidates, truth, output["windows"])
    coarse_by_id = {candidate.proposal_id: candidate for candidate in output["coarse"][point["setting"]]}
    coarse_candidates = [coarse_by_id[candidate.proposal_id] for candidate in candidates]
    coarse_prediction = commit_candidates(coarse_candidates, NonnegativePlatt(**point["calibration"]),
                                           point["acceptance"], point["nms"])
    from bme_eating.training.statsfusion_joint_trainer import boundary_pair_gate
    boundary = boundary_pair_gate(coarse_prediction, prediction, truth, ignore, output["windows"], config,
                                  require_sixty=False, refined_candidates=boundary_diagnostics)
    locked_coarse = point["coarse_operating_point"]
    locked_prediction = commit_candidates(coarse_candidates, NonnegativePlatt(**locked_coarse["calibration"]),
                                          locked_coarse["acceptance"], locked_coarse["nms"])
    locked_metrics, _ = metrics_with_exposure(truth, locked_prediction, ignore, output["windows"])
    boundary["locked_coarse"] = locked_metrics
    boundary["passed"] &= locked_metrics["f1"] - metrics["f1"] <= config["gates"]["boundary_maximum_f1_drop"]
    greedy_metrics, _ = evaluate_events(truth, event_frame(prediction), ignore=ignore, method="greedy")
    report = {"protocol": PROTOCOL, "fold": fold, "identity": identity, "epoch": epoch,
              "training_subjects": sorted(subjects), "prediction_subjects": sorted(outer),
              "metrics": metrics, "coverage": coverage, "operating_point": point,
              "checkpoint_qualified": True, "boundary": boundary,
              "checkpoint_sha256": sha256_file(directory / "model" / "latest.pt"),
              "greedy_f1": greedy_metrics["f1"], "diagnostic_complete": True,
              "stress_f1_drop": point["f1"] - metrics["f1"]}
    output["windows"].to_parquet(directory / "outer_windows.parquet", index=False)
    event_frame(prediction).to_parquet(directory / "outer_predictions.parquet", index=False)
    event_frame(candidates).to_parquet(directory / "outer_candidates.parquet", index=False)
    matches.to_csv(directory / "matches.csv", index=False)
    truth.to_parquet(directory / "truth.parquet", index=False)
    ignore.to_parquet(directory / "ignore.parquet", index=False)
    write_json_atomic(directory / "budget_audit.json", budget_audit)
    save_torch_atomic(directory / "outer_evidence.pt", {"prediction": prediction, "candidates": candidates})
    write_json_atomic(complete, report)
    paths = ["evaluation.json", "outer_evidence.pt", "outer_windows.parquet", "outer_predictions.parquet",
             "outer_candidates.parquet", "truth.parquet", "ignore.parquet", "model/latest.pt",
             "internal/internal_manifest.json", "budget_audit.json", "matches.csv"]
    write_json_atomic(directory / "completion.json", {"protocol": PROTOCOL, "fold": fold,
        "run_identity": run_identity, **protocol_identities(config, inputs.identity),
        "evidence_sha256": {relative: sha256_file(directory / relative) for relative in paths}})
    return report


def audit_reference(reference_root: Path, inputs):
    try:
        return _audit_reference(reference_root, inputs)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, EOFError,
            pickle.UnpicklingError) as error:
        return {"qualified": False, "comparison_performed": False, "evidence_sha256": {},
                "reasons": [f"Reference audit failed: {type(error).__name__}: {error}"]}


def _audit_reference(reference_root: Path, inputs):
    reasons, files = [], {}
    path = reference_root / "joint_reference_evidence.json"
    legacy = list(reference_root.glob("**/final_manifest.json")) if reference_root.is_dir() else []
    final_manifest = reference_root.parent.parent / "final" / reference_root.name / "final_manifest.json"
    if final_manifest.is_file():
        legacy.append(final_manifest)
    if not path.is_file():
        reasons.append("No complete reference cohort/exposure/calibration/exclusion evidence")
        protocols = []
        for manifest in legacy:
            files[str(manifest)] = sha256_file(manifest)
            protocol = read_json(manifest).get("promotion_protocol")
            protocols.append(protocol)
            if protocol == "time_constrained_single_holdout_v1":
                reasons.append("Historical reference declares time_constrained_single_holdout_v1, without qualified complete outer OOF comparison evidence")
        return {"qualified": False, "comparison_performed": False, "reasons": reasons,
                "evidence_sha256": files, "legacy_manifests": [str(item) for item in legacy],
                "legacy_promotion_protocols": protocols}
    payload = read_json(path)
    required = {"truth_sha256", "ignore_sha256", "subject_folds_sha256", "predictions_file",
                "predictions_sha256", "exposure_file", "exposure_sha256", "matching", "iou_threshold",
                "strict_iou", "seed", "calibration_lineage", "model_lineages", "candidate_budget",
                "full_outer_subjects", "identity_file", "identity_sha256"}
    if not required.issubset(payload):
        reasons.append("Reference evidence schema incomplete")
    if (payload.get("subject_folds_sha256") != inputs.identity["subject_folds"]
            or payload.get("matching") != "max_cardinality_iou" or payload.get("iou_threshold") != .25
            or payload.get("strict_iou") is not True or payload.get("seed") != 2026
            or set(payload.get("full_outer_subjects", [])) != set(inputs.subject_folds)):
        reasons.append("Reference matching/seed/cohort differs")
    def verify_artifact(relative, expected_hash, description):
        target = (reference_root / relative).resolve()
        if not relative or not target.is_relative_to(reference_root.resolve()) or not target.is_file():
            reasons.append(f"Reference {description} unavailable")
        elif sha256_file(target) != expected_hash:
            reasons.append(f"Reference {description} hash differs")
        else:
            files[relative] = sha256_file(target)
            return target
        return None

    for key in ("predictions", "exposure", "identity"):
        verify_artifact(payload.get(key + "_file", ""), payload.get(key + "_sha256"), key)
    subjects = set(inputs.subject_folds)
    for kind in ("model_lineages", "calibration_lineage"):
        lineages = payload.get(kind, [])
        if len(lineages) != 5 or {lineage.get("fold") for lineage in lineages} != set(range(5)):
            reasons.append(f"Reference {kind} does not cover five outer folds exactly once")
        for lineage in lineages:
            holdout = {subject for subject, fold in inputs.subject_folds.items() if fold == lineage.get("fold")}
            train = set(lineage.get("training_subjects", []))
            prediction = set(lineage.get("prediction_subjects", []))
            if not holdout or prediction != holdout or train != subjects - holdout:
                reasons.append("Reference model/calibration exclusion or cohort failed")
            target = verify_artifact(lineage.get("artifact_file", ""), lineage.get("artifact_sha256"), kind)
            if target is not None:
                metadata = (torch.load(target, map_location="cpu", weights_only=True)
                            if target.suffix == ".pt" else read_json(target))
                if set(metadata.get("training_subjects", [])) != train:
                    reasons.append("Reference artifact training identity differs from claimed lineage")
            if kind == "calibration_lineage":
                oof_predictions = []
                for excluded in lineage.get("oof_lineages", []):
                    fit = set(excluded.get("training_subjects", []))
                    excluded_prediction = set(excluded.get("prediction_subjects", []))
                    if (not fit or not excluded_prediction or fit & (excluded_prediction | holdout)
                            or fit | excluded_prediction != train):
                        reasons.append("Reference calibration OOF exclusion failed")
                    oof_predictions.extend(excluded_prediction)
                if set(oof_predictions) != train or len(oof_predictions) != len(set(oof_predictions)):
                    reasons.append("Reference calibration lacks fully excluded OOF for every training subject")
    files[path.name] = sha256_file(path)
    return {"qualified": not reasons, "comparison_performed": False, "reasons": reasons,
            "evidence_sha256": files, "root": str(reference_root.resolve()), "payload": payload}


def execute_gates(inputs, config, run, reference_root):
    reports, prediction, candidates, windows, truth, ignore, evidence = [], [], [], [], [], [], {}
    for fold in range(5):
        directory = run / f"fold_{fold}"
        report = verify_outer_completion(inputs, config, run, fold)
        expected = {subject for subject, index in inputs.subject_folds.items() if index == fold}
        if set(report["prediction_subjects"]) != expected:
            raise RuntimeError("Outer evaluation subject cohort mismatch")
        if report["checkpoint_sha256"] != sha256_file(directory / "model" / "latest.pt"):
            raise RuntimeError("Outer checkpoint changed after holdout prediction")
        verify_oof_lineage(report["operating_point"]["lineage"], set(inputs.subject_folds) - expected, expected,
                           identities=protocol_identities(config, inputs.identity))
        payload = torch.load(directory / "outer_evidence.pt", map_location="cpu", weights_only=False)
        reports.append(report)
        prediction.extend(payload["prediction"])
        candidates.extend(payload["candidates"])
        _, _, canonical_events, _ = inputs.frames(expected, labels=True)
        canonical_truth, canonical_ignore = partition_evaluation_events(canonical_events, expected)
        for relative, canonical in (("truth.parquet", canonical_truth), ("ignore.parquet", canonical_ignore)):
            actual = pd.read_parquet(directory / relative)
            columns = sorted(canonical.columns)
            if (set(actual.columns) != set(columns) or digest(actual[columns].sort_values(columns).to_dict("records"))
                    != digest(canonical[columns].sort_values(columns).to_dict("records"))):
                raise RuntimeError("Outer truth/ignore differs from the current canonical cohort")
        if any(candidate.subject_key not in expected or candidate.model_identity != report["identity"]
               for candidate in payload["prediction"] + payload["candidates"]):
            raise RuntimeError("Outer candidate/prediction model exclusion identity mismatch")
        for key, target in (("outer_windows.parquet", windows), ("truth.parquet", truth), ("ignore.parquet", ignore)):
            target.append(pd.read_parquet(directory / key))
        for path in directory.rglob("*"):
            if path.is_file() and "cache" not in path.parts and path.suffix in {".json", ".pt", ".parquet"}:
                evidence[str(path.relative_to(run)).replace("\\", "/")] = sha256_file(path)
    windows, truth, ignore = [pd.concat(values, ignore_index=True) for values in (windows, truth, ignore)]
    if windows.duplicated(["subject_key", "session_id", "timestamp_ms"]).any():
        raise RuntimeError("Duplicate outer predictions on a subject timeline")
    metrics, _ = metrics_with_exposure(truth, prediction, ignore, windows)
    coverage = candidate_coverage(candidates, truth, windows)
    gates = config["gates"]
    checks = {"five_folds": len(reports) == 5, "training_qualified": all(report["checkpoint_qualified"] for report in reports),
              "f1": metrics["f1"] >= gates["minimum_f1"], "fp_per_hour": metrics["fp_per_hour"] <= gates["maximum_fp_per_hour"],
              "coverage": coverage["covered"] >= gates["minimum_candidate_covered"],
              "truth_cohort": coverage["truth"] == gates["expected_truth_count"],
              "fold_coverage": all(report["coverage"]["recall"] >= gates["minimum_fold_candidate_recall"] for report in reports),
              "operating_point_qualified": all(report["operating_point"]["promotion_eligible"] for report in reports),
              "stress": all(report["stress_f1_drop"] <= gates["maximum_stress_f1_drop"] for report in reports),
              "boundary": all(not report["operating_point"]["boundary_enabled"] or report["boundary"]["passed"] for report in reports)}
    reference = audit_reference(reference_root, inputs) if reference_root else {
        "qualified": False, "comparison_performed": False, "reasons": ["Reference not provided"]}
    checks["reference"] = reference["qualified"]
    paired = {"comparison_performed": False}
    if reference["qualified"]:
        payload = reference["payload"]
        if payload["truth_sha256"] != digest(truth.to_dict("records")) or payload["ignore_sha256"] != digest(ignore.to_dict("records")):
            reference["qualified"] = checks["reference"] = False
            reference["reasons"].append("Truth/ignore cohort identity differs")
        else:
            frame = pd.read_parquet(reference_root / payload["predictions_file"])
            reference_windows = pd.read_parquet(reference_root / payload["exposure_file"])
            exposure_keys = ["subject_key", "session_id", "timestamp_ms", "valid"]
            if digest(windows[exposure_keys].sort_values(exposure_keys[:3]).to_dict("records")) != digest(
                    reference_windows[exposure_keys].sort_values(exposure_keys[:3]).to_dict("records")):
                reference["qualified"] = checks["reference"] = False
                reference["reasons"].append("Reference exposure differs")
            else:
                paired = {"comparison_performed": True, **paired_subject_bootstrap(
                    truth, prediction, frame, ignore, windows, gates["bootstrap_replicates"])}
                reference["comparison_performed"] = True
                checks["bootstrap"] = paired["advantage_probability"] >= gates["minimum_bootstrap_advantage"]
                reference_metrics, _ = evaluate_events(truth, frame, ignore=ignore)
                reference_greedy, _ = evaluate_events(truth, frame, ignore=ignore, method="greedy")
                new_greedy, _ = evaluate_events(truth, event_frame(prediction), ignore=ignore, method="greedy")
                maximum_delta = metrics["f1"] - reference_metrics["f1"]
                greedy_delta = new_greedy["f1"] - reference_greedy["f1"]
                checks["matching_ranking"] = maximum_delta * greedy_delta >= 0
                reference_matched = evaluate_events(truth, frame, ignore=ignore)[1]
                for hand in ("same", "different"):
                    denominator = int(truth.hand_relation.eq(hand).sum())
                    recall = int(reference_matched.hand_relation.eq(hand).sum()) / max(denominator, 1) if len(reference_matched) else 0
                    checks[hand + "_recall"] = metrics[hand + "_recall"] is not None and (
                        recall - metrics[hand + "_recall"] <= gates["maximum_hand_recall_drop"])
                if len({report["operating_point"]["budget"] for report in reports}) == 1 and (
                        reports[0]["operating_point"]["budget"] == payload["candidate_budget"]):
                    checks["reference_coverage"] = coverage["covered"] >= payload.get("candidate_covered", float("inf"))
    manifest = {"protocol": PROTOCOL, "identity_sha256": sha256_file(run / "identity.json"),
                "diagnostic_complete": True, "promotable": all(checks.values()), "checks": checks,
                "metrics": metrics, "coverage": coverage, "folds": reports, "fold4": reports[4],
                "reference": reference, "paired_bootstrap": paired, "evidence_sha256": evidence,
                "s0": read_json(run / "identity.json")["s0"]}
    write_json_atomic(run / "joint_gate_manifest.json", manifest)
    write_json_atomic(run / "reference_audit.json", reference)
    event_frame(prediction).to_parquet(run / "pooled_outer_predictions.parquet", index=False)
    pd.DataFrame([report["metrics"] for report in reports]).to_csv(run / "fold_metrics.csv", index=False)
    if not manifest["promotable"]:
        write_failure(run, "gates", RuntimeError("Joint diagnostic complete; formal promotion failed"))
    return manifest


def execute_final(inputs, config, run, resume):
    require_gate(run)
    subjects, directory = set(inputs.subject_folds), run / "final"
    run_identity = sha256_file(run / "identity.json")
    epoch, point = train_partition_protocol(inputs, subjects, set(), config,
                                            directory / "internal", run_identity, resume)
    verify_oof_lineage(point["lineage"], subjects, set(), identities=protocol_identities(config, inputs.identity))
    if not point["promotion_eligible"]:
        raise RuntimeError("Dedicated deployment internal OOF has no qualified operating point")
    identity = digest({"run": run_identity, "training": sorted(subjects), "deployment": True, "epoch": epoch})
    train_joint_model(inputs, subjects, config, directory / "model", identity, epochs=epoch, resume=resume)
    model, payload = load_trained(directory / "model", config, identity, subjects, inputs.identity)
    if point["boundary_enabled"] and not payload["boundary"]["qualified"]:
        raise RuntimeError("Deployment Boundary head is not qualified")
    mapping = fit_source_mapping(model, inputs, subjects, config, payload["preprocessing"],
                                  directory / "source_fit", identity)
    write_json_atomic(directory / "operating_point.json", point)
    write_json_atomic(directory / "source_mapping.json", mapping.to_json())
    manifest = {"protocol": PROTOCOL, "promotable": True, "epoch": epoch, "model_identity": identity,
                "identity_sha256": run_identity, "gate_sha256": sha256_file(run / "joint_gate_manifest.json"),
                "checkpoint_sha256": sha256_file(directory / "model" / "latest.pt"),
                "operating_point_sha256": sha256_file(directory / "operating_point.json"),
                "source_mapping_sha256": sha256_file(directory / "source_mapping.json"),
                "internal_oof_sha256": sha256_file(directory / "internal" / "internal_oof.pt"),
                "internal_manifest_sha256": sha256_file(directory / "internal" / "internal_manifest.json"),
                "training_subjects": sorted(subjects),
                "lineage": point["lineage"], "boundary_enabled": point["boundary_enabled"]}
    write_json_atomic(directory / "final_manifest.json", manifest)
    return manifest


def export_joint_bundle(config, run, source_root, *, resume=False):
    gate = require_gate(run)
    final = run / "final"
    manifest = read_json(final / "final_manifest.json")
    if not manifest["promotable"] or manifest["identity_sha256"] != sha256_file(run / "identity.json"):
        raise RuntimeError("Final manifest is unqualified")
    for name, relative in (("checkpoint", "model/latest.pt"), ("operating_point", "operating_point.json"),
                           ("source_mapping", "source_mapping.json"), ("internal_oof", "internal/internal_oof.pt")):
        if sha256_file(final / relative) != manifest[name + "_sha256"]:
            raise RuntimeError(f"Final {name} lineage changed")
    if manifest["gate_sha256"] != sha256_file(run / "joint_gate_manifest.json"):
        raise RuntimeError("Final gate lock changed")
    run_identity = read_json(run / "identity.json")
    subjects = {subject for fold in gate["folds"] for subject in fold["prediction_subjects"]}
    if set(manifest.get("training_subjects", [])) != subjects:
        raise RuntimeError("Final training cohort differs from five-fold gate")
    point = verify_internal_manifest(final / "internal", config, run_identity["data"], subjects, set(),
                                     sha256_file(run / "identity.json"))
    if sha256_file(final / "internal" / "internal_manifest.json") != manifest.get("internal_manifest_sha256"):
        raise RuntimeError("Final internal OOF completion lock changed")
    payload = torch.load(final / "model" / "latest.pt", map_location="cpu", weights_only=True)
    from bme_eating.joint_protocol import checkpoint_qualified, config_identity, source_identity
    if not checkpoint_qualified(payload, manifest["model_identity"], subjects,
                                 config["training"]["checkpoint_min_epoch"]):
        raise RuntimeError("Final checkpoint failed qualification")
    if payload["config_sha256"] != config_identity(config) or payload["source_sha256"] != digest(source_identity(source_root)):
        raise RuntimeError("Final checkpoint config/source mismatch")
    if (payload["data_sha256"] != digest(run_identity["data"]) or payload.get("excluded_subjects")
            or set(payload["preprocessing"]["training_subjects"]) != subjects):
        raise RuntimeError("Final checkpoint data/preprocessing/exclusion identity mismatch")
    oof = torch.load(final / "internal" / "internal_oof.pt", map_location="cpu", weights_only=False)
    verify_oof_lineage(oof["lineages"], subjects, set(), identities=protocol_identities(config, run_identity["data"]))
    if oof["lineages"] != manifest["lineage"]:
        raise RuntimeError("Final OOF lineage differs from final manifest")
    if read_json(final / "operating_point.json") != point:
        raise RuntimeError("Final operating point differs from dedicated internal OOF")
    if not point["promotion_eligible"] or point["boundary_enabled"] != manifest["boundary_enabled"]:
        raise RuntimeError("Deployment operating point is unqualified or differs from manifest")
    if point["boundary_enabled"]:
        boundary_reports = read_json(final / "internal" / "boundary_crossfit.json")
        passed = any(report["passed"] and report["entropy"] == point["boundary_entropy"] for report in boundary_reports)
        if not payload["boundary"]["qualified"] or not passed:
            raise RuntimeError("Deployment Boundary gate failed")
    bundle = run / "bundle"
    if bundle.exists():
        if resume:
            from bme_eating.joint_pipeline import StatsFusionJointDetector
            detector = StatsFusionJointDetector(bundle, device="cpu")
            if detector.identity != manifest["model_identity"]:
                raise RuntimeError("Existing bundle model differs from dedicated final checkpoint")
            return {"bundle": str(bundle), "resumed": True}
        raise FileExistsError("A bundle already exists; export cannot overwrite it")
    staging = run / "bundle.partial"
    if staging.exists():
        write_failure(run, "export", RuntimeError("Quarantining incomplete export before resume"))
    staging.mkdir()
    deployment_payload = {key: payload[key] for key in (
        "model", "identity", "epoch", "epoch_complete", "promotion_eligible", "clipping_fraction",
        "training_subjects", "excluded_subjects", "preprocessing", "boundary",
        "config_sha256", "source_sha256", "data_sha256")}
    save_torch_atomic(staging / "checkpoint.pt", deployment_payload)
    deployment_manifest = {**manifest, "checkpoint_sha256": sha256_file(staging / "checkpoint.pt")}
    write_json_atomic(staging / "final_manifest.json", deployment_manifest)
    pairs = [
             (final / "operating_point.json", "operating_point.json"),
             (final / "source_mapping.json", "source_mapping.json"),
             (run / "resolved_config.json", "resolved_config.json"),
             (run / "joint_gate_manifest.json", "joint_gate_manifest.json"),
             (run / "identity.json", "identity.json")]
    for source, name in pairs:
        shutil.copyfile(source, staging / name)
    for source in (source_root / "src" / "bme_eating").rglob("*.py"):
        destination = staging / "src" / source.relative_to(source_root / "src")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    hashes = {str(path.relative_to(staging)).replace("\\", "/"): sha256_file(path)
              for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS.json"}
    write_json_atomic(staging / "SHA256SUMS.json", hashes)
    os.replace(staging, bundle)
    return {"bundle": str(bundle), "sha256": hashes}
