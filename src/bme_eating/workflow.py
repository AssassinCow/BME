from __future__ import annotations

import json
import time
from pathlib import Path
from pickle import UnpicklingError

import torch

from bme_eating.artifacts import sha256_file, write_json_atomic
from bme_eating.development_split import load_split
from bme_eating.diagnostics import (
    locked_subject_bootstrap,
    operating_point_stability,
    stratified_diagnostics,
)
from bme_eating.evaluation import (
    NonnegativePlatt,
    apply_online_budget,
    candidate_coverage,
    commit_candidates,
    event_frame,
    metrics_with_exposure,
)
from bme_eating.progress import print_metrics
from bme_eating.protocol import (
    calibration_discriminates,
    checkpoint_qualified,
    config_identity,
    digest,
    event_score_formula,
    protocol_for,
    training_source_identity,
    verify_artifact_hashes,
)
from bme_eating.training.statsfusion_joint_trainer import (
    choose_operating_point,
    fit_source_mapping,
    load_trained,
    partition_evaluation_events,
    predict_joint_model,
    tensors_finite,
    train_joint_model,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def phase_identity(run, config, split, phase, epoch=None):
    train = split["training_subjects"] if phase == "dev" else split["all_subjects"]
    return digest({"run_identity": sha256_file(run / "identity.json"), "protocol": protocol_for(config),
                   "phase": phase, "training_subjects": train,
                   "split_sha256": config["development"]["split_sha256"], "fixed_epoch": epoch})


def bind_manifest(run, config, phase, payload, paths):
    return {**payload, "protocol": protocol_for(config), "phase": phase, "promotable": False,
            "identity_sha256": sha256_file(run / "identity.json"),
            "split_sha256": config["development"]["split_sha256"],
            "evidence_sha256": {relative: sha256_file(run / relative) for relative in paths}}


def verify_manifest(run, config, filename, phase):
    manifest = read_json(run / filename)
    if (manifest.get("protocol") != protocol_for(config) or manifest.get("phase") != phase
            or manifest.get("promotable") is not False
            or manifest.get("identity_sha256") != sha256_file(run / "identity.json")
            or manifest.get("split_sha256") != config["development"]["split_sha256"]):
        raise RuntimeError("v414 stage identity differs: " + filename)
    verify_artifact_hashes(run, manifest["evidence_sha256"])
    if filename == "final39_training_manifest.json":
        verify_final39_training(run, config, manifest)
    return manifest


def neighbor_checkpoint_records(run, config, model_identity, subjects, epochs):
    identity = read_json(run / "identity.json")
    preprocessing = read_json(run / "final39/model/preprocessing.json")
    expected = {"config_sha256": config_identity(config), "data_sha256": digest(identity["data"]),
                "source_sha256": digest(identity["source"])}
    records = []
    for epoch in epochs:
        relative = f"final39/model/nearby/epoch_{epoch:03d}.pt"
        path = run / relative
        if not path.is_file():
            raise RuntimeError("Missing v414 neighbor checkpoint: " + relative)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("epoch") != epoch or payload.get("epoch_complete") is not True
                or payload.get("identity") != model_identity
                or set(payload.get("training_subjects", [])) != subjects
                or payload.get("excluded_subjects") != []
                or payload.get("preprocessing") != preprocessing
                or set(preprocessing.get("training_subjects", [])) != subjects
                or set(preprocessing.get("scaler", {}).get("training_subjects", [])) != subjects
                or not payload.get("model") or not tensors_finite(payload["model"].values())
                or not 0 <= payload.get("clipping_fraction", -1) <= 1
                or any(payload.get(name) != value for name, value in expected.items())):
            raise RuntimeError("v414 neighbor checkpoint epoch/lineage/finite contract failed: " + relative)
        records.append({"path": relative, "epoch": epoch, "sha256": sha256_file(path),
            "epoch_complete": True, "promotion_eligible": bool(payload.get("promotion_eligible")),
            "clipping_fraction": payload["clipping_fraction"]})
    return records


def verify_final39_training(run, config, manifest):
    selection = verify_manifest(run, config, "dev_selection.json", "dev")
    split = load_split(config)
    epoch = selection["selected_epoch"]
    epochs = [epoch + offset for offset in (-4, -2, 0, 2, 4) if epoch + offset > 0]
    model_identity = phase_identity(run, config, split, "final39", epoch)
    required = {"dev_selection.json", "split_manifest.json", "final39/model/latest.pt",
                "final39/model/preprocessing.json", "final39/model/training_history.json",
                *(f"final39/model/nearby/epoch_{retained:03d}.pt" for retained in epochs)}
    if (manifest.get("fixed_epoch") != epoch or manifest.get("executed_epochs") != epoch + 4
            or manifest.get("retained_checkpoint_epochs") != epochs
            or manifest.get("model_identity") != model_identity
            or set(manifest.get("training_subjects", [])) != set(split["all_subjects"])
            or manifest.get("excluded_subjects") != [] or manifest.get("selector_subjects") != []
            or manifest.get("from_scratch") is not True or manifest.get("training_qualified") is not True
            or manifest.get("lr_horizon_epochs") != 64
            or manifest.get("development_selection_sha256") != sha256_file(run / "dev_selection.json")
            or not required.issubset(manifest.get("evidence_sha256", {}))):
        raise RuntimeError("v414 final39 training/neighbor evidence contract failed")
    records = neighbor_checkpoint_records(run, config, model_identity, set(split["all_subjects"]), epochs)
    if (manifest.get("neighbor_checkpoints") != records
            or any(manifest["evidence_sha256"][record["path"]] != record["sha256"] for record in records)):
        raise RuntimeError("v414 neighbor checkpoint metadata/checksum differs")
    deployment = next(record for record in records if record["epoch"] == epoch)
    if (not deployment["promotion_eligible"] or epoch < config["training"]["checkpoint_min_epoch"]
            or deployment["clipping_fraction"] > config["training"]["clipping_promotion_fraction"]
            or manifest.get("clipping_fraction") != deployment["clipping_fraction"]):
        raise RuntimeError("v414 default deployment checkpoint is not qualified")
    if sha256_file(run / "final39/model/latest.pt") != deployment["sha256"]:
        raise RuntimeError("v414 default checkpoint differs from retained deployment epoch")
    history = read_json(run / "final39/model/training_history.json")
    if [record["epoch"] for record in history] != list(range(1, epoch + 5)) or any("selector" in record for record in history):
        raise RuntimeError("v414 final39 epoch history is incomplete or contains selection")


def final39_training_progress(run, config):
    selection = verify_manifest(run, config, "dev_selection.json", "dev")
    split = load_split(config)
    epoch = selection["selected_epoch"]
    model_identity = phase_identity(run, config, split, "final39", epoch)
    progress = {"fixed_epoch": epoch, "target_epochs": epoch + 4,
                "deployment_checkpoint_saved": False, "deployment_checkpoint_qualified": False,
                "last_recorded_epoch": 0, "extension_training_verified": False}
    path = run / "final39/model/latest.pt"
    if path.exists():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            identity = read_json(run / "identity.json")
            expected = {"config_sha256": config_identity(config), "source_sha256": digest(identity["source"]),
                        "data_sha256": digest(identity["data"])}
            progress["deployment_checkpoint_saved"] = bool(payload.get("epoch") == epoch
                and payload.get("epoch_complete") and payload.get("identity") == model_identity
                and set(payload.get("training_subjects", [])) == set(split["all_subjects"])
                and payload.get("excluded_subjects") == []
                and all(payload.get(name) == value for name, value in expected.items())
                and payload.get("model") and tensors_finite(payload["model"].values()))
            progress["deployment_checkpoint_qualified"] = bool(progress["deployment_checkpoint_saved"]
                and checkpoint_qualified(payload, model_identity, set(split["all_subjects"]),
                                         config["training"]["checkpoint_min_epoch"]))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError, UnpicklingError) as error:
            progress["deployment_checkpoint_error"] = {"type": type(error).__name__, "message": str(error)}
    history = run / "final39/model/training_history.json"
    if history.exists():
        progress["last_recorded_epoch"] = max((record["epoch"] for record in read_json(history)), default=0)
    if (run / "final39_training_manifest.json").exists():
        try:
            verify_manifest(run, config, "final39_training_manifest.json", "final39")
            progress["extension_training_verified"] = True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError, UnpicklingError) as error:
            progress["extension_verification_error"] = {"type": type(error).__name__, "message": str(error)}
    return progress


def audit(inputs, config, run):
    split = load_split(config, inputs)
    path = run / "split_manifest.json"
    if path.exists():
        if read_json(path) != split:
            raise RuntimeError("Run split manifest differs from the frozen partition")
    else:
        write_json_atomic(path, split)
    write_json_atomic(run / "input_audit.json", {"data_identity": inputs.identity,
        "training": split["training_statistics"], "development": split["development_statistics"],
        "boundary_deployment_mode": "off", "boundary_reason": "Pre-registered v414 policy; development support below 60",
        "formal_gates_applied": False, "promotable": False})
    return split


def checkpoint_contract(payload, config, inputs, model_identity, subjects, excluded, epoch):
    expected = {"config_sha256": config_identity(config), "data_sha256": digest(inputs.identity),
                "source_sha256": digest(training_source_identity(Path(config["_config_path"]).resolve().parents[1], config))}
    if (not checkpoint_qualified(payload, model_identity, subjects, config["training"]["checkpoint_min_epoch"])
            or payload["epoch"] != epoch or set(payload.get("excluded_subjects", [])) != excluded
            or set(payload["preprocessing"]["training_subjects"]) != subjects
            or any(payload.get(name) != value for name, value in expected.items())):
        raise RuntimeError("v414 checkpoint completion/epoch/cohort/source/finite/clipping gate failed")


def evaluate_development(model, payload, inputs, config, run, split, phase, model_identity):
    directory = run / phase / "evaluation"
    train = set(split["training_subjects"] if phase == "dev" else split["all_subjects"])
    dev = set(split["development_subjects"])
    started = time.monotonic()
    mapping = fit_source_mapping(model, inputs, train, config, payload["preprocessing"],
                                 directory / "source_fit", model_identity)
    scope = "subject_excluded_development" if phase == "dev" else "development_on_final_model"
    authorization = None if phase == "dev" else {"kind": scope,
        "split_sha256": config["development"]["split_sha256"], "data_sha256": digest(inputs.identity)}
    outputs = predict_joint_model(model, inputs, dev, config, payload["preprocessing"],
        directory / "prediction", model_identity, {**payload["boundary"], "search_eligible": False},
        mapping=mapping, development_calibration=authorization)
    _, _, events, _ = inputs.frames(dev, labels=True)
    truth, ignore = partition_evaluation_events(events, dev)
    point, records = choose_operating_point(outputs["coarse"], outputs["windows"], truth, ignore,
                                          {model_identity: mapping}, config)
    point = {**point, "boundary_enabled": False, "boundary_mode": "off", "boundary_entropy": None,
        "score_formula": event_score_formula(config), "calibration_kind": scope,
        "independent_calibration": False, "subject_excluded_calibration": phase == "dev", "independent_test_evidence": False,
        "promotable": False, "model_identity": model_identity,
        "checkpoint_sha256": sha256_file(run / phase / "model" / ("selected.pt" if phase == "dev" else "latest.pt")),
        "split_sha256": config["development"]["split_sha256"], "training_subjects": sorted(train),
        "calibration_subjects": sorted(dev), "uses_test_labels": False,
        "boundary_reason": "v414 deployment Boundary is pre-registered off"}
    candidates, budget_audit = apply_online_budget(outputs["coarse"][point["setting"]], outputs["windows"],
        {model_identity: mapping}, point["budget"], config["decoder"]["quotas"])
    predictions = commit_candidates(candidates, NonnegativePlatt(**point["calibration"]), point["acceptance"], point["nms"])
    metrics, matches = metrics_with_exposure(truth, predictions, ignore, outputs["windows"])
    for name in ("f1", "true_positive", "false_positive", "false_negative", "fp_per_hour"):
        if abs(metrics[name] - point[name]) > 1e-9:
            raise RuntimeError("Search and locked streaming evaluation differ: " + name)
    report = {"calibration_kind": scope, "independent_test_evidence": False, "metrics": metrics,
        "coverage": candidate_coverage(candidates, truth, outputs["windows"]),
        "strata": stratified_diagnostics(truth, predictions, candidates, ignore, outputs["windows"], split),
        "bootstrap": locked_subject_bootstrap(truth, predictions, ignore, outputs["windows"], dev,
                                            config["gates"]["bootstrap_replicates"]),
        "operating_point_stability": operating_point_stability(records, point),
        "elapsed_seconds": time.monotonic() - started, "registered_operating_points": len(records),
        "boundary_enabled": False, "promotable": False}
    report["formal_gate_ratio_diagnostics"] = {
        "expected_truth_count": len(truth), "absolute_161_event_gate_applied": False,
        "registered_coverage_ratio": config["gates"]["minimum_candidate_covered"] / config["gates"]["expected_truth_count"],
        "observed_coverage_ratio": report["coverage"]["recall"], "used_for_promotion": False}
    from bme_eating.diagnostics import endpoint_diagnostics
    report["endpoints"] = endpoint_diagnostics(outputs["windows"], truth, ignore, candidates)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("truth", truth), ("ignore", ignore), ("windows", outputs["windows"]),
                        ("predictions", event_frame(predictions)), ("candidates", event_frame(candidates))):
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    matches.to_csv(directory / "matches.csv", index=False)
    for name, value in (("operating_point", point), ("report", report), ("search", records), ("budget_audit", budget_audit)):
        write_json_atomic(directory / f"{name}.json", value)
    print_metrics(f"{phase} DEVELOPMENT ({scope})", {**point, **metrics})
    return point, report


def evaluation_paths(phase):
    return [f"{phase}/evaluation/{name}" for name in
        ("operating_point.json", "report.json", "search.json", "budget_audit.json", "truth.parquet",
         "ignore.parquet", "windows.parquet", "predictions.parquet", "candidates.parquet", "matches.csv",
         "source_fit/source_mapping.json", "source_fit/performance.json", "prediction/performance.json")]


def execute_dev(inputs, config, run, *, resume=False):
    split = audit(inputs, config, run)
    if (run / "dev_selection.json").exists():
        if not resume:
            raise FileExistsError("Development selection already exists")
        return verify_manifest(run, config, "dev_selection.json", "dev")
    train, dev = set(split["training_subjects"]), set(split["development_subjects"])
    identity = phase_identity(run, config, split, "dev")
    directory = run / "dev/model"
    epoch = train_joint_model(inputs, train, config, directory, identity,
        epochs=config["training"]["max_epochs"], resume=resume, selector=dev)
    model, payload = load_trained(directory, config, identity, train, inputs.identity, checkpoint_name="selected.pt")
    checkpoint_contract(payload, config, inputs, identity, train, dev, epoch)
    point, report = evaluate_development(model, payload, inputs, config, run, split, "dev", identity)
    if not point["promotion_eligible"] or not calibration_discriminates(config, point["calibration"]):
        raise RuntimeError("Selected development model has no qualified operating point")
    history = read_json(directory / "training_history.json")
    paths = ["split_manifest.json", "dev/model/selected.pt", "dev/model/preprocessing.json",
             "dev/model/training_history.json", *evaluation_paths("dev")]
    manifest = bind_manifest(run, config, "dev", {"selected_epoch": epoch, "model_identity": identity,
        "training_subjects": sorted(train), "development_subjects": sorted(dev),
        "metrics": report["metrics"], "coverage": report["coverage"], "operating_point": point,
        "clipping_fraction": payload["clipping_fraction"], "selection_state": history[-1].get("selection_state"),
        "epoch_selection_repeated_in_final": False, "formal_gates_applied": False,
        "evidence_scope": "subject_excluded_development_used_for_epoch_and_point_selection"}, paths)
    write_json_atomic(run / "dev_selection.json", manifest)
    return manifest


def execute_final39(inputs, config, run, *, resume=False):
    split = audit(inputs, config, run)
    selection = verify_manifest(run, config, "dev_selection.json", "dev")
    epoch = selection["selected_epoch"]
    train = set(split["all_subjects"])
    identity = phase_identity(run, config, split, "final39", epoch)
    directory = run / "final39/model"
    if (run / "final_manifest.json").exists():
        if not resume:
            raise FileExistsError("Final39 already complete")
        verify_manifest(run, config, "final39_training_manifest.json", "final39")
        verify_manifest(run, config, "calibration_manifest.json", "final39")
        return verify_manifest(run, config, "final_manifest.json", "final39")
    if (run / "final39_training_manifest.json").exists():
        if not resume:
            raise FileExistsError("Final39 training already exists; use --resume")
        verify_manifest(run, config, "final39_training_manifest.json", "final39")
    else:
        completed_epoch = train_joint_model(inputs, train, config, directory, identity,
            epochs=epoch + 4, resume=resume, deployment_epoch=epoch)
        if completed_epoch != epoch + 4:
            raise RuntimeError("Final39 must complete the frozen epoch plus four neighbor epochs")
        model, payload = load_trained(directory, config, identity, train, inputs.identity)
        checkpoint_contract(payload, config, inputs, identity, train, set(), epoch)
        history = read_json(directory / "training_history.json")
        if [record["epoch"] for record in history] != list(range(1, epoch + 5)) or any("selector" in record for record in history):
            raise RuntimeError("Final39 contains incomplete epochs or repeated epoch selection")
        retained_epochs = [epoch + offset for offset in (-4, -2, 0, 2, 4) if epoch + offset > 0]
        neighbors = neighbor_checkpoint_records(run, config, identity, train, retained_epochs)
        training_manifest = bind_manifest(run, config, "final39", {"model_identity": identity,
            "fixed_epoch": epoch, "executed_epochs": epoch + 4,
            "retained_checkpoint_epochs": retained_epochs, "neighbor_checkpoints": neighbors,
            "training_subjects": sorted(train), "excluded_subjects": [],
            "from_scratch": True, "selector_subjects": [], "lr_horizon_epochs": 64,
            "development_selection_sha256": sha256_file(run / "dev_selection.json"),
            "training_qualified": True, "clipping_fraction": payload["clipping_fraction"],
            "uses_official_test_labels": False}, ["dev_selection.json", "split_manifest.json",
                "final39/model/latest.pt", "final39/model/preprocessing.json", "final39/model/training_history.json",
                *(record["path"] for record in neighbors)])
        verify_final39_training(run, config, training_manifest)
        write_json_atomic(run / "final39_training_manifest.json", training_manifest)
        del model
    model, payload = load_trained(directory, config, identity, train, inputs.identity)
    checkpoint_contract(payload, config, inputs, identity, train, set(), epoch)
    point, report = evaluate_development(model, payload, inputs, config, run, split, "final39", identity)
    calibration = bind_manifest(run, config, "final39", {"calibration_kind": "development_on_final_model",
        "model_identity": identity, "calibration_subjects": split["development_subjects"],
        "training_subjects": sorted(train), "labels_participated_in_training": True,
        "independent_calibration": False, "independent_test_evidence": False,
        "uses_official_test_labels": False, "calibration": point["calibration"],
        "bootstrap": report["bootstrap"], "operating_point_stability": report["operating_point_stability"],
        "qualified": bool(point["promotion_eligible"] and calibration_discriminates(config, point["calibration"]))},
        ["final39_training_manifest.json", *evaluation_paths("final39")])
    write_json_atomic(run / "calibration_manifest.json", calibration)
    if not calibration["qualified"]:
        raise RuntimeError("Final39 development calibration/coverage/FP-rate qualification failed; export closed")
    write_json_atomic(run / "operating_point.json", point)
    development_report = read_json(run / "dev/evaluation/report.json")
    write_json_atomic(run / "development_comparison.json", {"independent_test_evidence": False,
        "development_labels_used_by_final_model": True,
        "dev28": {name: development_report[name] for name in ("metrics", "coverage", "strata")},
        "final39": {name: report[name] for name in ("metrics", "coverage", "strata")}})
    manifest = bind_manifest(run, config, "final39", {"artifact_kind": "full_data_official_test_candidate",
        "official_test_candidate": True, "export_eligible": True, "model_identity": identity,
        "fixed_epoch": epoch, "training_subjects": sorted(train), "excluded_subjects": [],
        "calibration_subjects": split["development_subjects"], "calibration_kind": "development_on_final_model",
        "labels_participated_in_training": True, "independent_calibration": False,
        "independent_test_evidence": False, "uses_official_test_labels": False,
        "boundary_enabled": False, "boundary_mode": "off", "score_formula": event_score_formula(config),
        "metrics": report["metrics"], "coverage": report["coverage"]},
        ["dev_selection.json", "final39_training_manifest.json", "calibration_manifest.json",
         "operating_point.json", "split_manifest.json", "development_comparison.json",
         "final39/model/latest.pt", *evaluation_paths("final39")])
    write_json_atomic(run / "final_manifest.json", manifest)
    return manifest


def write_resource_log(run):
    stages = {}
    for phase in ("dev", "final39"):
        history = run / phase / "model/training_history.json"
        stages[phase] = {"epochs": [{name: record.get(name) for name in (
            "epoch", "elapsed_seconds", "peak_gpu_gb", "peak_gpu_reserved_gb", "process_tree_private_ram_gb")}
            for record in read_json(history)] if history.exists() else []}
        for component in ("source_fit", "prediction"):
            path = run / phase / "evaluation" / component / "performance.json"
            if path.exists():
                stages[phase][component] = read_json(path)
    write_json_atomic(run / "resource_log.json", stages)


def save_run_hashes(run):
    paths = [path for path in run.glob("*.json") if path.name != "SHA256SUMS.json"]
    write_json_atomic(run / "SHA256SUMS.json", {path.name: sha256_file(path) for path in paths})


def stage_verified(run, config, stage):
    if stage == "audit":
        return read_json(run / "split_manifest.json") == load_split(config)
    if stage == "dev":
        verify_manifest(run, config, "dev_selection.json", "dev")
    elif stage == "final39":
        verify_manifest(run, config, "final39_training_manifest.json", "final39")
        verify_manifest(run, config, "calibration_manifest.json", "final39")
        verify_manifest(run, config, "final_manifest.json", "final39")
    elif stage in {"export", "replay"}:
        from bme_eating.bundle import StatsFusionJointFinalDataDetector
        StatsFusionJointFinalDataDetector(run / "bundle")
        if stage == "replay" and read_json(run / "bundle_replay.json").get("no_xgboost_replay") is not True:
            raise RuntimeError("Missing completed replay evidence")
    return True
