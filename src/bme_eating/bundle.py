from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import torch

from bme_eating.artifacts import sha256_file, write_json_atomic
from bme_eating.candidates import FrozenSourceCDF
from bme_eating.development_split import validate_split
from bme_eating.protocol import (
    calibration_discriminates,
    checkpoint_qualified,
    config_identity,
    configure_numerics,
    digest,
    event_score_formula,
    protocol_for,
    validate_config,
    verify_artifact_hashes,
)
from bme_eating.streaming import JointDetector
from bme_eating.workflow import phase_identity, read_json, verify_manifest


def export_bundle(config, run, source_root, *, resume=False):
    final = verify_manifest(run, config, "final_manifest.json", "final39")
    if not final.get("export_eligible"):
        raise RuntimeError("Final39 qualification failed; no deployment bundle")
    verify_manifest(run, config, "final39_training_manifest.json", "final39")
    bundle = run / "bundle"
    if bundle.exists():
        if not resume:
            raise FileExistsError("v414 bundle exists; use --resume")
        detector = StatsFusionJointFinalDataDetector(bundle)
        if detector.identity != final["model_identity"] or sha256_file(bundle / "final_manifest.json") != sha256_file(run / "final_manifest.json"):
            raise RuntimeError("Existing bundle differs from completed final39")
        return {"bundle": str(bundle), "resumed": True, "promotable": False}
    staging = run / "bundle.partial"
    if staging.exists():
        if not resume:
            raise RuntimeError("Incomplete v414 bundle requires --resume")
        backup = run / "failures" / ("incomplete_v414_bundle_" + uuid.uuid4().hex)
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, backup)
    staging.mkdir()
    filenames = {"identity.json", "resolved_config.json", "split_manifest.json", "dev_selection.json",
                 "final39_training_manifest.json", "calibration_manifest.json", "final_manifest.json", "operating_point.json"}
    for name in ("dev_selection.json", "final39_training_manifest.json", "calibration_manifest.json", "final_manifest.json"):
        filenames.update(read_json(run / name)["evidence_sha256"])
    for relative in sorted(filenames):
        source = (run / relative).resolve()
        if not source.is_relative_to(run.resolve()):
            raise RuntimeError("Unsafe bundle evidence path")
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(run / "final39/model/latest.pt", staging / "checkpoint.pt")
    shutil.copyfile(run / "final39/evaluation/source_fit/source_mapping.json", staging / "source_mapping.json")
    split_path = config["development"]["split_path"]
    split_target = staging / split_path
    split_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_root / split_path, split_target)
    identity = read_json(run / "identity.json")
    for relative, checksum in identity["source"].items():
        if relative.startswith("src/"):
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / "source_snapshot" / relative, target)
            if sha256_file(target) != checksum:
                raise RuntimeError("Bundle source differs from frozen run snapshot")
    hashes = {path.relative_to(staging).as_posix(): sha256_file(path)
              for path in staging.rglob("*") if path.is_file()}
    write_json_atomic(staging / "SHA256SUMS.json", hashes)
    StatsFusionJointFinalDataDetector(staging)
    os.replace(staging, bundle)
    return {"bundle": str(bundle), "promotable": False, "official_test_candidate": True}


class StatsFusionJointFinalDataDetector(JointDetector):
    def __init__(self, bundle: str | Path, *, device="cpu"):
        root = Path(bundle).resolve()
        hashes = read_json(root / "SHA256SUMS.json")
        required = {"checkpoint.pt", "source_mapping.json", "operating_point.json", "final_manifest.json",
                    "final39_training_manifest.json", "calibration_manifest.json", "dev_selection.json",
                    "split_manifest.json", "identity.json", "resolved_config.json"}
        if not required.issubset(hashes):
            raise RuntimeError("Incomplete v414 final-data bundle")
        verify_artifact_hashes(root, hashes)
        files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name != "SHA256SUMS.json" and "__pycache__" not in path.parts}
        if files != set(hashes):
            raise RuntimeError("v414 checksum coverage differs")
        self.config = read_json(root / "resolved_config.json")
        self.config["_config_path"] = str(root / "configs/resolved_config.json")
        validate_config(self.config)
        split = validate_split(read_json(root / "split_manifest.json"))
        split_path = root / self.config["development"]["split_path"]
        if not split_path.resolve().is_relative_to(root) or sha256_file(split_path) != self.config["development"]["split_sha256"] or read_json(split_path) != split:
            raise RuntimeError("Bundle split identity differs")
        selection = verify_manifest(root, self.config, "dev_selection.json", "dev")
        training = verify_manifest(root, self.config, "final39_training_manifest.json", "final39")
        calibration = verify_manifest(root, self.config, "calibration_manifest.json", "final39")
        final = verify_manifest(root, self.config, "final_manifest.json", "final39")
        identity = read_json(root / "identity.json")
        if (identity.get("protocol") != protocol_for(self.config)
                or identity.get("config_sha256") != config_identity(self.config)
                or split["data_identity"] != identity.get("data")
                or final.get("artifact_kind") != "full_data_official_test_candidate"
                or final.get("official_test_candidate") is not True or final.get("export_eligible") is not True
                or training.get("training_qualified") is not True
                or training.get("from_scratch") is not True or training.get("selector_subjects") != []
                or training.get("lr_horizon_epochs") != 64
                or training.get("development_selection_sha256") != sha256_file(root / "dev_selection.json")
                or final.get("fixed_epoch") != selection.get("selected_epoch")
                or training.get("fixed_epoch") != final["fixed_epoch"]
                or calibration.get("qualified") is not True):
            raise RuntimeError("v414 final training/epoch/qualification lock failed")
        train, dev = set(split["all_subjects"]), set(split["development_subjects"])
        if (selection.get("model_identity") != phase_identity(root, self.config, split, "dev")
                or final.get("model_identity") != phase_identity(root, self.config, split, "final39", final["fixed_epoch"])
                or set(selection.get("training_subjects", [])) != set(split["training_subjects"])
                or set(selection.get("development_subjects", [])) != dev):
            raise RuntimeError("v414 checkpoint identities do not belong to this run and phase")
        for manifest in (final, calibration):
            if (manifest.get("calibration_kind") != "development_on_final_model"
                    or manifest.get("labels_participated_in_training") is not True
                    or manifest.get("independent_calibration") is not False
                    or manifest.get("independent_test_evidence") is not False
                    or manifest.get("uses_official_test_labels") is not False
                    or set(manifest.get("training_subjects", [])) != train
                    or set(manifest.get("calibration_subjects", [])) != dev):
                raise RuntimeError("v414 final development calibration scope differs")
        if set(training.get("training_subjects", [])) != train or training.get("excluded_subjects") != []:
            raise RuntimeError("v414 final39 cohort differs")
        self.point = read_json(root / "operating_point.json")
        if (self.point != read_json(root / "final39/evaluation/operating_point.json")
                or self.point.get("boundary_enabled") is not False or self.point.get("boundary_mode") != "off"
                or self.point.get("boundary_entropy") is not None
                or final.get("boundary_enabled") is not False or final.get("boundary_mode") != "off"
                or self.point.get("score_formula") != event_score_formula(self.config)
                or final.get("score_formula") != event_score_formula(self.config)
                or self.point.get("calibration_kind") != "development_on_final_model"
                or self.point.get("independent_calibration") is not False
                or not self.point.get("promotion_eligible") or not self.point["calibration"].get("qualified")
                or self.point["calibration"] != calibration["calibration"]
                or not calibration_discriminates(self.config, self.point["calibration"])):
            raise RuntimeError("v414 locked scoring/calibration/Boundary differs")
        payload = torch.load(root / "checkpoint.pt", map_location=device, weights_only=True)
        if (sha256_file(root / "checkpoint.pt") != sha256_file(root / "final39/model/latest.pt")
                or self.point.get("checkpoint_sha256") != sha256_file(root / "checkpoint.pt")
                or not checkpoint_qualified(payload, final["model_identity"], train, self.config["training"]["checkpoint_min_epoch"])
                or payload["epoch"] != final["fixed_epoch"] or payload.get("excluded_subjects") != []
                or set(payload["preprocessing"]["training_subjects"]) != train
                or set(payload["preprocessing"]["scaler"]["training_subjects"]) != train
                or payload.get("config_sha256") != config_identity(self.config)
                or payload.get("data_sha256") != digest(identity["data"])
                or payload.get("source_sha256") != digest(identity["source"])
                or training["model_identity"] != payload["identity"]
                or calibration["model_identity"] != payload["identity"]
                or self.point.get("model_identity") != payload["identity"]):
            raise RuntimeError("v414 checkpoint/model/data/source lineage differs")
        for relative, checksum in identity["source"].items():
            if relative.startswith("src/") and hashes.get(relative) != checksum:
                raise RuntimeError("v414 bundle source differs from immutable run identity")
            if relative.startswith("src/") and sha256_file(Path(__file__).resolve().parents[2] / relative) != checksum:
                raise RuntimeError("Imported v414 runtime differs from frozen bundle source; use bundle/src in a new process")
        self.mapping = FrozenSourceCDF(**read_json(root / "source_mapping.json"))
        if self.mapping.training_subjects != sorted(train) or sha256_file(root / "source_mapping.json") != sha256_file(root / "final39/evaluation/source_fit/source_mapping.json"):
            raise RuntimeError("v414 natural source CDF training cohort differs")
        configure_numerics(self.config)
        from bme_eating.models.statsfusion_joint_event import build_joint_model
        self.model = build_joint_model(self.config).to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.preprocessing, self.boundary = payload["preprocessing"], payload["boundary"]
        self.identity = payload["identity"]
