from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import uuid
from pathlib import Path

import torch

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import (
    config_identity,
    digest,
    initialize_run,
    run_lock,
    runtime_identity,
    save_torch_atomic,
    source_identity,
    verify_artifact_hashes,
)
from bme_eating.joint_resume_migration import source_changes

ALLOWED_SOURCE_CHANGES = {
    "scripts/run_statsfusion_joint.py", "scripts/fork_statsfusion_joint_run.py",
    "src/bme_eating/joint_protocol.py", "src/bme_eating/joint_progress.py",
    "src/bme_eating/joint_continuation.py", "src/bme_eating/joint_single_fold.py",
    "src/bme_eating/joint_validation.py", "src/bme_eating/joint_workflow.py",
    "src/bme_eating/joint_v411_prediction.py", "src/bme_eating/joint_v411_search.py",
    "src/bme_eating/training/statsfusion_joint_trainer.py",
}
TRAINING_SOURCES = (
    "src/bme_eating/training/statsfusion_joint_trainer.py",
    "src/bme_eating/joint_v411_prediction.py",
    "src/bme_eating/joint_v411_search.py",
)


def training_semantics(source):
    class WithoutProgress(ast.NodeTransformer):
        def __init__(self):
            self.iterables = {}

        def visit_ImportFrom(self, node):
            return None if node.module == "bme_eating.joint_progress" else node

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)):
                function = node.value.func
                if isinstance(function, ast.Name) and function.id == "progress":
                    self.iterables[node.targets[0].id] = self.visit(node.value.args[0])
                    return None
                if (node.targets[0].id == "groups" and isinstance(function, ast.Attribute)
                        and isinstance(function.value, ast.Name) and function.value.id == "executor"
                        and function.attr == "map"):
                    self.iterables[node.targets[0].id] = node.value
                    return None
            return self.generic_visit(node)

        def visit_Name(self, node):
            return copy.deepcopy(self.iterables.get(node.id, node)) if isinstance(node.ctx, ast.Load) else node

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "progress":
                return self.visit(node.args[0])
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                function = node.value.func
                if isinstance(function, ast.Name) and function.id in {"print", "print_metrics"}:
                    return None
                if (isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                        and function.value.id in self.iterables and function.attr in {"set_postfix", "close"}):
                    return None
            return self.generic_visit(node)

    return ast.dump(WithoutProgress().visit(ast.parse(source)), include_attributes=False)


def verify_training_sources(parent, root):
    for relative in TRAINING_SOURCES:
        previous = (parent / "source_snapshot" / relative).read_text(encoding="utf-8")
        current = (root / relative).read_text(encoding="utf-8")
        if training_semantics(previous) != training_semantics(current):
            raise RuntimeError(f"Continuation changes training or validation semantics: {relative}")


def continuation_config_matches(previous, current):
    previous = copy.deepcopy({key: value for key, value in previous.items() if not key.startswith("_")})
    current = copy.deepcopy({key: value for key, value in current.items() if not key.startswith("_")})
    old_epochs, new_epochs = previous["training"].pop("max_epochs"), current["training"].pop("max_epochs")
    previous.pop("execution", None)
    current.pop("execution", None)
    return new_epochs > old_epochs and previous == current


def extend_selector_payload(payload, config, identity, source, data):
    payload = copy.deepcopy(payload)
    state = payload["state"]
    old_epochs = payload["requested_epochs"]
    if (not state.get("training_complete") or state.get("epoch") != old_epochs + 1
            or state.get("next_chunk") != 0 or state.get("session_key") is not None
            or state.get("positive_gradients") or state.get("session_count") != 0
            or state.get("clipped_steps") != 0 or state.get("optimizer_steps") != 0
            or state.get("stale_checks", 0) >= config["training"]["patience_checks"]
            or config["training"]["max_epochs"] <= old_epochs):
        raise RuntimeError("Continuation requires a clean epoch-limit completion, not an early stop or partial session")
    payload.update({"identity": identity, "config_sha256": config_identity(config),
                    "source_sha256": digest(source), "data_sha256": digest(data),
                    "requested_epochs": config["training"]["max_epochs"]})
    state["training_complete"] = False
    return payload


def fork_completed_selector(parent: Path, config, root: Path, output: Path, run_name: str):
    parent, root, output = parent.resolve(), root.resolve(), output.resolve()
    target = output / "runs" / run_name
    if target.exists() or target.resolve() == parent:
        raise FileExistsError("Continuation requires a new, empty run name")
    with run_lock(parent):
        original = json.loads((parent / "identity.json").read_text(encoding="utf-8"))
        previous = json.loads((parent / "resolved_config.json").read_text(encoding="utf-8"))
        status = json.loads((parent / "execution_status.json").read_text(encoding="utf-8"))
        if status.get("status") not in {"PAUSED", "FAILED", "COMPLETE"}:
            raise RuntimeError("Pause the parent run before creating a continuation")
        if not continuation_config_matches(previous, config):
            raise RuntimeError("Continuation may only extend max_epochs and change evaluation scope")
        if original["config_sha256"] != config_identity(previous):
            raise RuntimeError("Parent resolved configuration differs from its identity")
        current_source = source_identity(root)
        changes = source_changes(original["source"], current_source)
        if set(changes) - ALLOWED_SOURCE_CHANGES:
            raise RuntimeError("Continuation includes unauthorized model/training source changes")
        verify_training_sources(parent, root)
        if original["runtime"] != runtime_identity():
            raise RuntimeError("Continuation requires the same Python/Torch/CUDA/environment identity")
        verify_artifact_hashes(parent / "source_snapshot", original["source"])
        selection = parent / "fold_0" / "internal" / "partition_0" / "selection"
        resume_path = selection / "resume.pt"
        original_hashes = {"identity.json": sha256_file(parent / "identity.json"),
                           "resolved_config.json": sha256_file(parent / "resolved_config.json")}
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (resume.get("config_sha256") != original["config_sha256"]
                or resume.get("source_sha256") != digest(original["source"])
                or resume.get("data_sha256") != digest(original["data"])):
            raise RuntimeError("Parent selector checkpoint identity differs")
        old_epochs = resume["requested_epochs"]
        if old_epochs != previous["training"]["max_epochs"]:
            raise RuntimeError("Parent checkpoint epoch ceiling differs")
        extend_selector_payload(resume, config, resume["identity"], current_source, original["data"])
        prediction_exclusions = sorted(set(resume["excluded_subjects"]) - set(resume["selector_subjects"]))
        from bme_eating.config import require_environment_path
        from bme_eating.data.joint_sessions import JointInputs
        from bme_eating.training.hierarchical_v4_trainer import stacking_partitions

        output_base = require_environment_path(config, "output_root_env")
        inputs = JointInputs.audit(output_base / config["project"]["input_artifact_schema_version"],
                                  output_base / config["project"]["canonical_schema_version"])
        if inputs.identity != original["data"]:
            raise RuntimeError("Continuation input data differs from the parent")
        outer = {subject for subject, fold in inputs.subject_folds.items() if fold == 0}
        outer_train = set(inputs.subject_folds) - outer
        partitions = stacking_partitions(outer_train, 3, 2026)
        prediction = {subject for subject, partition in partitions.items() if partition == 0}
        train = outer_train - prediction
        old_identity = digest({"run": original_hashes["identity.json"], "training": sorted(train),
            "prediction": sorted(prediction), "outer": sorted(outer), "partition": 0})
        if (resume["identity"] != old_identity or set(prediction_exclusions) != outer | prediction
                or not resume["training_subjects"] or not resume["selector_subjects"]
                or set(resume["training_subjects"]) & set(resume["selector_subjects"])
                or set(resume["excluded_subjects"]) != outer | prediction | set(resume["selector_subjects"])
                or set(resume["training_subjects"]) | set(resume["selector_subjects"]) != train
                or set(resume["training_subjects"]) & set(resume["excluded_subjects"])):
            raise RuntimeError("Parent fold0/internal0 exclusion lineage differs")
        for name in ("selected.pt", "latest.pt", "diagnostic_selected.pt"):
            payload = torch.load(selection / name, map_location="cpu", weights_only=True)
            if (payload["identity"] != old_identity or payload["config_sha256"] != original["config_sha256"]
                    or payload["source_sha256"] != digest(original["source"])
                    or payload["data_sha256"] != digest(original["data"])):
                raise RuntimeError("Parent epoch checkpoint differs from its frozen identity")
        staging_output = output / ".fork" / uuid.uuid4().hex[:12]
        run = initialize_run(config, root, staging_output, run_name, original["data"], fresh=True,
            skip_s0=original["s0"]["skipped"], historical_reference=original["historical_reference"])
        new_identity_hash = sha256_file(run / "identity.json")
        identity = digest({"run": new_identity_hash, "training": sorted(train),
            "prediction": sorted(prediction), "outer": sorted(outer), "partition": 0})
        destination = run / "fold_0" / "internal" / "partition_0" / "selection"
        destination.mkdir(parents=True)
        copied = {}
        for path in selection.rglob("*"):
            if not path.is_file() or "cache" in path.relative_to(selection).parts or path.suffix == ".partial":
                continue
            relative = path.relative_to(selection)
            copied_path = destination / relative
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            expected = sha256_file(path)
            shutil.copyfile(path, copied_path)
            if sha256_file(path) != expected or sha256_file(copied_path) != expected:
                raise RuntimeError("Parent artifact changed while copying")
            copied[relative.as_posix()] = expected
        for name in ("selector_split.json",):
            shutil.copyfile(selection.parent / name, destination.parent / name)
        for name in ("selected.pt", "latest.pt", "diagnostic_selected.pt"):
            payload = torch.load(destination / name, map_location="cpu", weights_only=True)
            payload.update({"identity": identity, "config_sha256": config_identity(config),
                            "source_sha256": digest(current_source)})
            save_torch_atomic(destination / name, payload)
        extended = extend_selector_payload(resume, config, identity, current_source, original["data"])
        save_torch_atomic(destination / "resume.pt", extended)
        provenance = run / "continuation_parent"
        provenance.mkdir()
        for name in ("identity.json", "resolved_config.json", "execution_status.json"):
            shutil.copyfile(parent / name, provenance / name)
        shutil.copyfile(resume_path, provenance / "selector_resume.pt")
        shutil.copytree(parent / "source_snapshot", provenance / "source_snapshot")
        receipt = {"continuation": "v411_epoch_limit_extension_v1", "parent_run": str(parent),
            "parent_identity_sha256": original_hashes["identity.json"],
            "new_identity_sha256": new_identity_hash, "old_model_identity": old_identity,
            "new_model_identity": identity, "previous_max_epochs": old_epochs,
            "maximum_epochs": config["training"]["max_epochs"], "resume_epoch": extended["state"]["epoch"],
            "model_optimizer_gradients_rng_preserved": True,
            "early_stopping_preserved": True, "source_changes": changes,
            "training_validation_ast_unchanged": list(TRAINING_SOURCES),
            "copied_selection_sha256_before_rebinding": copied,
            "copied_selection_sha256_after_rebinding": {name: sha256_file(destination / name) for name in copied},
            "parent_backup_sha256": {path.relative_to(provenance).as_posix(): sha256_file(path)
                                     for path in provenance.rglob("*") if path.is_file()},
            "not_reused": ["retrain", "internal OOF", "operating points for final deployment", "outer predictions"],
            "historical_metrics_scope": "parent selector development evidence"}
        write_json_atomic(run / "continuation_manifest.json", receipt)
        write_json_atomic(run / "execution_status.json", {"status": "PAUSED", "stages": {},
            "current_stage": "outer", "current_fold": 0, "evaluation_mode": config.get("execution", {}).get("mode", "fivefold"),
            "continuation_sha256": sha256_file(run / "continuation_manifest.json"),
            "resume_epoch": extended["state"]["epoch"]})
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("Continuation target appeared during preparation")
        os.rename(run, target)
        run.parent.rmdir()
        staging_output.rmdir()
        return target

