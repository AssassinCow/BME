from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_continuation import (
    TRAINING_SOURCES,
    continuation_config_matches,
    extend_selector_payload,
    fork_completed_selector,
    training_semantics,
)
from bme_eating.joint_protocol import (
    config_identity,
    digest,
    execution_folds,
    initialize_run,
    require_gate,
    save_torch_atomic,
    source_identity,
    stages_for,
    validate_config,
    write_failure,
)


def config_for_tests():
    return load_config(Path(__file__).resolve().parents[1] /
                       "configs/statsfusion_v411_fold0_continue64.yaml")


def completed_payload():
    return {"identity": "parent", "requested_epochs": 32,
            "model": {"weight": torch.tensor([1.25])},
            "optimizer": {"state": {0: {"exp_avg": torch.tensor([.02]), "step": torch.tensor(2784)}}},
            "gradients": {"weight": torch.tensor([.5])}, "torch_rng": torch.get_rng_state(),
            "cuda_rng": [], "python_rng": (1, 2), "numpy_rng": (3, 4),
            "state": {"epoch": 33, "next_chunk": 0, "session_key": None, "session_count": 0,
                "positive_gradients": {}, "optimizer_steps": 0, "clipped_steps": 0,
                "training_complete": True, "best_epoch": 32, "best_f1": .689655,
                "stale_checks": 0, "history": [{"epoch": 32, "selector": {"f1": .689655}}],
                "journal_offsets": {"training_history": 8187347}}}


def test_continuation_preserves_optimizer_gradients_rng_and_selection():
    payload = completed_payload()
    extended = extend_selector_payload(payload, config_for_tests(), "new", {"source": "hash"}, {"data": "hash"})
    assert extended["requested_epochs"] == 64 and extended["identity"] == "new"
    assert extended["state"]["epoch"] == 33 and not extended["state"]["training_complete"]
    for key in ("model", "optimizer", "gradients", "torch_rng"):
        def equal_trees(original, copied):
            if isinstance(original, torch.Tensor):
                assert torch.equal(original, copied)
            elif isinstance(original, dict):
                assert original.keys() == copied.keys()
                for name in original:
                    equal_trees(original[name], copied[name])
            else:
                assert original == copied
        equal_trees(payload[key], extended[key])
    expected = {**payload["state"], "training_complete": False}
    assert expected == extended["state"]
    assert payload["state"]["training_complete"]


@pytest.mark.parametrize("changes", [
    {"epoch": 32}, {"next_chunk": 1}, {"session_key": ("subject", "session")},
    {"session_count": 1}, {"stale_checks": 3}, {"training_complete": False},
    {"positive_gradients": {"event": {}}}, {"optimizer_steps": 1},
])
def test_continuation_rejects_partial_sessions_and_early_stops(changes):
    payload = completed_payload()
    payload["state"].update(changes)
    with pytest.raises(RuntimeError, match="clean epoch-limit"):
        extend_selector_payload(payload, config_for_tests(), "new", {}, {})


def test_scope_and_epoch_extension_cannot_change_training_parameters():
    current = config_for_tests()
    previous = copy.deepcopy(current)
    previous.pop("execution")
    previous["training"]["max_epochs"] = 32
    assert continuation_config_matches(previous, current)
    for section, key, value in (("loss", "event", 2), ("model", "hidden_dim", 128),
                                 ("training", "patience_checks", 9),
                                 ("training", "encoder_learning_rate", .002)):
        changed = copy.deepcopy(current)
        changed[section][key] = value
        assert not continuation_config_matches(previous, changed)


def test_fold0_mode_keeps_internal_protocol_and_does_not_run_ablations():
    config = config_for_tests()
    validate_config(config)
    assert execution_folds(config) == [0]
    assert stages_for(config) == ("audit", "validate", "outer", "gates", "final", "export", "replay")
    assert config["training"]["partitions"] == 3
    config["execution"]["mode"] = "fivefold"
    assert execution_folds(config) == [0, 1, 2, 3, 4]
    assert "ablation" in stages_for(config)


def test_single_fold_diagnostic_never_opens_formal_gate(tmp_path):
    write_json_atomic(tmp_path / "identity.json", {"protocol": "statsfusion-joint-event-v2"})
    write_json_atomic(tmp_path / "joint_gate_manifest.json", {
        "protocol": "statsfusion-joint-event-v2", "promotable": False,
        "export_eligible": True, "evaluation_mode": "fold0"})
    with pytest.raises(RuntimeError, match="closed"):
        require_gate(tmp_path)
    assert json.loads((tmp_path / "joint_gate_manifest.json").read_text())["promotable"] is False


def assert_equal_trees(original, copied):
    if isinstance(original, torch.Tensor):
        assert torch.equal(original, copied)
    elif isinstance(original, np.ndarray):
        np.testing.assert_array_equal(original, copied)
    elif isinstance(original, dict):
        assert original.keys() == copied.keys()
        for name in original:
            assert_equal_trees(original[name], copied[name])
    elif isinstance(original, (tuple, list)):
        assert type(original) is type(copied) and len(original) == len(copied)
        for left, right in zip(original, copied):
            assert_equal_trees(left, right)
    else:
        assert original == copied


def test_progress_changes_preserve_training_ast_and_detect_math_changes():
    previous = "def train(loader):\n    for item in loader:\n        loss = item * 2\n        print(loss)\n"
    current = ("from bme_eating.joint_progress import progress\ndef train(loader):\n"
               "    bar = progress(loader, total=2, description='train')\n"
               "    for item in bar:\n        loss = item * 2\n"
               "        bar.set_postfix(loss=loss)\n    bar.close()\n")
    assert training_semantics(previous) == training_semantics(current)
    assert training_semantics(previous) != training_semantics(current.replace("item * 2", "item * 3"))


def prepare_parent(tmp_path, monkeypatch):
    from bme_eating.data.joint_sessions import JointInputs
    from bme_eating.training.hierarchical_v4_trainer import stacking_partitions

    root, output = tmp_path / "source", tmp_path / "outputs"
    for relative in TRAINING_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def unchanged_training():\n    return 2\n", encoding="utf-8")
    config = config_for_tests()
    config["training"]["max_epochs"] = 32
    config.pop("execution")
    data = {"fixture": "frozen inputs"}
    folds = {f"subject_{index}": index % 5 for index in range(10)}
    outer = {subject for subject, fold in folds.items() if fold == 0}
    partitions = stacking_partitions(set(folds) - outer, 3, 2026)
    prediction = {subject for subject, partition in partitions.items() if partition == 0}
    train = set(partitions) - prediction
    selector = {max(train)}
    fit = train - selector
    inputs = SimpleNamespace(identity=data, subject_folds=folds)
    monkeypatch.setattr(JointInputs, "audit", lambda *args: inputs)
    monkeypatch.setattr("bme_eating.config.require_environment_path", lambda *args: output)
    parent = initialize_run(config, root, output / "v411", "parent", data, fresh=True,
                            skip_s0=True, historical_reference="reference")
    write_json_atomic(parent / "execution_status.json", {"status": "PAUSED", "stages": {}})
    identity = digest({"run": sha256_file(parent / "identity.json"), "training": sorted(train),
                       "prediction": sorted(prediction), "outer": sorted(outer), "partition": 0})
    payload = completed_payload()
    payload.update({"identity": identity, "config_sha256": config_identity(config),
        "source_sha256": digest(source_identity(root)), "data_sha256": digest(data),
        "selector_subjects": sorted(selector), "training_subjects": sorted(fit),
        "excluded_subjects": sorted(outer | prediction | selector)})
    selection = parent / "fold_0/internal/partition_0/selection"
    save_torch_atomic(selection / "resume.pt", payload)
    for name in ("selected.pt", "latest.pt", "diagnostic_selected.pt"):
        save_torch_atomic(selection / name, {**payload, "epoch": 32})
    (selection / "training_history.json").write_text('[{"epoch": 32}]', encoding="utf-8")
    write_json_atomic(selection.parent / "selector_split.json", {"fit": sorted(fit), "selector": sorted(selector)})
    write_json_atomic(selection / "cache/obsolete.json", {"epoch": 32})
    write_json_atomic(selection.parent / "retrain/incomplete.json", {"partial": True})
    return parent, root, output / "v411", payload


def test_real_fork_preserves_checkpoint_state_and_parent_artifacts(tmp_path, monkeypatch):
    parent, root, output, payload = prepare_parent(tmp_path, monkeypatch)
    original = {path.relative_to(parent).as_posix(): sha256_file(path)
                for path in parent.rglob("*") if path.is_file()}
    config = config_for_tests()
    run = fork_completed_selector(parent, config, root, output, "continued")
    restored = torch.load(run / "fold_0/internal/partition_0/selection/resume.pt",
                          map_location="cpu", weights_only=False)
    for name in ("model", "optimizer", "gradients", "torch_rng", "cuda_rng", "python_rng", "numpy_rng"):
        assert_equal_trees(payload[name], restored[name])
    assert_equal_trees({**payload["state"], "training_complete": False}, restored["state"])
    assert restored["requested_epochs"] == 64 and restored["state"]["epoch"] == 33
    assert not (run / "fold_0/internal/partition_0/selection/cache").exists()
    assert not (run / "fold_0/internal/partition_0/retrain").exists()
    assert initialize_run(config, root, output, "continued", {"fixture": "frozen inputs"},
        fresh=False, skip_s0=True, historical_reference="reference") == run
    for relative, expected in original.items():
        assert sha256_file(parent / relative) == expected
    with pytest.raises(FileExistsError):
        fork_completed_selector(parent, config, root, output, "continued")


def test_fork_rejects_unverified_training_changes_before_creating_target(tmp_path, monkeypatch):
    parent, root, output, _ = prepare_parent(tmp_path, monkeypatch)
    (root / TRAINING_SOURCES[0]).write_text("def unchanged_training():\n    return 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="semantics"):
        fork_completed_selector(parent, config_for_tests(), root, output, "invalid")
    assert not (output / "runs/invalid").exists()


def test_interrupted_fork_does_not_publish_a_resumable_partial_run(tmp_path, monkeypatch):
    parent, root, output, _ = prepare_parent(tmp_path, monkeypatch)
    copy_file = shutil.copyfile

    def interrupted(source, destination, *args, **kwargs):
        if Path(source).name == "resume.pt":
            raise OSError("copy interrupted")
        return copy_file(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="copy interrupted"):
        fork_completed_selector(parent, config_for_tests(), root, output, "interrupted")
    assert not (output / "runs/interrupted").exists()


def test_fold0_export_failure_quarantines_diagnostic_bundle(tmp_path):
    write_json_atomic(tmp_path / "identity.json", {"protocol": "statsfusion-joint-event-v2"})
    for name in ("fold0_bundle", "fold0_bundle.partial"):
        write_json_atomic(tmp_path / name / "checkpoint.json", {"incomplete": True})
    write_failure(tmp_path, "replay", RuntimeError("determinism failed"))
    assert not (tmp_path / "fold0_bundle").exists()
    assert not (tmp_path / "fold0_bundle.partial").exists()
    report = json.loads((tmp_path / "failure_report.json").read_text())
    assert len(report["quarantined_artifacts"]) == 2 and report["deployable"] is False


@pytest.mark.parametrize("mode", ["fold0", "fivefold"])
def test_runner_executes_selected_scope_and_separate_save_paths(tmp_path, monkeypatch, mode):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("joint_runner_scope_test", root / "scripts/run_statsfusion_joint.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = config_for_tests()
    config["execution"]["mode"] = mode
    run = tmp_path / "run"
    write_json_atomic(run / "execution_status.json", {"status": "PAUSED", "stages": {}, "error": "old error"})
    calls, folds = [], []

    def outer(inputs, config, run, fold, resume):
        folds.append(fold)
        write_json_atomic(run / f"fold_{fold}/completion.json", {"test": True})

    def recorded(name):
        def call(*args, **kwargs):
            calls.append(name)
            return {"promotable": True}
        return call

    monkeypatch.setattr(runner, "require_idle_gpu", lambda *args: None)
    monkeypatch.setattr(runner, "execute_outer", outer)
    monkeypatch.setattr(runner, "execute_gates", recorded("fivefold_gate"))
    monkeypatch.setattr(runner, "execute_final", recorded("full_data_final"))
    monkeypatch.setattr(runner, "export_joint_bundle", recorded("formal_export"))
    monkeypatch.setattr("bme_eating.joint_single_fold.execute_fold_gate", recorded("fold0_gate"))
    monkeypatch.setattr("bme_eating.joint_single_fold.save_fold_final", recorded("saved_fold0_model"))
    monkeypatch.setattr("bme_eating.joint_single_fold.export_fold_bundle", recorded("fold0_export"))
    monkeypatch.setattr("bme_eating.joint_v411_ablation.execute_ablations", recorded("ablations"))
    monkeypatch.setattr("bme_eating.joint_validation.validate_joint_implementation", recorded("validate"))

    def replay(bundle, config):
        calls.append(bundle.name)
        return {"test": True}

    monkeypatch.setattr("bme_eating.joint_validation.replay_joint_bundle", replay)
    arguments = SimpleNamespace(stage="all", fold=None, resume=True)
    runner.execute_stages(arguments, config, SimpleNamespace(identity={"data": "test"}), root, run, None)
    assert folds == ([0] if mode == "fold0" else list(range(5)))
    assert calls == (["validate", "fold0_gate", "saved_fold0_model", "fold0_export", "fold0_bundle"]
                     if mode == "fold0" else ["validate", "ablations", "fivefold_gate",
                                             "full_data_final", "formal_export", "bundle"])
    status = json.loads((run / "execution_status.json").read_text())
    assert status["status"] == "COMPLETE" and "error" not in status
