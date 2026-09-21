import argparse
import hashlib
import json

import pytest

from bme_eating import cli
from bme_eating.fusion import FusionGateError


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_v4_source_hash_change_is_rejected(tmp_path):
    source = tmp_path / "experiments" / "baseline_dtp_fusion_source" / "fold_0"
    source.mkdir(parents=True)
    for name in ("dtp_oof_predictions.parquet", "dtp_test_predictions.parquet"):
        (source / name).write_bytes(name.encode())
    manifest = {
        "experiment": {"name": "baseline_dtp_fusion_source", "fold": 0},
        "git": {"dirty": False},
        "artifact_hashes": {
            name: _digest(source / name)
            for name in ("dtp_oof_predictions.parquet", "dtp_test_predictions.parquet")
        },
    }
    (source / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "dtp_oof_predictions.parquet").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="artifact changed"):
        cli._validate_v4_prediction_source(
            tmp_path, "baseline_dtp_fusion_source", fold=0
        )


def test_meta_gate_failure_blocks_outer_evaluation(tmp_path, monkeypatch):
    run_name = "baseline_dtp_fusion_v4test"
    fold_dir = tmp_path / "experiments" / run_name / "fold_1"
    fold_dir.mkdir(parents=True)
    selection = {
        "protocol_version": 4,
        "run_name": run_name,
        "fold": 1,
        "meta_oof_gate": {"passed": False},
        "outer_fold_gate": {"passed": None},
    }
    selection_path = fold_dir / "selected_fusion.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    manifest = {
        "experiment": {"name": run_name, "fold": 1},
        "artifact_hashes": {"selected_fusion.json": _digest(selection_path)},
    }
    (fold_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _path: {
            "fusion": {"protocol_version": 4},
            "experiment": {"name": "baseline_dtp_fusion"},
        },
    )
    monkeypatch.setattr(cli, "require_clean_git_worktree", lambda: "commit")
    monkeypatch.setattr(cli, "resolve_roots", lambda _config: (tmp_path, tmp_path))
    monkeypatch.setattr(cli, "validate_quality_gate", lambda _root: None)
    args = argparse.Namespace(config="unused.yaml", run_name=run_name, fold=1)

    with pytest.raises(FusionGateError, match="Meta-OOF promotion gate failed"):
        cli.command_evaluate_fusion_v4(args)


def test_v4_source_finalizer_does_not_write_outer_evaluation_artifacts(
    tmp_path, monkeypatch
):
    fold_dir = tmp_path / "experiments" / "baseline_dtp_fusion_source" / "fold_0"
    fold_dir.mkdir(parents=True)
    monkeypatch.setattr(
        cli,
        "write_run_manifest",
        lambda output_dir, _config, _root: (
            output_dir / "run_manifest.json"
        ).write_text("{}", encoding="utf-8"),
    )

    cli._finalize_v4_prediction_source(
        fold_dir,
        {"fusion": {"protocol_version": 4}},
        tmp_path,
        protocol_version=4,
        experiment_name="baseline_dtp_fusion_source",
        fold=0,
        crossfit_models=[{"partition": 0}],
        baseline_name="baseline",
        baseline_source_commit="abcdef0",
        baseline_info={"artifact_hashes": {"model.json": "a"}, "input_hashes": {}},
    )

    assert (fold_dir / "dtp_source.json").is_file()
    assert (fold_dir / "run_manifest.json").is_file()
    assert not (fold_dir / "selected_fusion.json").exists()
    assert not (fold_dir / "test_metrics.json").exists()
    assert not (fold_dir / "test_predictions.parquet").exists()
