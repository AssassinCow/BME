import json

from bme_eating.reproducibility import write_run_manifest


def test_run_manifest_hashes_inputs_without_absolute_paths(tmp_path):
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text("project: {}\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    index_dir = output_root / "indices"
    index_dir.mkdir(parents=True)
    (index_dir / "subject_folds.json").write_text("{}", encoding="utf-8")
    experiment = output_root / "experiments" / "candidate" / "fold_0"
    experiment.mkdir(parents=True)
    config = {
        "_config_path": str(config_path),
        "project": {"seed": 2026},
        "xgboost": {"random_seed": 2026},
    }
    path = write_run_manifest(
        experiment,
        config,
        output_root,
        command=[str(tmp_path / "private" / "train.py"), "--fold", "0"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["command"] == ["train.py", "--fold", "0"]
    assert "subject_folds" in payload["hashes"]
    assert str(tmp_path) not in path.read_text(encoding="utf-8")
