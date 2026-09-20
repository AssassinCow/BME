from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.quality import validate_quality_gate
from bme_eating.reproducibility import require_clean_git_worktree, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill fingerprint manifests for the already frozen five-fold baseline."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--source-commit", default="3ca55bb")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.experiment):
        raise ValueError("--experiment must be a safe directory name")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.source_commit):
        raise ValueError("--source-commit must be a 7-40 character hexadecimal Git hash")
    project_root = Path(__file__).resolve().parents[1]
    require_clean_git_worktree(project_root)
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    validate_quality_gate(output_root)
    fold_directories = [
        output_root / "experiments" / args.experiment / f"fold_{fold}" for fold in range(5)
    ]
    required_artifacts = (
        "model.json",
        "metadata.json",
        "validation_predictions.parquet",
        "test_predictions.parquet",
        "selected_postprocess.json",
        "test_metrics.json",
    )
    missing_artifacts = [
        f"fold_{fold}/{required}"
        for fold, fold_dir in enumerate(fold_directories)
        for required in required_artifacts
        if not (fold_dir / required).is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            f"Frozen baseline artifacts are incomplete; missing: {missing_artifacts}"
        )
    for fold in range(5):
        fold_dir = fold_directories[fold]
        path = write_run_manifest(fold_dir, config, output_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["backfilled"] = True
        payload["backfill_scope"] = "fingerprints captured after the frozen run"
        payload["artifact_provenance"] = {
            "claimed_source_commit": args.source_commit,
            "verification": "team-declared; backfill cannot prove original training commit",
        }
        temporary_path = path.with_name(path.name + ".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(path)
        print(path)


if __name__ == "__main__":
    main()
