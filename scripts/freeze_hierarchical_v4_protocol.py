from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import _canonical_hash, sha256_file, write_json_atomic
from bme_eating.hierarchical_v4_gates import verify_gate_evidence
from bme_eating.reproducibility import git_worktree_identity, require_git_worktree


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze v4 before folds 2-4")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    _, _, output_root = resolve_artifact_roots(config)
    path = output_root / "experiments" / args.run_name / "freeze_manifest.json"
    if path.exists():
        if args.fresh:
            raise FileExistsError(path)
        return
    if args.resume:
        raise FileNotFoundError(path)
    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    evidence: dict[str, str] = {}
    for fold in (0, 1):
        manifest = output_root / "experiments" / args.run_name / f"fold_{fold}" / "run_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Fold {fold} is not complete enough to freeze")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("stage") != "EVALUATED":
            raise RuntimeError(f"Fold {fold} must be EVALUATED before protocol freeze")
        evidence[f"fold_{fold}_manifest"] = sha256_file(manifest)
    for name, relative in (
        ("fold0_ablation", "ablation/fold_0_report.json"),
        ("development_gate", "development_gate.json"),
    ):
        report_path = output_root / "experiments" / args.run_name / relative
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("passed", False)):
            raise RuntimeError(f"V4 {name} did not pass; folds 2-4 remain blocked")
        verify_gate_evidence(output_root.parent, report)
        evidence[name] = sha256_file(report_path)
    write_json_atomic(
        path,
        {
            "resolved_config_sha256": _canonical_hash(public),
            "git": git_worktree_identity(project_root),
            "locked_after_folds": [0, 1],
            "evidence_sha256": evidence,
        },
    )


if __name__ == "__main__":
    main()
