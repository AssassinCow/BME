from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import _canonical_hash, sha256_file, write_json_atomic
from bme_eating.hierarchical_v4_artifacts import resume_config_hash
from bme_eating.hierarchical_v4_gates import verify_gate_evidence
from bme_eating.reproducibility import git_worktree_identity, require_git_worktree
from bme_eating.v4_protocol import BLOCKED_PREDECESSORS, CODE_VERSION, PROTOCOL_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze v4 before folds 2-4")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion_r3.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.fold != 1:
        raise SystemExit("StatsFusion-r3 may be frozen only after fold 1")
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
    evidence: dict[str, dict[str, str]] = {}

    def lock(path):
        return {
            "relative_path": path.resolve().relative_to(output_root.parent.resolve()).as_posix(),
            "sha256": sha256_file(path),
        }

    for fold in (0, 1):
        fold_root = output_root / "experiments" / args.run_name / f"fold_{fold}"
        manifest = fold_root / "run_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Fold {fold} is not complete enough to freeze")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("stage") != "EVALUATED":
            raise RuntimeError(f"Fold {fold} must be EVALUATED before protocol freeze")
        if (
            payload.get("code_version") != CODE_VERSION
            or payload.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise RuntimeError(f"Fold {fold} is not a {PROTOCOL_VERSION}/{CODE_VERSION} artifact")
        if payload.get("resume_config_sha256") != resume_config_hash(config):
            raise RuntimeError(f"Fold {fold} configuration differs from the freeze configuration")
        for name, relative in (
            ("manifest", "run_manifest.json"),
            ("resolved_config", "resolved_config.yaml"),
            ("selected_pipeline", "selection/selected_pipeline.json"),
        ):
            artifact = fold_root / relative
            if not artifact.is_file():
                raise FileNotFoundError(f"Fold {fold} freeze evidence is missing: {relative}")
            evidence[f"fold_{fold}_{name}"] = lock(artifact)
    for name, relative in (
        ("fold0_ablation", "ablation/fold_0_report.json"),
        ("development_gate", "development_gate.json"),
    ):
        report_path = output_root / "experiments" / args.run_name / relative
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("passed", False)):
            raise RuntimeError(f"V4 {name} did not pass; folds 2-4 remain blocked")
        if name == "fold0_ablation":
            if (
                report.get("protocol_version") != PROTOCOL_VERSION
                or report.get("selected_run") != args.run_name
            ):
                raise RuntimeError(
                    "Fold-0 promotion report does not select this StatsFusion-r3 run"
                )
        elif report.get("mode") != "development" or report.get("candidate_run") != args.run_name:
            raise RuntimeError("Development gate does not select this StatsFusion-r3 run")
        verify_gate_evidence(output_root.parent, report)
        evidence[name] = lock(report_path)
    write_json_atomic(
        path,
        {
            "schema_version": 2,
            "code_version": CODE_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "blocked_predecessors": list(BLOCKED_PREDECESSORS),
            "selected_run": args.run_name,
            "resolved_config_sha256": _canonical_hash(public),
            "resume_config_sha256": resume_config_hash(config),
            "git": git_worktree_identity(project_root),
            "locked_after_folds": [0, 1],
            "candidate_minimum_seconds": int(config["decoder"]["candidate_minimum_seconds"]),
            "candidate_maximum_seconds": int(config["decoder"]["candidate_maximum_seconds"]),
            "evidence_sha256": evidence,
        },
    )


if __name__ == "__main__":
    main()
