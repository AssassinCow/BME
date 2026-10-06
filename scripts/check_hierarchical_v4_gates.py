from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.hierarchical_v4_gates import (
    evaluate_crossfold_gate,
    evaluate_fold0_ablations,
)
from bme_eating.integrated_v49 import (
    evaluate_v49_outer_preflight,
    is_v49,
    write_failure_report,
)
from bme_eating.reproducibility import require_git_worktree


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StatsFusion v4 promotion gates")
    parser.add_argument(
        "--config", default="configs/hierarchical_v4_r32_pooled_heads_early_select.yaml"
    )
    parser.add_argument("--mode", choices=("ablation", "development", "stress", "full"), required=True)
    parser.add_argument("--run-name", required=True, help="Selected r3 candidate run name")
    parser.add_argument("--s0-run", required=True)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="TYPE:BASELINE:CANDIDATE",
        help="Hash-locked fold-0 module gate; repeat in selected-path order",
    )
    parser.add_argument("--v3-run")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, _, output_root = resolve_artifact_roots(config)
    if args.mode == "ablation":
        comparisons = []
        for value in args.compare:
            parts = value.split(":", 2)
            if len(parts) != 3 or not all(parts):
                parser.error("--compare must use TYPE:BASELINE:CANDIDATE")
            comparisons.append((parts[0], parts[1], parts[2]))
        report = evaluate_fold0_ablations(
            output_root,
            selected_run=args.run_name,
            sensor_only_run=args.s0_run,
            comparisons=comparisons,
            gate=config["promotion_gate"],
        )
    else:
        output_base = Path(config["_output_base_path"])
        run_root = output_root / "experiments" / args.run_name
        resolved_config = run_root / "fold_0" / "resolved_config.yaml"
        config_path = resolved_config if resolved_config.is_file() else Path(args.config)
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            if args.mode == "full" and is_v49(config):
                report = evaluate_v49_outer_preflight(output_root, args.run_name, args.s0_run, config)
            else:
                report = evaluate_crossfold_gate(
                    output_root,
                    candidate_run=args.run_name,
                    s0_run=args.s0_run,
                    folds=(0, 1) if args.mode == "development" else ((2, 3, 4) if args.mode == "stress" else (0, 1, 2, 3, 4)),
                    gate=config["promotion_gate"],
                    mode=args.mode,
                    v3_root=output_base / "v3" if args.v3_run else None,
                    v3_run=args.v3_run,
                )
        except Exception as exc:
            if args.mode != "full":
                raise
            report = {
                "mode": args.mode,
                "candidate_run": args.run_name,
                "s0_run": args.s0_run,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if args.mode == "full":
            manifest = {
                "protocol": "v49_gate_manifest_v1",
                "candidate_run": args.run_name,
                "s0_run": args.s0_run,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "config_path": str(config_path),
                "crossfold_gate": report,
                "identity": report.get("identity", {}),
                "evidence_sha256": report.get("evidence_sha256", {}),
                "reference_evidence_sha256": report.get("reference_evidence_sha256", {}),
                "passed": bool(report.get("passed")),
                "folds": list(range(5)),
                "protocol_lock": {
                    "requires_final_nested_state_oof": True,
                    "final_deep_thresholds_deferred": True,
                },
            }
            write_json_atomic(run_root / "v49_gate_manifest.json", manifest)
            if not manifest["passed"]:
                write_failure_report(
                    run_root, "crossfold_gate", RuntimeError(report.get("error", str(report.get("checks")))),
                )
    print("PASS" if report["passed"] else "FAIL")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
