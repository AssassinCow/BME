from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_v4_gates import (
    evaluate_crossfold_gate,
    evaluate_fold0_ablations,
)
from bme_eating.reproducibility import require_git_worktree


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StatsFusion v4 promotion gates")
    parser.add_argument(
        "--config", default="configs/hierarchical_v4_r32_pooled_heads_early_select.yaml"
    )
    parser.add_argument("--mode", choices=("ablation", "development", "stress"), required=True)
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
        report = evaluate_crossfold_gate(
            output_root,
            candidate_run=args.run_name,
            s0_run=args.s0_run,
            folds=(0, 1) if args.mode == "development" else (2, 3, 4),
            gate=config["promotion_gate"],
            mode=args.mode,
            v3_root=output_base / "v3" if args.v3_run else None,
            v3_run=args.v3_run,
        )
    print("PASS" if report["passed"] else "FAIL")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
