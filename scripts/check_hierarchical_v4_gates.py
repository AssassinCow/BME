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
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--mode", choices=("ablation", "development", "stress"), required=True)
    parser.add_argument("--run-name", required=True, help="S4 candidate run name")
    parser.add_argument("--s0-run", required=True)
    parser.add_argument("--s1-run")
    parser.add_argument("--s2-run")
    parser.add_argument("--s3-run")
    parser.add_argument("--ppg-only-run")
    parser.add_argument("--v3-run")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, _, output_root = resolve_artifact_roots(config)
    if args.mode == "ablation":
        required = {
            "--s1-run": args.s1_run,
            "--s2-run": args.s2_run,
            "--s3-run": args.s3_run,
            "--ppg-only-run": args.ppg_only_run,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"ablation mode requires {', '.join(missing)}")
        report = evaluate_fold0_ablations(
            output_root,
            s0_run=args.s0_run,
            s1_run=str(args.s1_run),
            s2_run=str(args.s2_run),
            s3_run=str(args.s3_run),
            s4_run=args.run_name,
            ppg_only_run=str(args.ppg_only_run),
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
