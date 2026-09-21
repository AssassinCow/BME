from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_roots
from bme_eating.fusion_v4_promotion import (
    evaluate_v4_final_promotion,
    load_v4_promotion_evidence,
    write_promotion_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the protocol-v4 frozen folds 1-4 fusion promotion gate."
    )
    parser.add_argument("--config", default="configs/dtp_fusion.yaml")
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--baseline-run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if int(config["fusion"].get("protocol_version", 0)) != 4:
        raise ValueError("promote_fusion_v4.py requires fusion.protocol_version=4")
    _, output_root = resolve_roots(config)
    baseline_run = str(
        args.baseline_run or config["experiment"].get("baseline_name", "baseline")
    )
    baseline_root = output_root / "experiments" / baseline_run
    candidate_root = output_root / "experiments" / str(args.candidate_run)
    report_path = args.output or candidate_root / "promotion_v4.json"
    baseline_rows, candidate_rows, evidence = load_v4_promotion_evidence(
        baseline_root, candidate_root
    )
    decision = evaluate_v4_final_promotion(
        baseline_rows,
        candidate_rows,
        config["fusion"]["final_promotion_gate"],
    )
    decision.update(
        {
            "baseline_run": baseline_run,
            "candidate_run": str(args.candidate_run),
            "evidence": evidence,
        }
    )
    report_path, sidecar_path = write_promotion_report(report_path, decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(report_path)
    print(sidecar_path)
    if not decision["promote"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
