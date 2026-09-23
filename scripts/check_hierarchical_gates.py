from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.hierarchical_gates import (
    evaluate_development_gate,
    evaluate_stress_gate,
    select_fold0_mode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen v3 promotion gates")
    parser.add_argument("--config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--phase", choices=("fold0-mode", "development", "stress"), required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--verifier-only-run")
    parser.add_argument("--state-xgb-run")
    parser.add_argument("--no-embedding-run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    if args.phase == "fold0-mode":
        if not args.verifier_only_run or not args.state_xgb_run:
            raise SystemExit("fold0-mode requires --verifier-only-run and --state-xgb-run")
        decision = select_fold0_mode(
            output_root,
            args.verifier_only_run,
            args.state_xgb_run,
            args.no_embedding_run,
        )
    elif args.phase == "development":
        if not args.run_name:
            raise SystemExit("development requires --run-name")
        decision = evaluate_development_gate(input_root, output_root, args.run_name)
    else:
        if not args.run_name:
            raise SystemExit("stress requires --run-name")
        decision = evaluate_stress_gate(input_root, output_root, args.run_name)
    relative_output = Path(args.output)
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise SystemExit("--output must be a relative path inside outputs/v3")
    write_json_atomic(output_root / relative_output, decision)
    print(decision)


if __name__ == "__main__":
    main()
