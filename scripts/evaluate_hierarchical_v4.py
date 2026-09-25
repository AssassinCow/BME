from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_v4_artifacts import initialize_v4_run
from bme_eating.reproducibility import require_git_worktree
from bme_eating.training.hierarchical_v4_trainer import evaluate_v4_outer, load_v4_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and evaluate one locked v4 outer fold")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.fresh:
        raise SystemExit("Outer evaluation continues a locked run; use --resume")
    require_git_worktree()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    run = initialize_v4_run(config, input_root, output_root, args.run_name, args.fold, fresh=False)
    run.require_stage("SELECTED")
    evaluate_v4_outer(
        run,
        config,
        load_v4_inputs(
            config,
            input_root,
            fold=args.fold,
            event_role="outer_test",
            allow_outer_labels=True,
        ),
    )


if __name__ == "__main__":
    main()
