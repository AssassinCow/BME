from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import initialize_hierarchical_run
from bme_eating.reproducibility import require_git_worktree
from bme_eating.training.hierarchical_trainer import (
    load_hierarchical_inputs,
    train_state_crossfit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v3 hierarchical state crossfit models")
    parser.add_argument("--config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    run = initialize_hierarchical_run(
        config, input_root, output_root, args.run_name, args.fold, fresh=args.fresh
    )
    inputs = load_hierarchical_inputs(
        config, input_root, fold=args.fold, event_role="outer_train"
    )
    train_state_crossfit(run, config, inputs, resume=args.resume)


if __name__ == "__main__":
    main()
