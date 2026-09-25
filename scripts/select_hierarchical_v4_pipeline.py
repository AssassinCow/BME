from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_v4_artifacts import initialize_v4_run
from bme_eating.reproducibility import require_git_worktree
from bme_eating.training.hierarchical_v4_trainer import load_v4_inputs, select_v4_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and lock the StatsFusion v4 pipeline")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.fresh:
        raise SystemExit("Selection continues a boundary run; use --resume")
    require_git_worktree()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    run = initialize_v4_run(config, input_root, output_root, args.run_name, args.fold, fresh=False)
    select_v4_pipeline(
        run,
        config,
        load_v4_inputs(config, input_root, fold=args.fold, event_role="outer_train"),
    )


if __name__ == "__main__":
    main()
