from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.reproducibility import require_git_worktree
from bme_eating.training.hierarchical_final import train_hierarchical_final
from bme_eating.training.hierarchical_trainer import load_hierarchical_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sealed full-data v3 hierarchy")
    parser.add_argument("--config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", choices=("all",), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    train_hierarchical_final(
        config,
        load_hierarchical_inputs(config, input_root, event_role="all"),
        input_root,
        output_root,
        args.run_name,
        fresh=args.fresh,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
