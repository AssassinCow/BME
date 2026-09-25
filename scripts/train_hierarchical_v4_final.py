from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.reproducibility import require_git_worktree
from bme_eating.training.hierarchical_v4_trainer import train_hierarchical_final_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final StatsFusion v4 models")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_git_worktree()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    train_hierarchical_final_v4(
        config,
        input_root,
        output_root,
        args.run_name,
        fresh=args.fresh,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
