from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import initialize_hierarchical_run
from bme_eating.reproducibility import require_clean_git_worktree
from bme_eating.training.hierarchical_trainer import reuse_hierarchical_state_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse an identical frozen state cache for a verifier-only ablation"
    )
    parser.add_argument("--source-config", default="configs/hierarchical_v3.yaml")
    parser.add_argument("--config", default="configs/hierarchical_v3_no_embedding.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require_clean_git_worktree()
    source_config = load_config(args.source_config)
    target_config = load_config(args.config)
    _, source_input, source_output = resolve_artifact_roots(source_config)
    _, target_input, target_output = resolve_artifact_roots(target_config)
    if source_input != target_input or source_output != target_output:
        raise RuntimeError("Ablation state reuse requires identical artifact roots")
    source = initialize_hierarchical_run(
        source_config,
        source_input,
        source_output,
        args.source_run,
        args.fold,
        fresh=False,
    )
    target = initialize_hierarchical_run(
        target_config,
        target_input,
        target_output,
        args.run_name,
        args.fold,
        fresh=args.fresh,
    )
    if args.resume and target.stage != "CREATED":
        raise RuntimeError("A resumed ablation cache must still be at CREATED stage")
    reuse_hierarchical_state_artifacts(source, target, source_config, target_config)


if __name__ == "__main__":
    main()
