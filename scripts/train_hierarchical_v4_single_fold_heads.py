from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.training.hierarchical_v4_single_fold_heads import (
    load_v4_run_for_single_fold_diagnostics,
    record_single_fold_candidate_compatibility,
    train_single_fold_heads_v4,
)
from bme_eating.training.hierarchical_v4_trainer import (
    build_candidates_v4,
    load_v4_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train diagnostic Logistic, Deep verifier, and Boundary heads from one completed "
            "outer fold without pretending they are five-fold pooled evidence"
        )
    )
    parser.add_argument("--config", default="configs/hierarchical_v4_r32_pooled_heads.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    run = load_v4_run_for_single_fold_diagnostics(
        config,
        input_root,
        output_root,
        args.run_name,
        args.fold,
    )
    train_inputs = load_v4_inputs(config, input_root, fold=args.fold, event_role="outer_train")
    if run.stage == "STATE_COMPLETE":
        build_candidates_v4(run, config, train_inputs)
        record_single_fold_candidate_compatibility(run)
    root = train_single_fold_heads_v4(
        run,
        config,
        train_inputs,
        input_root,
        fresh=args.fresh,
        resume=args.resume,
    )
    print(root)


if __name__ == "__main__":
    main()
