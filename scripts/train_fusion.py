from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_train_fusion
from bme_eating.fusion import FusionGateError


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one causal DTP cross-fit source. Under protocol v4 this command stops "
            "after frozen predictions; tune_fusion.py and evaluate_fusion.py are separate."
        )
    )
    parser.add_argument("--config", default="configs/dtp_fusion.yaml")
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(5),
        help="Outer subject fold: 0, 1, 2, 3, or 4.",
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Isolated experiment directory name. Must start with "
            "baseline_dtp_fusion_."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Create a brand-new named fold-0 run and fail if its experiment "
            "directory already exists."
        ),
    )
    parser.add_argument(
        "--baseline-source-commit",
        help=(
            "Expected Git commit recorded by the baseline manifests. Use the commit "
            "that produced a newly retrained baseline."
        ),
    )
    parser.add_argument(
        "--require-clean-baseline",
        action="store_true",
        help=(
            "Require baseline manifests from a clean, non-backfilled working tree at "
            "the expected commit."
        ),
    )
    parser.add_argument(
        "--resume",
        help=(
            "Resume the whole fusion fold from any crossfit_0..2/last.pt checkpoint; "
            "completed partitions are verified and skipped automatically."
        ),
    )
    parser.add_argument(
        "--confirmation-run",
        help=(
            "Protocol-v4 result run whose fold 1 gate authorizes source training for "
            "folds 2-4."
        ),
    )
    args = parser.parse_args()
    try:
        command_train_fusion(args)
    except FusionGateError as error:
        print(str(error))
        raise SystemExit(error.exit_code) from error


if __name__ == "__main__":
    main()
