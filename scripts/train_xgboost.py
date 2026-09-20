from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_train_xgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one XGBoost outer fold.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help=(
            "Model configuration path. Use baseline_boundary.yaml for the new baseline "
            "or baseline_dyadic_lite.yaml for the gated dyadic candidate."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(5),
        help="Outer subject fold: 0, 1, 2, 3, or 4.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore the XGBoost trial checkpoint and start the search from scratch.",
    )
    parser.add_argument(
        "--feature-ablation",
        choices=("fused", "motion_only", "ppg_only"),
        default=None,
        help="Optional dyadic feature ablation; non-fused runs use a suffixed experiment name.",
    )
    parser.add_argument(
        "--oof-only",
        action="store_true",
        help="Stop after validation OOF calibration; do not evaluate the outer fold.",
    )
    command_train_xgb(parser.parse_args())


if __name__ == "__main__":
    main()

