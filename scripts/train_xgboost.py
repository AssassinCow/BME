from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from bme_eating.cli import command_train_xgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one XGBoost outer fold.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Model configuration path. Use baseline_dyadic.yaml for dyadic features.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(5),
        help="Outer subject fold: 0, 1, 2, 3, or 4.",
    )
    command_train_xgb(parser.parse_args())


if __name__ == "__main__":
    main()

