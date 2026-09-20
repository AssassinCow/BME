from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_train_dtp


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one DTP-SQF outer fold on CUDA.")
    parser.add_argument(
        "--config",
        default="configs/dtp_sqf.yaml",
        help="Causal or future-context DTP-SQF configuration path.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(5),
        help="Outer subject fold: 0, 1, 2, 3, or 4.",
    )
    parser.add_argument(
        "--resume",
        help="Optional checkpoint path from which to resume training.",
    )
    command_train_dtp(parser.parse_args())


if __name__ == "__main__":
    main()

