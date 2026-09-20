from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_smoke_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic DTP-SQF CUDA forward pass.")
    parser.add_argument(
        "--config",
        default="configs/dtp_sqf.yaml",
        help="Causal or future-context DTP-SQF configuration path.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size. Default: 1.")
    command_smoke_model(parser.parse_args())


if __name__ == "__main__":
    main()

