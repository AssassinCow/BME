from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from bme_eating.cli import command_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess and evaluate saved probabilities.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML configuration path.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Input prediction Parquet containing state and boundary probabilities.",
    )
    parser.add_argument("--output", required=True, help="Output directory for events and metrics.")
    command_evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

