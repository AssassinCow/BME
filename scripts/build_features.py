from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from bme_eating.cli import command_build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local or dyadic XGBoost features.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Feature configuration path. Default: configs/baseline.yaml.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Worker process count. Defaults to features.workers in the YAML file.",
    )
    command_build_features(parser.parse_args())


if __name__ == "__main__":
    main()

