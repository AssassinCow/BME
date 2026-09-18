from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from bme_eating.cli import command_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, segment, and resample all sensor ZIPs.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML configuration path.")
    parser.add_argument(
        "--workers",
        type=int,
        help="Worker process count. Defaults to preprocess.workers in the YAML file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate segment caches that already exist.",
    )
    command_preprocess(parser.parse_args())


if __name__ == "__main__":
    main()

