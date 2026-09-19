from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from bme_eating.cli import command_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build secure indices and audit the PPG layout.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML configuration path.")
    parser.add_argument(
        "--schema-zips",
        default="40",
        help="Number of ZIP files to inspect, or 'all'. Default: 40.",
    )
    parser.add_argument(
        "--maximum-rows",
        type=int,
        default=100_000,
        help="Maximum text rows inspected per ZIP. Default: 100000.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any schema audit checkpoint and start from the first ZIP.",
    )
    command_audit(parser.parse_args())


if __name__ == "__main__":
    main()

