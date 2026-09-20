from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from bme_eating.cli import command_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Python, CUDA, and RTX 4080 availability.")
    arguments = parser.parse_args()
    command_environment(arguments)


if __name__ == "__main__":
    main()

