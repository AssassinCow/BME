from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from bme_eating.cli import command_audit_multisection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit repeated-header sensor attachments without modifying raw data."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="YAML configuration path.")
    command_audit_multisection(parser.parse_args())


if __name__ == "__main__":
    main()
