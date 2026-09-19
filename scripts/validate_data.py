from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.quality import (
    build_quality_report,
    validate_configured_expectations,
    validate_quality_gate,
    validate_quality_invariants,
    write_quality_expectations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen aggregate data gate.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML configuration path.")
    parser.add_argument(
        "--write-expectations",
        action="store_true",
        help="Freeze the currently reviewed aggregate report as the expected v2 dataset.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    report = build_quality_report(output_root)
    validate_configured_expectations(report, config["quality_gates"])
    validate_quality_invariants(report)
    if args.write_expectations:
        path = write_quality_expectations(output_root)
        print(path)
        return
    if (output_root / "indices" / "quality_expectations.json").exists():
        report = validate_quality_gate(output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
