from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_roots
from bme_eating.reporting import generate_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize fold metrics and generate a local HTML experiment dashboard."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment directory names under outputs/experiments.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    default_experiment = str(
        config.get("experiment", {}).get(
            "name",
            "baseline_dyadic"
            if config.get("features", {}).get("include_dyadic")
            else "baseline",
        )
    )
    experiments = args.experiments or [default_experiment]
    experiment_roots = {
        name: output_root / "experiments" / name for name in experiments
    }
    output_dir = args.output_dir or output_root / "reports" / "_vs_".join(experiments)
    paths = generate_experiment_report(experiment_roots, output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
