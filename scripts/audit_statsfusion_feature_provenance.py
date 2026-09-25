from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.hierarchical_artifacts import write_json_atomic
from bme_eating.stats_features import audit_feature_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the provenance of the 12 v4 statistics")
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--run-name", default="statsfusion-v4-provenance")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    _, input_root, output_root = resolve_artifact_roots(config)
    target = output_root / "feature_provenance.json"
    if target.exists() and args.fresh:
        raise FileExistsError(target)
    if not target.exists() and args.resume:
        raise FileNotFoundError(target)
    provenance = audit_feature_provenance(
        project_root=Path(__file__).resolve().parents[1],
        input_root=input_root,
        source_paths=config["feature_provenance"]["source_paths"],
        selection_note_path=config["feature_provenance"].get("selection_note_path"),
        assumed_used_all_outer_folds=bool(
            config["feature_provenance"].get("assumed_used_all_outer_folds", True)
        ),
    )
    if target.exists():
        import json

        if json.loads(target.read_text(encoding="utf-8")) != provenance:
            raise RuntimeError("Existing feature provenance differs from the current audit")
    else:
        write_json_atomic(target, provenance)


if __name__ == "__main__":
    main()
