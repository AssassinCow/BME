from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_roots
from bme_eating.data.quality import validate_quality_gate
from bme_eating.reproducibility import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill fingerprint manifests for the already frozen five-fold baseline."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--source-commit", default="3ca55bb")
    args = parser.parse_args()
    config = load_config(args.config)
    _, output_root = resolve_roots(config)
    validate_quality_gate(output_root)
    for fold in range(5):
        fold_dir = output_root / "experiments" / args.experiment / f"fold_{fold}"
        for required in (
            "model.json",
            "metadata.json",
            "validation_predictions.parquet",
            "test_predictions.parquet",
            "selected_postprocess.json",
            "test_metrics.json",
        ):
            if not (fold_dir / required).exists():
                raise FileNotFoundError(
                    f"Frozen baseline artifact is missing: fold_{fold}/{required}"
                )
        path = write_run_manifest(fold_dir, config, output_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["backfilled"] = True
        payload["backfill_scope"] = "fingerprints captured after the frozen run"
        payload["artifact_provenance"] = {
            "claimed_source_commit": args.source_commit,
            "verification": "team-declared; backfill cannot prove original training commit",
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
