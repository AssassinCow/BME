from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from bme_eating.config import load_config, resolve_artifact_roots
from bme_eating.data.stats_fusion_inputs import prepare_canonical_statsfusion_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the session-wide canonical anchors and 15-second statistics for StatsFusion r2"
    )
    parser.add_argument("--config", default="configs/hierarchical_v4_statsfusion.yaml")
    parser.add_argument("--workers", type=int, default=8)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("experiment", {}).get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("Canonical StatsFusion preparation requires statsfusion-r2")
    _, input_root, output_root = resolve_artifact_roots(config)
    manifest = prepare_canonical_statsfusion_inputs(
        input_root,
        output_root,
        workers=args.workers,
        fresh=args.fresh,
        resume=args.resume,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
