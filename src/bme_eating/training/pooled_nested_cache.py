from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LEGACY_TRAINER_SHA256 = "d20589c2fce6b046591419bfa9cbe0318f01a8a5a8c11ed322ac5bc037db43dc"
FIXED_TRAINER_LF_SHA256 = "ecef0d8ea255aa4052a6350b1afdd5be36f53759c526c582b5ca30c3430f44ee"


def compatible_postprocess_cache(
    saved: dict[str, Any], current: dict[str, Any], saved_sha256: str | None
) -> bool:
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    if saved.get("trainer_sha256") != LEGACY_TRAINER_SHA256:
        return False
    trainer_source = Path(__file__).with_name("hierarchical_v4_trainer.py").read_bytes()
    if hashlib.sha256(trainer_source).hexdigest() != current.get("trainer_sha256"):
        return False
    if hashlib.sha256(trainer_source.replace(b"\r\n", b"\n")).hexdigest() != (
        FIXED_TRAINER_LF_SHA256
    ):
        return False
    calculated = hashlib.sha256(json.dumps(saved, sort_keys=True).encode("utf-8")).hexdigest()
    if calculated != saved_sha256:
        return False
    return {key: value for key, value in saved.items() if key != "trainer_sha256"} == {
        key: value for key, value in current.items() if key != "trainer_sha256"
    }
