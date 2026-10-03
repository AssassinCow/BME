from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    reference_report = reference / "deep_crossfit.json"
    candidate_report = candidate / "deep_crossfit.json"
    reference_scores = reference / "deep_crossfit_scores.parquet"
    candidate_scores = candidate / "deep_crossfit_scores.parquet"
    reference_config = reference / "resolved_config.yaml"
    candidate_config = candidate / "resolved_config.yaml"
    for path in (reference_report, candidate_report, reference_scores, candidate_scores,
                 reference_config, candidate_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    before_config = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    after_config = yaml.safe_load(candidate_config.read_text(encoding="utf-8"))
    if before_config["decoder"].get("candidate_protocol") != "v4.8":
        raise RuntimeError("Raw IMU comparison requires v4.8 candidate protocol")
    if (bool(before_config["verifier"].get("use_raw_imu_branch", False))
            or not bool(after_config["verifier"].get("use_raw_imu_branch", False))):
        raise RuntimeError("Reference must disable raw IMU and candidate must enable it")
    comparable = deepcopy(before_config)
    comparable["verifier"]["use_raw_imu_branch"] = True
    comparable["experiment"]["name"] = after_config["experiment"]["name"]
    if comparable != after_config:
        raise RuntimeError("Raw IMU runs differ in more than the verifier branch")
    before = pd.read_parquet(reference_scores)
    after = pd.read_parquet(candidate_scores)
    keys = ["proposal_id", "subject_key", "session_id", "coarse_start_ms", "coarse_end_ms"]
    if before["proposal_id"].duplicated().any() or after["proposal_id"].duplicated().any():
        raise RuntimeError("Raw IMU comparison requires unique proposal IDs")
    if not before[keys].sort_values("proposal_id").reset_index(drop=True).equals(
        after[keys].sort_values("proposal_id").reset_index(drop=True)
    ):
        raise RuntimeError("Raw IMU comparison requires the identical candidate pool")
    baseline = json.loads(reference_report.read_text(encoding="utf-8"))["deep"]["point"]
    experimental = json.loads(candidate_report.read_text(encoding="utf-8"))["deep"]["point"]
    f1_before = float(baseline["f1"])
    f1_after = float(experimental["f1"])
    fp_before = float(baseline["fp_per_hour"])
    fp_after = float(experimental["fp_per_hour"])
    passed = f1_after >= f1_before + 0.010 or (
        f1_after >= f1_before - 0.005 and fp_after <= fp_before * 0.85
    )
    return {
        "protocol": "v48_raw_imu_identical_candidate_gate_v1",
        "evidence_class": "development_stress_joint_tuning_optimistic",
        "reference_deep_sha256": _sha256(reference_report),
        "current_deep_sha256": _sha256(candidate_report),
        "reference_config_sha256": _sha256(reference_config),
        "current_config_sha256": _sha256(candidate_config),
        "reference_scores_sha256": _sha256(reference_scores),
        "current_scores_sha256": _sha256(candidate_scores),
        "candidate_count": len(before),
        "reference_f1": f1_before,
        "current_f1": f1_after,
        "reference_fp_per_hour": fp_before,
        "current_fp_per_hour": fp_after,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-final-root", type=Path, required=True)
    parser.add_argument("--candidate-final-root", type=Path, required=True)
    arguments = parser.parse_args()
    reference = arguments.reference_final_root.resolve()
    candidate = arguments.candidate_final_root.resolve()
    if reference == candidate:
        raise ValueError("Raw IMU reference and candidate runs must differ")
    output = candidate / "v48_raw_imu_gate.json"
    if output.exists():
        raise FileExistsError("Raw IMU gate already exists; preserve prior evidence")
    report = compare(reference, candidate)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
