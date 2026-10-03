from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from bme_eating.proposals_v4 import (
    ProposalSource,
    generate_event_candidates_v4,
    interval_iou,
    observed_hours_v4,
)
from bme_eating.structured_decoder import (
    FixedLagSemiMarkovDecoder,
    TruncatedLogNormalDurationPrior,
)

MECHANISMS = (
    ("jitter", (), (), 60),
    ("backtrack_180", (180,), (), 60),
    ("backtrack_600", (600,), (), 60),
    ("expand_300", (), (300,), 60),
    ("expand_600", (), (600,), 60),
    ("backtrack_expand_gap120", (600,), (600,), 120),
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _covered(proposals: pd.DataFrame, truth: pd.DataFrame) -> set[str]:
    grouped = {
        (str(subject), str(session)): group
        for (subject, session), group in proposals.groupby(
            ["subject_key", "session_id"], sort=False
        )
    }
    covered: set[str] = set()
    for event in truth.itertuples(index=False):
        candidates = grouped.get((str(event.subject_key), str(event.session_id)))
        if candidates is None:
            continue
        if any(
            interval_iou(
                int(event.start_ms), int(event.end_ms),
                int(candidate.coarse_start_ms), int(candidate.coarse_end_ms),
            ) > 0.25
            for candidate in candidates.itertuples(index=False)
        ):
            covered.add(str(event.event_id))
    return covered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    source = arguments.source_root.resolve()
    target = arguments.output_root.resolve()
    if target == source or source in target.parents:
        raise ValueError("Replay output must not be inside the frozen source run")
    if target.exists():
        raise FileExistsError("Replay output already exists; choose a fresh directory")

    frames: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    fold_inputs = []
    for fold in range(5):
        root = source / f"fold_{fold}"
        paths = {
            "windows": root / "outer" / "window_predictions.parquet",
            "truth": root / "evaluation" / "truth_events.parquet",
            "baseline": root / "outer" / "proposals.parquet",
            "prior": root / "decoder" / "duration_prior.json",
            "decoder": root / "decoder" / "selected_decoder.json",
        }
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            source_hashes[f"fold_{fold}/{name}"] = _digest(path)
        windows = pd.read_parquet(paths["windows"])
        truth = pd.read_parquet(paths["truth"])
        if "gyro_valid_fraction" not in windows:
            raise ValueError(f"Fold {fold} windows lack GYRO validity diagnostics")
        session_windows = {
            (str(subject), str(session)): group
            for (subject, session), group in windows.groupby(
                ["subject_key", "session_id"], sort=False
            )
        }
        gyro_missing = []
        for event in truth.itertuples(index=False):
            session = session_windows.get((str(event.subject_key), str(event.session_id)))
            if session is None:
                gyro_missing.append(True)
                continue
            event_windows = session[
                session["timestamp_ms"].between(int(event.start_ms), int(event.end_ms))
            ]
            mean_validity = event_windows["gyro_valid_fraction"].mean()
            gyro_missing.append(
                event_windows.empty or pd.isna(mean_validity) or mean_validity <= 0.0
            )
        truth = truth.assign(gyro_missing=gyro_missing)
        baseline = pd.read_parquet(paths["baseline"])
        prior = TruncatedLogNormalDurationPrior.from_json(
            json.loads(paths["prior"].read_text(encoding="utf-8"))
        )
        base = json.loads(paths["decoder"].read_text(encoding="utf-8"))["decoder"]
        fold_inputs.append((windows, truth, baseline, prior, base))

    for fold, (windows, truth, baseline, prior, base) in enumerate(fold_inputs):
        decoder = FixedLagSemiMarkovDecoder(
            prior,
            grid_seconds=int(base["grid_seconds"]),
            fixed_lag_seconds=int(base["fixed_lag_seconds"]),
        )
        for name, backtrack, expansion, gap in MECHANISMS:
            for high in (0.06, 0.08):
                config = {
                    **base,
                    "candidate_protocol": "v4.8",
                    "high_threshold": high,
                    "low_threshold": 0.04,
                    "ema_half_life_seconds": 6,
                    "gap_merge_seconds": gap,
                    "backtrack_seconds": backtrack,
                    "start_expansion_seconds": expansion,
                    "use_transition_candidates": False,
                }
                proposals = generate_event_candidates_v4(
                    windows, decoder, config, split_role="v48_frozen_outer_replay"
                )
                covered = _covered(proposals, truth)
                baseline_covered = _covered(baseline, truth)
                baseline_without_transition = _covered(
                    baseline.loc[
                        (baseline["source_mask"].astype(int)
                         & int(ProposalSource.TRANSITION)) == 0
                    ], truth,
                )
                relation = truth.get(
                    "hand_relation", pd.Series("unknown", index=truth.index)
                ).fillna("unknown").astype(str)
                frame = truth.assign(covered=truth["event_id"].astype(str).isin(covered))
                baseline_frame = truth.assign(
                    covered=truth["event_id"].astype(str).isin(baseline_covered)
                )
                short = (frame["end_ms"] - frame["start_ms"]) < 120_000
                gyro_missing = frame["gyro_missing"].astype(bool)
                frames.append({
                    "fold": fold,
                    "mechanism": name,
                    "high_threshold": high,
                    "truth_count": len(truth),
                    "covered_count": len(covered),
                    "baseline_covered_count": len(baseline_covered),
                    "baseline_transition_unique_truth": len(
                        baseline_covered - baseline_without_transition
                    ),
                    "candidate_count": len(proposals),
                    "candidates_per_hour": len(proposals) / max(
                        observed_hours_v4(windows), 1e-9
                    ),
                    "same_covered": int(frame.loc[relation.eq("same"), "covered"].sum()),
                    "different_covered": int(frame.loc[relation.eq("different"), "covered"].sum()),
                    "short_covered": int(frame.loc[short, "covered"].sum()),
                    "short_truth_count": int(short.sum()),
                    "gyro_missing_covered": int(frame.loc[gyro_missing, "covered"].sum()),
                    "gyro_missing_truth_count": int(gyro_missing.sum()),
                    "baseline_same_covered": int(
                        baseline_frame.loc[relation.eq("same"), "covered"].sum()
                    ),
                    "baseline_different_covered": int(
                        baseline_frame.loc[relation.eq("different"), "covered"].sum()
                    ),
                    "baseline_short_covered": int(
                        baseline_frame.loc[short, "covered"].sum()
                    ),
                    "baseline_gyro_missing_covered": int(
                        baseline_frame.loc[gyro_missing, "covered"].sum()
                    ),
                })

    results = pd.DataFrame(frames)
    choices = []
    for fold in range(5):
        training = results.loc[results["fold"] != fold]
        ranked = (
            training.groupby(["mechanism", "high_threshold"], as_index=False)
            [["covered_count", "candidate_count"]].sum()
            .sort_values(
                ["covered_count", "candidate_count", "mechanism", "high_threshold"],
                ascending=[False, True, True, True], kind="stable",
            )
        )
        winner = ranked.iloc[0]
        selected = results.loc[
            results["fold"].eq(fold)
            & results["mechanism"].eq(winner.mechanism)
            & results["high_threshold"].eq(winner.high_threshold)
        ].iloc[0]
        choices.append(selected.to_dict())
    target.mkdir(parents=True)
    results.to_parquet(target / "mechanism_grid.parquet", index=False)
    selected_count = sum(int(value["covered_count"]) for value in choices)
    baseline_count = sum(int(value["baseline_covered_count"]) for value in choices)
    relation_gates = {
        relation: sum(int(value[f"{relation}_covered"]) for value in choices)
        >= sum(int(value[f"baseline_{relation}_covered"]) for value in choices) - 2
        for relation in ("same", "different")
    }
    (target / "crossfold_selection.json").write_text(
        json.dumps({
            "evidence_type": "development_stress_joint_replay",
            "selection_rule": "other_four_folds_only",
            "source_sha256": source_hashes,
            "selected_folds": choices,
            "baseline_covered_count": baseline_count,
            "selected_covered_count": selected_count,
            "candidate_gate_passed": selected_count >= baseline_count + 5
            and all(relation_gates.values()),
            "relation_gates": relation_gates,
        }, indent=2, default=int), encoding="utf-8",
    )
    print(results.groupby(["mechanism", "high_threshold"])[
        ["covered_count", "candidate_count"]
    ].sum().to_string())
    print("crossfold covered:", selected_count, "baseline:", baseline_count)


if __name__ == "__main__":
    main()
