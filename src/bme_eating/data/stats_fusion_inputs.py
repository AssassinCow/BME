from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from bme_eating.data.labels import build_statsfusion_session_anchor_index
from bme_eating.features.baseline import build_segment_features
from bme_eating.fusion import ALIGNMENT_KEYS
from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.stats_features import STATS_FEATURE_COLUMNS

CANONICAL_INPUT_DIRECTORY = "canonical_input"
CANONICAL_ANCHORS_FILE = "anchors.parquet"
CANONICAL_STATISTICS_FILE = "statistics.parquet"
CANONICAL_MANIFEST_FILE = "manifest.json"
CANONICAL_PREPARATION_IDENTITY_FILE = "preparation_identity.json"
CANONICAL_ANCHORS_IDENTITY_FILE = "anchors.sha256.json"


def canonical_input_paths(output_root: Path) -> dict[str, Path]:
    root = Path(output_root) / CANONICAL_INPUT_DIRECTORY
    return {
        "root": root,
        "anchors": root / CANONICAL_ANCHORS_FILE,
        "statistics": root / CANONICAL_STATISTICS_FILE,
        "manifest": root / CANONICAL_MANIFEST_FILE,
        "preparation_identity": root / CANONICAL_PREPARATION_IDENTITY_FILE,
        "anchors_identity": root / CANONICAL_ANCHORS_IDENTITY_FILE,
    }


def _segment_archive_identities(segments: pd.DataFrame) -> list[dict[str, str]]:
    identities: list[dict[str, str]] = []
    for row in segments.sort_values(["session_id", "segment_id"], kind="stable").itertuples(
        index=False
    ):
        path = Path(str(row.segment_path))
        if not path.is_file():
            raise FileNotFoundError(f"StatsFusion segment archive is missing: {path}")
        identities.append(
            {
                "session_id": str(row.session_id),
                "segment_id": str(row.segment_id),
                "sha256": sha256_file(path),
            }
        )
    return identities


def _segment_archive_digest(segments: pd.DataFrame) -> str:
    payload = json.dumps(
        _segment_archive_identities(segments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _segment_archive_stat_digest(segments: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in segments.sort_values(["session_id", "segment_id"], kind="stable").itertuples(
        index=False
    ):
        path = Path(str(row.segment_path))
        if not path.is_file():
            raise FileNotFoundError(f"StatsFusion segment archive is missing: {path}")
        stat = path.stat()
        digest.update(str(row.session_id).encode("utf-8"))
        digest.update(str(row.segment_id).encode("utf-8"))
        digest.update(f"{stat.st_size}|{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def _session_cache_path(
    cache_root: Path,
    session_id: str,
    anchors: pd.DataFrame,
    segments: pd.DataFrame,
    archive_sha256: dict[tuple[str, str], str] | None = None,
) -> Path:
    digest = hashlib.sha256()
    digest.update(b"statsfusion-canonical-statistics-v1")
    digest.update(str(session_id).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(anchors, index=False).to_numpy().tobytes())
    for row in segments.sort_values("segment_id", kind="stable").itertuples(index=False):
        path = Path(str(row.segment_path))
        digest.update(str(row.segment_id).encode("utf-8"))
        key = (str(row.session_id), str(row.segment_id))
        content_sha256 = (
            archive_sha256[key]
            if archive_sha256 is not None and key in archive_sha256
            else sha256_file(path)
        )
        digest.update(content_sha256.encode("ascii"))
    return cache_root / f"{session_id}-{digest.hexdigest()}.parquet"


def _preparation_identity(
    segments_path: Path,
    events_path: Path,
    segments: pd.DataFrame,
) -> dict[str, Any]:
    archives = _segment_archive_identities(segments)
    archive_payload = json.dumps(
        archives,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "version": 1,
        "protocol_version": "statsfusion-r2",
        "anchor_semantics": "right_endpoint_half_open",
        "step_seconds": 3,
        "statistics_window_seconds": 15,
        "feature_order": list(STATS_FEATURE_COLUMNS),
        "source_sha256": {
            "segments": sha256_file(segments_path),
            "events": sha256_file(events_path),
            "segment_archives": hashlib.sha256(archive_payload).hexdigest(),
        },
        "segment_archives": archives,
    }


def _require_matching_preparation_identity(
    path: Path,
    current: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "Partial StatsFusion canonical inputs lack preparation_identity.json; "
            "refuse resume and rebuild with --fresh in a clean canonical_input directory"
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != current:
        raise RuntimeError(
            "StatsFusion canonical source identity changed; refuse resume of stale partial inputs"
        )
    return stored


def _verify_anchor_identity(paths: dict[str, Path]) -> None:
    if not paths["anchors_identity"].is_file():
        raise RuntimeError("StatsFusion canonical anchors lack their identity sidecar")
    identity = json.loads(paths["anchors_identity"].read_text(encoding="utf-8"))
    if identity.get("anchors_sha256") != sha256_file(paths["anchors"]):
        raise RuntimeError("StatsFusion canonical anchors differ from their identity sidecar")
    if identity.get("preparation_identity_sha256") != sha256_file(
        paths["preparation_identity"]
    ):
        raise RuntimeError("StatsFusion canonical anchors belong to a different source identity")


def _build_session_statistics(
    anchors: pd.DataFrame, segments: pd.DataFrame, cache_path: str
) -> tuple[str, str | None]:
    target = Path(cache_path)
    if target.is_file():
        return str(target), None
    try:
        first_path = str(segments.sort_values("start_ms", kind="stable").iloc[0].segment_path)
        frame = build_segment_features(
            first_path,
            anchors,
            15,
            False,
            [],
            [],
            context_segments=segments,
        )
        frame = frame[[*ALIGNMENT_KEYS, *STATS_FEATURE_COLUMNS]].copy()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
        return str(target), None
    except Exception as error:  # noqa: BLE001 - worker failures must cross process boundaries
        return str(target), f"{type(error).__name__}: {error}"


def verify_canonical_statsfusion_inputs(
    input_root: Path, output_root: Path
) -> dict[str, Any]:
    paths = canonical_input_paths(output_root)
    required = (
        "anchors",
        "statistics",
        "manifest",
        "preparation_identity",
        "anchors_identity",
    )
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "StatsFusion canonical session-grid inputs are missing: "
            f"{missing}; run prepare_statsfusion_v4_inputs.py first"
        )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != "statsfusion-r2":
        raise RuntimeError("StatsFusion canonical inputs use an incompatible protocol")
    if manifest.get("anchor_semantics") != "right_endpoint_half_open":
        raise RuntimeError("StatsFusion canonical inputs use incompatible anchor semantics")
    source_paths = {
        "segments": Path(input_root) / "indices" / "segments.parquet",
        "events": Path(input_root) / "indices" / "events.parquet",
    }
    segments = pd.read_parquet(source_paths["segments"])
    current_identity = _preparation_identity(
        source_paths["segments"],
        source_paths["events"],
        segments,
    )
    stored_identity = _require_matching_preparation_identity(
        paths["preparation_identity"], current_identity
    )
    if manifest.get("preparation_identity_sha256") != sha256_file(
        paths["preparation_identity"]
    ):
        raise RuntimeError("StatsFusion preparation identity hash differs from its manifest")
    if manifest.get("source_sha256") != stored_identity["source_sha256"]:
        raise RuntimeError("StatsFusion canonical source hashes differ from preparation identity")
    _verify_anchor_identity(paths)
    expected_output = manifest.get("output_sha256", {})
    for name in ("anchors", "statistics", "anchors_identity"):
        if expected_output.get(name) != sha256_file(paths[name]):
            raise RuntimeError(f"StatsFusion canonical artifact changed: {name}")
    return manifest


def prepare_canonical_statsfusion_inputs(
    input_root: Path,
    output_root: Path,
    *,
    workers: int,
    fresh: bool,
    resume: bool,
) -> dict[str, Any]:
    if fresh == resume:
        raise ValueError("Exactly one of fresh or resume must be selected")
    paths = canonical_input_paths(output_root)
    root = paths["root"]
    segments_path = Path(input_root) / "indices" / "segments.parquet"
    events_path = Path(input_root) / "indices" / "events.parquet"
    segments = pd.read_parquet(segments_path)
    events = pd.read_parquet(events_path)
    current_identity = _preparation_identity(segments_path, events_path, segments)
    if paths["manifest"].is_file():
        if fresh:
            raise FileExistsError(f"StatsFusion canonical inputs already exist: {root}")
        return verify_canonical_statsfusion_inputs(input_root, output_root)
    if fresh and root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Partial canonical input directory already exists: {root}; continue with --resume"
        )
    if resume:
        if not root.is_dir():
            raise RuntimeError("StatsFusion --resume requires an existing partial canonical input")
        _require_matching_preparation_identity(paths["preparation_identity"], current_identity)
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(paths["preparation_identity"], current_identity)
    if not paths["anchors"].is_file():
        anchors = build_statsfusion_session_anchor_index(
            segments,
            events,
            output_step_seconds=3,
            output_path=paths["anchors"],
        )
        write_json_atomic(
            paths["anchors_identity"],
            {
                "anchors_sha256": sha256_file(paths["anchors"]),
                "preparation_identity_sha256": sha256_file(
                    paths["preparation_identity"]
                ),
            },
        )
    else:
        _verify_anchor_identity(paths)
        anchors = pd.read_parquet(paths["anchors"])
    cache_root = root / ".cache"
    archive_sha256 = {
        (str(record["session_id"]), str(record["segment_id"])): str(record["sha256"])
        for record in current_identity["segment_archives"]
    }
    jobs: list[tuple[pd.DataFrame, pd.DataFrame, Path]] = []
    grouped_segments = {
        str(session): group.copy()
        for session, group in segments.groupby("session_id", sort=False)
    }
    for session_id, group in anchors.groupby("session_id", sort=False):
        context = grouped_segments[str(session_id)]
        cache_path = _session_cache_path(
            cache_root,
            str(session_id),
            group,
            context,
            archive_sha256,
        )
        jobs.append((group.copy(), context, cache_path))
    cache_paths: list[Path] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_build_session_statistics, group, context, str(cache_path)): str(
                group.iloc[0].session_id
            )
            for group, context, cache_path in jobs
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="StatsFusion canonical statistics"
        ):
            cache_path, error = future.result()
            if error is None:
                cache_paths.append(Path(cache_path))
            else:
                failures.append(f"{futures[future]}: {error}")
    if failures:
        raise RuntimeError(
            f"StatsFusion canonical statistics failed for {len(failures)} sessions: {failures[:5]}"
        )
    statistics = pd.concat(
        (pd.read_parquet(path) for path in sorted(cache_paths)), ignore_index=True
    ).sort_values(ALIGNMENT_KEYS, kind="stable")
    expected_keys = anchors[ALIGNMENT_KEYS].sort_values(ALIGNMENT_KEYS, kind="stable").reset_index(
        drop=True
    )
    actual_keys = statistics[ALIGNMENT_KEYS].reset_index(drop=True)
    if not expected_keys.equals(actual_keys):
        raise RuntimeError("Canonical statistics do not exactly cover the canonical anchor grid")
    temporary = paths["statistics"].with_name(paths["statistics"].name + ".tmp")
    statistics.to_parquet(temporary, index=False)
    temporary.replace(paths["statistics"])
    manifest = {
        "version": 2,
        "protocol_version": "statsfusion-r2",
        "anchor_semantics": "right_endpoint_half_open",
        "anchor_scope": "session",
        "step_seconds": 3,
        "statistics_window_seconds": 15,
        "preparation_identity_sha256": sha256_file(paths["preparation_identity"]),
        "source_sha256": current_identity["source_sha256"],
        "source_diagnostics": {
            "segment_archive_stat_index": _segment_archive_stat_digest(segments),
        },
        "output_sha256": {
            "anchors": sha256_file(paths["anchors"]),
            "statistics": sha256_file(paths["statistics"]),
            "anchors_identity": sha256_file(paths["anchors_identity"]),
        },
        "counts": {
            "anchors": len(anchors),
            "sessions": int(anchors["session_id"].nunique()),
            "subjects": int(anchors["subject_key"].nunique()),
            "state_loss_masked_anchors": int(
                (anchors["state_loss_mask"].fillna(0.0).astype(float) <= 0).sum()
            ),
        },
    }
    write_json_atomic(paths["manifest"], manifest)
    return manifest
