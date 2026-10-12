from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from bme_eating.artifacts import write_json_atomic
from bme_eating.data import stats_fusion_inputs as inputs


def replace_source_hash(identity, name, source_hash):
    identity["feature_implementation"]["files"][name] = source_hash
    files = identity["feature_implementation"]["files"]
    identity["feature_implementation"]["sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    root = tmp_path / "package"
    for name in inputs.FEATURE_IMPLEMENTATION_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"def feature():\n    return 1\n")
    monkeypatch.setattr(inputs, "__file__", str(root / "data" / "stats_fusion_inputs.py"))
    current = {"version": 3, "protocol_version": "statsfusion-r3.2",
               "feature_implementation": inputs.feature_implementation_identity(),
               "source_sha256": {"events": "event-hash", "segments": "segment-hash", "segment_archives": "raw-hash"}}
    return root, tmp_path / "preparation_identity.json", current


@pytest.mark.parametrize("name", inputs.FEATURE_IMPLEMENTATION_FILES)
def test_canonical_reuse_accepts_only_newline_equivalence(source_fixture, name):
    root, identity_path, previous = source_fixture
    write_json_atomic(identity_path, previous)
    path = root / name
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    current = {**previous, "feature_implementation": inputs.feature_implementation_identity()}
    original = identity_path.read_bytes()
    assert inputs._require_matching_preparation_identity(identity_path, current) == previous
    assert identity_path.read_bytes() == original
    assert current["feature_implementation"]["files"][name] != previous["feature_implementation"]["files"][name]


def test_registered_noncanonical_migration_accepts_crlf_checkout(source_fixture, monkeypatch):
    root, identity_path, previous = source_fixture
    name = "data/stats_fusion_preprocess.py"
    original = previous["feature_implementation"]["files"][name]
    new_source = b"def feature():\n    return 2\n"
    new_hash = hashlib.sha256(new_source).hexdigest()
    monkeypatch.setattr(inputs, "NON_CANONICAL_SOURCE_MIGRATIONS", {name: {(original, new_hash)}})
    (root / name).write_bytes(new_source.replace(b"\n", b"\r\n"))
    current = {**previous, "feature_implementation": inputs.feature_implementation_identity()}
    write_json_atomic(identity_path, previous)
    assert inputs._require_matching_preparation_identity(identity_path, current) == previous


@pytest.mark.parametrize("name", list(inputs.LEGACY_MIXED_NEWLINE_SOURCE_HASHES))
def test_known_mixed_newline_identity_is_pinned_to_exact_normalized_source(tmp_path, name):
    current = {"version": 3, "feature_implementation": inputs.feature_implementation_identity()}
    stored = deepcopy(current)
    old_hash = next(iter(inputs.LEGACY_MIXED_NEWLINE_SOURCE_HASHES[name]))
    replace_source_hash(stored, name, old_hash)
    path = tmp_path / "preparation_identity.json"
    write_json_atomic(path, stored)
    assert inputs._require_matching_preparation_identity(path, current) == stored


@pytest.mark.parametrize("changed", ["source", "events", "segments", "segment_archives", "protocol"])
def test_newline_compatibility_never_bypasses_real_changes(source_fixture, changed):
    root, identity_path, previous = source_fixture
    write_json_atomic(identity_path, previous)
    name = "data/deep_dataset.py"
    (root / name).write_bytes((root / name).read_bytes().replace(b"\n", b"\r\n"))
    current = deepcopy(previous)
    if changed == "source":
        (root / name).write_bytes(b"def feature():\r\n    return 999\r\n")
    elif changed == "protocol":
        current["protocol_version"] = "different-protocol"
    else:
        current["source_sha256"][changed] = "changed-data"
    current["feature_implementation"] = inputs.feature_implementation_identity()
    with pytest.raises(RuntimeError, match="source identity changed"):
        inputs._require_matching_preparation_identity(identity_path, current)


def test_legacy_alias_cannot_authorize_different_source(source_fixture):
    root, identity_path, current = source_fixture
    name = "data/deep_dataset.py"
    old_hash = next(iter(inputs.LEGACY_MIXED_NEWLINE_SOURCE_HASHES[name]))
    stored = deepcopy(current)
    replace_source_hash(stored, name, old_hash)
    write_json_atomic(identity_path, stored)
    (root / name).write_bytes(b"def feature():\n    return 999\n")
    current["feature_implementation"] = inputs.feature_implementation_identity()
    with pytest.raises(RuntimeError, match="source identity changed"):
        inputs._require_matching_preparation_identity(identity_path, current)


def test_claimed_current_hash_must_match_actual_source(source_fixture):
    _, identity_path, current = source_fixture
    stored = deepcopy(current)
    replace_source_hash(stored, "data/deep_dataset.py", "0" * 64)
    replace_source_hash(current, "data/deep_dataset.py", "1" * 64)
    write_json_atomic(identity_path, stored)
    with pytest.raises(RuntimeError, match="source identity changed"):
        inputs._require_matching_preparation_identity(identity_path, current)
