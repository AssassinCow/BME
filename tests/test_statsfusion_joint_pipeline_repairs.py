from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import load_config
from bme_eating.joint_evaluation import NonnegativePlatt
from bme_eating.joint_stream import FrozenSourceCDF, JointCandidate, TimelineCache
from bme_eating.joint_v411_resources import SearchResourceMonitor
from bme_eating.joint_v411_search import CachedSearch, choose_point, search_group
from bme_eating.joint_v411_stream import make_stream
from bme_eating.training.statsfusion_joint_trainer import boundary_pair_gate, decoder_settings


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/statsfusion_v412_integrated_repair.yaml")


def candidate(identity="candidate", start=0, end=30_000):
    return JointCandidate("subject", "session", start, end, end + 30_000,
        "state", 3, .8, start, end, "family", identity, identity, "hash", "model",
        {"parent_source": "state"}, end + 30_000,
        event_logit=1., predicted_quality=.8, score_formula="calibrated_event_probability_v412")


def usage(private=0):
    return {"process_tree_private_ram_gb": private, "peak_process_tree_private_ram_gb": private,
            "worker_process_count": 0}


def search_fixture(config):
    config = copy.deepcopy(config)
    config["resources"]["search_processes"] = 1
    config["search"].update({"budgets": [20], "nms_ious": [.3],
                             "acceptance_min": .5, "acceptance_max": .5})
    item = candidate()
    truth = pd.DataFrame([{"subject_key": "subject", "session_id": "session", "event_id": "truth",
                           "start_ms": 0, "end_ms": 30_000, "hand_relation": "unknown"}])
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
                            "timestamp_ms": np.arange(1, 21) * 3000, "valid": 1.})
    mapping = FrozenSourceCDF.fit([item], {"subject"})
    return config, [item], truth, truth.iloc[:0], windows, {"model": mapping}


def test_parent_cdf_matches_full_path_without_version_marker(config):
    decoder = next(iter(decoder_settings(config).values()))
    pending = [
        {"source": "proposal_15", "start": 0, "end": 120_000, "score": .5},
        {"source": "state", "start": 30_000, "end": 90_000, "score": .2},
        {"source": "state", "start": 93_000, "end": 120_000, "score": .8},
    ]
    parents = make_stream("subject", "session", "model", {**decoder, "parent_proposals_only": True})
    complete = make_stream("subject", "session", "model", decoder)
    for stream in (parents, complete):
        stream.origin = 0
        stream.pending = copy.deepcopy(pending)
    parent_candidates = parents._ready(150_000)
    variants = complete._ready(150_000)
    assert len({item.proposal_family_id for item in parent_candidates}) == 1
    assert len({item.ancestor_id for item in parent_candidates}) == 3
    assert all(not item.generation_parameters.get("v411") for item in parent_candidates)
    parent_cdf = FrozenSourceCDF.fit(parent_candidates, {"subject"})
    full_cdf = FrozenSourceCDF.fit(variants, {"subject"})
    assert parent_cdf.distributions == full_cdf.distributions
    assert parent_cdf.distributions["state:3"] == [.2, .8]
    low = next(item for item in variants if item.source == "state" and item.raw_score == .2)
    assert parent_cdf.transform(low) == .5
    for item in variants:
        assert parent_cdf.transform(item) == full_cdf.transform(item)


@pytest.mark.parametrize("field", ["subject_key", "session_id", "model_identity"])
def test_parent_cdf_uses_subject_session_model_namespace(field):
    first = candidate()
    second = copy.deepcopy(first)
    setattr(second, field, "other")
    second.raw_score = .2
    subjects = {first.subject_key, second.subject_key}
    assert FrozenSourceCDF.fit([first, second], subjects).distributions["state:3"] == [.2, .8]


@pytest.mark.parametrize("name", ["times.bin", "features.bin", "checksums.jsonl"])
def test_training_cache_rejects_same_size_committed_corruption(tmp_path, name):
    binding = {"subject": "fit", "session": "session", "epoch": 8}
    cache = TimelineCache(tmp_path, binding, 4)
    cache.append(np.array([3000]), np.ones((1, 4), np.float32))
    target = tmp_path / name
    content = bytearray(target.read_bytes())
    content[-2 if name == "checksums.jsonl" else 0] ^= 1
    target.write_bytes(content)
    with pytest.raises(RuntimeError, match="checksum"):
        TimelineCache(tmp_path, binding, 4, 1, cache.checksum)


def test_cache_checkpoint_hash_rejects_rewritten_checksum_evidence(tmp_path):
    binding = {"subject": "fit", "session": "session", "epoch": 8}
    cache = TimelineCache(tmp_path, binding, 4)
    cache.append(np.array([3000]), np.ones((1, 4), np.float32))
    committed = cache.checksum
    (tmp_path / "features.bin").write_bytes(np.full((1, 4), 999., np.float32).tobytes())
    seed = hashlib.sha256(json.dumps({"binding": binding, "width": 4},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    forged = hashlib.sha256(seed.encode() + (1).to_bytes(8, "little")
        + (tmp_path / "times.bin").read_bytes() + (tmp_path / "features.bin").read_bytes()).hexdigest()
    (tmp_path / "checksums.jsonl").write_text(json.dumps({"rows": 1, "sha256": forged}) + "\n")
    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        TimelineCache(tmp_path, binding, 4, 1, committed)


def test_cache_truncates_uncommitted_data_and_checksum_tail(tmp_path):
    binding = {"subject": "fit", "session": "session", "epoch": 8}
    cache = TimelineCache(tmp_path, binding, 4)
    cache.append(np.array([3000]), np.ones((1, 4), np.float32))
    committed = cache.checksum
    cache.append(np.array([6000]), np.zeros((1, 4), np.float32))
    with (tmp_path / "checksums.jsonl").open("ab") as journal:
        journal.write(b"partial-uncommitted-record")
    with (tmp_path / "features.bin").open("ab") as features:
        features.write(b"partial-uncommitted-feature")
    restored = TimelineCache(tmp_path, binding, 4, 1, committed)
    assert (tmp_path / "times.bin").stat().st_size == 8
    assert (tmp_path / "features.bin").stat().st_size == 16
    assert len((tmp_path / "checksums.jsonl").read_text().splitlines()) == 1
    restored.append(np.array([6000]), np.zeros((1, 4), np.float32))
    assert restored.checksum == cache.checksum


def test_cache_rejects_missing_integrity_evidence_or_partial_chunk_cursor(tmp_path):
    binding = {"subject": "fit", "session": "session", "epoch": 8}
    cache = TimelineCache(tmp_path, binding, 4)
    cache.append(np.array([3000, 6000]), np.ones((2, 4), np.float32))
    committed = {name: (tmp_path / name).read_bytes()
                 for name in ("times.bin", "features.bin", "checksums.jsonl")}
    with pytest.raises(RuntimeError, match="checksum"):
        TimelineCache(tmp_path, binding, 4, 1)
    assert all((tmp_path / name).read_bytes() == content for name, content in committed.items())
    TimelineCache(tmp_path, binding, 4, 2, cache.checksum)
    cache = TimelineCache(tmp_path, binding, 4)
    cache.append(np.array([3000]), np.ones((1, 4), np.float32))
    (tmp_path / "checksums.jsonl").unlink()
    with pytest.raises(RuntimeError, match="evidence is missing"):
        TimelineCache(tmp_path, binding, 4, 1, cache.checksum)


@pytest.mark.parametrize("relative", ["data/stats_fusion_sequence.py", "features/baseline.py"])
def test_feature_cache_identity_tracks_low_level_feature_dependencies(config, monkeypatch, relative):
    from bme_eating import joint_v412_cache as module
    root = Path(module.__file__).parent
    altered = {"enabled": False}
    tracked = []

    def source_hash(path):
        name = path.relative_to(root).as_posix()
        tracked.append(name)
        return "changed" if altered["enabled"] and name == relative else "original"

    monkeypatch.setattr(module, "sha256_file", source_hash)
    model = torch.nn.Linear(1, 1)
    inputs = SimpleNamespace(identity={"fixture": "data"})
    before = module.feature_binding(model, inputs, config, {}, "model")
    assert relative in tracked
    altered["enabled"] = True
    after = module.feature_binding(model, inputs, config, {}, "model")
    assert before["feature_source_sha256"] != after["feature_source_sha256"]


def boundary_fixture():
    coarse, refined, truths = [], [], []
    for index in range(60):
        origin = (index + 1) * 1_200_000
        truths.append({"subject_key": "subject", "session_id": "session", "event_id": str(index),
                       "start_ms": origin, "end_ms": origin + 100_000, "hand_relation": "unknown"})
        for kind, start in (("tp", origin - 99_000), ("fp", origin + 198_000)):
            item = candidate(str(index) + "_" + kind, start, origin + 240_000)
            coarse.append(item)
            changed = copy.deepcopy(item)
            changed.unrefined_start_ms, changed.unrefined_end_ms = item.start_ms, item.end_ms
            changed.start_ms = origin + 198_000 if kind == "tp" else origin
            changed.end_ms = item.end_ms if kind == "tp" else item.end_ms - 30_000
            changed.refined, changed.boundary_mode = True, "both"
            changed.boundary_valid_start_bins, changed.boundary_valid_end_bins = 300, 21
            refined.append(changed)
    truth = pd.DataFrame(truths)
    windows = pd.DataFrame({"subject_key": "subject", "session_id": "session",
                            "timestamp_ms": np.arange(1, 24_601) * 3000, "valid": 1.})
    return coarse, refined, truth, windows


def test_boundary_gate_rejects_tp_conversion_even_when_another_candidate_recovers_truth(config):
    coarse, refined, truth, windows = boundary_fixture()
    report = boundary_pair_gate(coarse, refined, truth, truth.iloc[:0], windows, config)
    assert not report["passed"]
    assert report["tp_to_fp_count"] == report["tp_to_fp_denominator"] == 60
    assert report["tp_to_fp_rate"] == 1
    assert report["lost_truth_count"] == 0 and report["truth_retention_rate"] == 1
    assert report["coarse"]["true_positive"] == report["refined"]["true_positive"] == 60
    assert report["improvements"]["start"] >= .1 and report["improvements"]["end"] >= .1


def test_boundary_reports_dropped_predictions_separately_from_false_positives(config):
    coarse, refined, truth, windows = boundary_fixture()
    retained = [item for item in refined if item.proposal_id.endswith("_fp")]
    report = boundary_pair_gate(coarse, retained, truth, truth.iloc[:0], windows, config)
    assert report["passed"]
    assert report["tp_to_fp_count"] == 0
    assert report["dropped_tp_prediction_count"] == report["tp_prediction_loss_count"] == 60
    assert report["lost_truth_count"] == 0


def test_boundary_ignore_predictions_are_not_counted_as_tp_to_fp(config):
    coarse, refined, truth, windows = boundary_fixture()
    ignore = truth.copy()
    ignore["start_ms"] += 198_000
    ignore["end_ms"] = ignore.start_ms + 42_000
    report = boundary_pair_gate(coarse, refined, truth, ignore, windows, config)
    assert report["tp_to_fp_count"] == 0
    assert report["ignored_tp_prediction_count"] == 60
    assert report["refined"]["false_positive"] == 0


def test_search_preparation_rejects_process_tree_over_limit(config, monkeypatch):
    config, pool, truth, ignore, windows, mappings = search_fixture(config)
    monkeypatch.setattr("bme_eating.joint_v411_resources.private_process_tree", lambda *args: usage(27))
    with pytest.raises(RuntimeError, match="Search process tree private RAM"):
        choose_point({"setting": pool}, windows, truth, ignore, mappings, config)


def test_search_worker_checks_root_process_tree(config, monkeypatch):
    config, pool, truth, ignore, windows, _ = search_fixture(config)
    roots = []

    def resource_probe(root_pid):
        roots.append(root_pid)
        return usage()

    monkeypatch.setattr("bme_eating.joint_v411_resources.private_process_tree", resource_probe)
    payload = (pool, truth, ignore, windows[["valid"]], {"setting": "setting", "budget": 20},
               config, NonnegativePlatt(1., 0., True), {"recall": 1}, 0)
    assert len(search_group(payload, root_pid=12345)) == 1
    assert len(roots) >= 3 and set(roots) == {12345}


def test_search_allocation_is_checked_before_large_overlap_matrix(config, monkeypatch):
    config, pool, truth, ignore, windows, _ = search_fixture(config)
    config["resources"]["maximum_private_ram_gb"] = 1
    monkeypatch.setattr("bme_eating.joint_v411_resources.private_process_tree",
                        lambda *args: usage(1 - 1 / 1024 ** 3))

    def forbidden_allocation(*args):
        pytest.fail("Overlap matrix must not allocate after projected RAM gate failure")

    monkeypatch.setattr("bme_eating.joint_v411_search.interval_iou_matrix", forbidden_allocation)
    with pytest.raises(RuntimeError, match="Search process tree private RAM"):
        CachedSearch(pool, truth, ignore, windows, NonnegativePlatt(1., 0., True), SearchResourceMonitor(config))


def test_parallel_search_preserves_registered_order_and_operating_point(config):
    config, pool, truth, ignore, windows, mappings = search_fixture(config)
    config["search"].update({"budgets": [10, 20], "nms_ious": [.3, .5], "acceptance_max": .51})
    sequential, original = choose_point({"setting": pool}, windows, truth, ignore, mappings, config)
    config["resources"]["search_processes"] = 2
    parallel, optimized = choose_point({"setting": pool}, windows, truth, ignore, mappings, config)
    pd.testing.assert_frame_equal(pd.DataFrame(original), pd.DataFrame(optimized), check_exact=True)
    for key in ("setting", "budget", "nms", "acceptance", "f1", "promotion_eligible", "candidate_recall"):
        assert parallel[key] == sequential[key]
    assert parallel["search_resources"]["search_processes"] == 2


def test_reference_search_and_boundary_search_also_enforce_ram(config, monkeypatch):
    from bme_eating.joint_evaluation import choose_operating_point
    from bme_eating.joint_v411_boundary import lock_boundary_modes
    config, pool, truth, ignore, windows, mappings = search_fixture(config)
    config["_reference_execution"] = True
    monkeypatch.setattr("bme_eating.joint_v411_resources.private_process_tree", lambda *args: usage(27))
    with pytest.raises(RuntimeError, match="Search process tree private RAM"):
        choose_operating_point({"setting": pool}, windows, truth, ignore, mappings, config)
    with pytest.raises(RuntimeError, match="Search process tree private RAM"):
        lock_boundary_modes({"candidate_recall": .9}, pool, [], [], truth, ignore, windows, config)
