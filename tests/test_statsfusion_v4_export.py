from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from bme_eating.hierarchical_v4_export import (
    REQUIRED_MODEL_FILES,
    export_hierarchical_v4_bundle,
)
from bme_eating.hierarchical_v4_pipeline import load_hierarchical_v4_bundle
from bme_eating.models.stats_fusion_state import StatsFusionStateModel
from bme_eating.stats_features import STATS_FEATURE_COLUMNS, FoldRobustScaler


def test_v4_bundle_excludes_xgboost_and_imports_when_it_is_blocked(tmp_path) -> None:
    final_root = tmp_path / "final"
    final_root.mkdir()
    model_config = {
        "architecture": "stats_fusion_state",
        "motion_block_seconds": 3,
        "ppg_block_seconds": 15,
        "motion_dilations": [1],
        "ppg_dilations": [1],
        "statistics_dilations": [1],
        "long_dilations": [1],
        "motion_embedding_dim": 8,
        "ppg_embedding_dim": 8,
        "hidden_dim": 8,
        "statistics_dim": 8,
        "long_pool_factor": 5,
        "dropout": 0.0,
        "gate_initial_bias": -2.0,
        "future_context_seconds": 0,
        "use_motion": True,
        "use_ppg": True,
        "use_statistics": True,
        "use_long_context": True,
        "stable_feature_columns": list(STATS_FEATURE_COLUMNS),
    }
    config = {
        "model": model_config,
        "hierarchical": {"maximum_event_latency_seconds": 60},
        "sequence": {
            "supervised_steps": 8,
            "short_receptive_field_steps": 3,
            "long_receptive_field_tokens": 3,
            "long_pool_factor": 5,
            "step_seconds": 3,
        },
        "decoder": {
            "grid_seconds": 15,
            "fixed_lag_seconds": 60,
            "use_semi_markov": False,
            "ema_half_life_seconds": 12,
            "high_threshold": 0.99,
            "low_threshold": 0.98,
            "gap_merge_seconds": 60,
            "transition_threshold": 0.99,
            "jitter_seconds": [0],
            "maximum_variants_per_event": 1,
            "maximum_candidates_per_hour": 20,
            "deduplication_iou": 0.9,
        },
        "verifier": {
            "left_context_seconds": 60,
            "right_context_seconds": 60,
            "left_bins": 4,
            "event_bins": 16,
            "right_bins": 4,
        },
        "boundary": {"safety_gap_seconds": 3},
    }
    for name in REQUIRED_MODEL_FILES:
        path = final_root / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix in {".yaml", ".yml"}:
            path.write_text("project: {}\n", encoding="utf-8")
        else:
            path.write_bytes(b"model")
    model = StatsFusionStateModel(model_config)
    for parameter in model.parameters():
        parameter.data.zero_()
    for seed in (2026, 2027, 2028):
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": model_config,
                "seed": seed,
                "training_subjects": ["private-subject"],
                "prediction_subjects": [],
                "globally_excluded_subjects": [],
            },
            final_root / f"state_seed_{seed}.pt",
        )
    (final_root / "statistics_scaler.json").write_text(
        json.dumps(
            FoldRobustScaler(
                STATS_FEATURE_COLUMNS,
                np.zeros(12),
                np.ones(12),
                ("private-subject",),
            ).to_json()
        ),
        encoding="utf-8",
    )
    (final_root / "state_calibration.json").write_text(
        json.dumps({"coefficient": 1.0, "intercept": -10.0}), encoding="utf-8"
    )
    (final_root / "sensor_normalization.json").write_text(
        json.dumps(
            {
                "motion_median": [0.0] * 6,
                "motion_iqr": [1.0] * 6,
                "ppg_median": 0.0,
                "ppg_iqr": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (final_root / "duration_prior.json").write_text(
        json.dumps(
            {
                "log_mean": float(np.log(120.0)),
                "log_standard_deviation": 0.5,
                "minimum_seconds": 15.0,
                "maximum_seconds": 600.0,
            }
        ),
        encoding="utf-8",
    )
    (final_root / "resolved_config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (final_root / "selected_pipeline.json").write_text(
        json.dumps(
            {
                "protocol_version": "statsfusion-r2",
                "selection_source": "pooled_outer_oof",
                "state_seeds": [2026, 2027, 2028],
                "verifier_seeds": [2026, 2027, 2028],
                "verifier_kind": "logistic",
                "boundary_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (final_root / "logistic_verifier.json").write_text(
        json.dumps(
            {
                "mean": [0.0] * 190,
                "scale": [1.0] * 190,
                "coefficients": [0.0] * 190,
                "intercept": -10.0,
            }
        ),
        encoding="utf-8",
    )
    (final_root / "final_manifest.json").write_text(
        json.dumps(
            {
                "stage": "COMPLETE",
                "protocol_version": "statsfusion-r2",
                "artifact_hashes": {},
            }
        ),
        encoding="utf-8",
    )
    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    bundle = export_hierarchical_v4_bundle(project_root, final_root, fresh=True, resume=False)
    names = {path.name.lower() for path in bundle.rglob("*") if path.is_file()}
    assert not any("xgboost" in name for name in names)
    exported_scaler = FoldRobustScaler.from_json(
        json.loads((bundle / "statistics_scaler.json").read_text(encoding="utf-8"))
    )
    assert exported_scaler.training_subjects == ()
    for seed in (2026, 2027, 2028):
        checkpoint = torch.load(
            bundle / f"state_seed_{seed}.pt", map_location="cpu", weights_only=False
        )
        assert "training_subjects" not in checkpoint
        assert "prediction_subjects" not in checkpoint
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(bundle / "runtime")
    code = (
        "import builtins\n"
        "import numpy as np\n"
        "import torch\n"
        "original=builtins.__import__\n"
        "def guarded(name,*args,**kwargs):\n"
        "    if name == 'xgboost' or name.startswith('xgboost.'):\n"
        "        raise RuntimeError('xgboost import forbidden')\n"
        "    return original(name,*args,**kwargs)\n"
        "builtins.__import__=guarded\n"
        "from bme_eating.data.stats_fusion_preprocess import RawSessionInput\n"
        "from bme_eating.hierarchical_v4_pipeline import load_hierarchical_v4_bundle\n"
        f"detector=load_hierarchical_v4_bundle({str(bundle)!r},device='cpu')\n"
        "motion_t=np.arange(12000,dtype=np.int64)*10\n"
        "ppg_t=np.arange(6000,dtype=np.int64)*20\n"
        "session=RawSessionInput('s','d',motion_t,np.zeros((len(motion_t),6),np.float32),np.ones((len(motion_t),6),bool),ppg_t,np.zeros(len(ppg_t),np.float32),np.ones(len(ppg_t),bool))\n"
        "events=detector.predict_session(session)\n"
        "assert isinstance(events,list)\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_v4_bundle_loader_fails_closed_without_hash_manifest(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="SHA256SUMS"):
        load_hierarchical_v4_bundle(tmp_path)


def test_v4_export_rejects_incomplete_deep_verifier_seed_mirror(tmp_path) -> None:
    final_root = tmp_path / "final"
    final_root.mkdir()
    for name in REQUIRED_MODEL_FILES:
        path = final_root / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix in {".yaml", ".yml"}:
            path.write_text("project: {}\n", encoding="utf-8")
        else:
            path.write_bytes(b"model")
    (final_root / "selected_pipeline.json").write_text(
        json.dumps(
            {
                "protocol_version": "statsfusion-r2",
                "selection_source": "pooled_outer_oof",
                "state_seeds": [2026, 2027, 2028],
                "verifier_seeds": [2026, 2027, 2028],
                "verifier_kind": "deep",
                "boundary_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (final_root / "proposal_calibration.json").write_text("{}", encoding="utf-8")
    (final_root / "verifier_seed_2026.pt").write_bytes(b"model")
    (final_root / "final_manifest.json").write_text(
        json.dumps(
            {
                "stage": "COMPLETE",
                "protocol_version": "statsfusion-r2",
                "artifact_hashes": {},
            }
        ),
        encoding="utf-8",
    )
    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    with pytest.raises(FileNotFoundError, match="verifier artifacts"):
        export_hierarchical_v4_bundle(project_root, final_root, fresh=True, resume=False)
