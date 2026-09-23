from __future__ import annotations

import json

import numpy as np
import torch
import yaml
from xgboost import XGBClassifier

from bme_eating.hierarchical_pipeline import load_hierarchical_bundle
from bme_eating.models.boundary_refiner import BoundaryRefiner
from bme_eating.models.event_verifier import EventVerifier
from bme_eating.models.factory import build_state_model


def test_minimal_hierarchical_bundle_loads(tmp_path) -> None:
    model_config = {
        "architecture": "hierarchical_state",
        "motion_block_seconds": 3,
        "ppg_block_seconds": 15,
        "motion_bucket_counts": [1, 2, 4, 8, 16, 32, 64],
        "ppg_bucket_counts": [1, 2, 4, 8, 16],
        "motion_dilations": [1],
        "ppg_dilations": [1],
        "local_embedding_dim": 4,
        "tcn_channels": 4,
        "state_embedding_dim": 4,
        "dropout": 0.0,
        "future_context_seconds": 0,
        "use_stable_state_features": False,
        "stable_feature_columns": ["feature"],
    }
    verifier_config = {
        "hidden_channels": 8,
        "hidden_dim": 8,
        "dropout": 0.0,
        "left_context_seconds": 60,
        "right_context_seconds": 60,
        "left_bins": 4,
        "event_bins": 16,
        "right_bins": 4,
        "use_state_embedding": False,
    }
    boundary_config = {
        "start_range_seconds": 60,
        "end_range_seconds": 60,
        "coarse_bin_seconds": 30,
        "fine_range_seconds": 15,
        "safety_gap_seconds": 3,
        "hidden_dim": 8,
        "dropout": 0.0,
    }
    config = {
        "model": model_config,
        "hierarchical": {
            "maximum_event_latency_seconds": 60,
            "stable_feature_columns": ["feature"],
        },
        "verifier": verifier_config,
        "boundary": boundary_config,
        "proposals": {},
    }
    (tmp_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    state = build_state_model(model_config)
    for seed in (2026, 2027, 2028):
        torch.save(
            {"model": state.state_dict(), "model_config": model_config},
            tmp_path / f"state_seed_{seed}.pt",
        )
    verifier = EventVerifier(7, 11, verifier_config)
    torch.save(
        {
            "model": verifier.state_dict(),
            "sequence_dim": 7,
            "scalar_dim": 11,
            "config": verifier_config,
        },
        tmp_path / "verifier.pt",
    )
    boundary = BoundaryRefiner(7, 11, boundary_config)
    torch.save(
        {
            "model": boundary.state_dict(),
            "sequence_dim": 7,
            "scalar_dim": 11,
            "config": boundary_config,
        },
        tmp_path / "boundary.pt",
    )
    xgb = XGBClassifier(n_estimators=1, max_depth=1, n_jobs=1, random_state=2026)
    xgb.fit(
        np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        np.asarray([0, 0, 1, 1]),
    )
    xgb.save_model(tmp_path / "xgboost_full.json")
    (tmp_path / "xgboost_full.metadata.json").write_text(
        json.dumps({"feature_columns": ["feature"]}), encoding="utf-8"
    )
    (tmp_path / "xgboost_postprocess.json").write_text("{}", encoding="utf-8")
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "event": {"temperature": 1.0},
                "iou": {"temperature": 1.0},
                "state": {"temperature": 1.0},
                "combiner": {"coefficients": [1.0, 1.0, 1.0], "intercept": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "selected_pipeline.json").write_text(
        json.dumps(
            {
                "acceptance_threshold": 0.5,
                "nms_iou_threshold": 0.5,
                "boundary_entropy_threshold": 0.75,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "duration_bounds.json").write_text(
        json.dumps({"minimum_seconds": 15, "maximum_seconds": 3600}),
        encoding="utf-8",
    )

    detector = load_hierarchical_bundle(tmp_path)

    assert len(detector.state_models) == 3
    assert detector.duration_bounds == (15.0, 3600.0)
    assert detector.xgb_feature_columns == ["feature"]
