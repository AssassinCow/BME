from __future__ import annotations

from bme_eating.config import load_config
from bme_eating.training.dtp_trainer import configured_state_feature_columns


def test_verifier_only_does_not_attach_state_stable_features() -> None:
    config = load_config("configs/hierarchical_v3.yaml")

    assert not config["model"]["use_stable_state_features"]
    assert config["model"]["stable_feature_columns"]
    assert configured_state_feature_columns(config["model"]) == ()
    assert config["training"]["inference_batch_size"] == 64
    assert config["training"]["inference_resume_chunk_rows"] == 32768


def test_state_xgb_attaches_registered_state_features() -> None:
    config = load_config("configs/hierarchical_v3_state_xgb.yaml")

    assert configured_state_feature_columns(config["model"]) == tuple(
        config["model"]["stable_feature_columns"]
    )
