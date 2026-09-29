import pytest

from bme_eating.config import feature_artifact_name


def test_feature_artifact_name_uses_current_default_and_allows_isolation():
    assert feature_artifact_name({"features": {}}) == "baseline"
    assert feature_artifact_name({"features": {"artifact_name": "statsfusion"}}) == (
        "statsfusion"
    )


def test_feature_artifact_name_rejects_paths():
    with pytest.raises(ValueError, match="artifact_name"):
        feature_artifact_name(
            {"features": {"artifact_name": "../overwrite"}}
        )
