import pytest

from bme_eating.config import feature_artifact_name


def test_feature_artifact_name_preserves_legacy_defaults_and_allows_isolation():
    assert feature_artifact_name({"features": {"include_dyadic": False}}) == "baseline"
    assert feature_artifact_name({"features": {"include_dyadic": True}}) == "baseline_dyadic"
    assert (
        feature_artifact_name(
            {
                "features": {
                    "include_dyadic": True,
                    "artifact_name": "baseline_dyadic_lite",
                }
            }
        )
        == "baseline_dyadic_lite"
    )


def test_feature_artifact_name_rejects_paths():
    with pytest.raises(ValueError, match="artifact_name"):
        feature_artifact_name(
            {"features": {"include_dyadic": True, "artifact_name": "../overwrite"}}
        )
