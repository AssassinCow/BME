from pathlib import Path

from bme_eating.data.manifest import _resolve_attachment_path, build_secure_indices


def test_resolve_attachment_path_falls_back_to_sensor_attachment_tree(tmp_path: Path) -> None:
    expected = tmp_path / "raw" / "sensor_attachments" / "HNU21004" / "sample.zip"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"zip")

    resolved = _resolve_attachment_path(tmp_path, Path("HNU21004") / "sample.zip")

    assert resolved == expected


def test_resolve_attachment_path_accepts_windows_separators(tmp_path: Path) -> None:
    expected = tmp_path / "raw" / "sensor_attachments" / "HNU21004" / "sample.zip"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"zip")

    resolved = _resolve_attachment_path(tmp_path, r"HNU21004\sample.zip")

    assert resolved == expected


def test_empty_download_state_keeps_index_schema(tmp_path: Path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "download_state.json").write_text(
        '{"attachments": []}', encoding="utf-8"
    )
    metadata = tmp_path / "derived" / "valid_metadata"
    metadata.mkdir(parents=True)
    (metadata / "sensor.csv").write_text("externalid\nHNU21001\n", encoding="utf-8")
    (metadata / "meal.csv").write_text(
        "externalid,beforeTime,afterTime,dietaryHand,wearHand,uniqueid\n"
        "HNU21001,1000,2000,left,right,e1\n",
        encoding="utf-8",
    )
    records, events = build_secure_indices(tmp_path, tmp_path / "out", set(), r"^HNU\d{5}$")
    assert list(records.columns) == [
        "subject_key", "uniqueid", "object_path", "zip_path", "zip_sha256", "zip_size_bytes"
    ]
    assert records.empty
    assert len(events) == 1
