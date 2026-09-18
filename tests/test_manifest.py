from pathlib import Path

from bme_eating.data.manifest import _resolve_attachment_path


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
