import csv
import io
import json
import pickle
import zipfile

import pytest

from bme_eating.data.packet_reader import (
    UnsupportedSensorFormatError,
    inspect_ppg_layout,
    parse_sensor_zip,
)


def test_packet_timestamps_are_expanded(tmp_path):
    header = (
        ["ACC_TIME", "PPG_TIME", "GYRO_TIME"]
        + [f"PPG{index}" for index in range(1, 45)]
        + ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    for packet_timestamp in (1000, 1100):
        for row_index in range(2):
            writer.writerow(
                [packet_timestamp, packet_timestamp, packet_timestamp]
                + list(range(1, 21))
                + [0] * 24
                + [row_index, 2, 3, 4, 5, 6]
            )
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("info.json", json.dumps({"files": []}))
        archive.writestr("sensor.txt", buffer.getvalue())
    parsed = parse_sensor_zip(path, ppg_samples_per_row=20)
    assert parsed.acc.values.shape == (4, 3)
    assert parsed.gyro.values.shape == (4, 3)
    assert parsed.ppg.values.shape == (80, 1)
    assert parsed.acc.timestamp_ms[0] == 1000
    assert parsed.acc.timestamp_ms[1] == 1050
    assert parsed.ppg.timestamp_ms[0] == 1000
    assert parsed.ppg.timestamp_ms[1] > parsed.ppg.timestamp_ms[0]


def test_binary_sensor_member_is_reported_without_crashing_audit(tmp_path):
    path = tmp_path / "binary.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("info.json", json.dumps({"metadataType": "SensorOriginalData"}))
        archive.writestr("sensor.txt", b"\x00\xff\x99\x05\x08\x01\x00\x00")

    layout = inspect_ppg_layout(path)
    assert layout["status"] == "unsupported_binary"
    assert layout["text_member"] == "sensor.txt"
    assert layout["rows_scanned"] == 0

    with pytest.raises(UnsupportedSensorFormatError, match="not UTF-8 text"):
        parse_sensor_zip(path)


def test_unsupported_sensor_format_error_is_process_pool_pickleable(tmp_path):
    error = UnsupportedSensorFormatError(tmp_path / "sample.zip", "sensor.txt", "binary")
    restored = pickle.loads(pickle.dumps(error))
    assert restored.zip_path == error.zip_path
    assert restored.member_name == error.member_name
    assert restored.reason == error.reason

