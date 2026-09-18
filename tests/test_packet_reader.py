import csv
import io
import json
import zipfile

from bme_eating.data.packet_reader import parse_sensor_zip


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

