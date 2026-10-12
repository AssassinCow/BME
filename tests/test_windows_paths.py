from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bme_eating.config import _load_project_dotenv, require_environment_path
from bme_eating.prediction_cache import PredictionFeatureCache

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows extended path regression")


def extended_path(path):
    return Path("\\\\?\\" + str(path.resolve()))


def test_extended_output_root_preserves_existing_directory_and_dotenv(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / ".env").write_text(f"BME_OUTPUT_ROOT={extended_path(output)}\n", encoding="utf-8")
    monkeypatch.delenv("BME_OUTPUT_ROOT", raising=False)
    config = {"_config_path": str(configs / "fixture.yaml"),
              "project": {"output_root_env": "BME_OUTPUT_ROOT"}}
    _load_project_dotenv(config)
    resolved = require_environment_path(config, "output_root_env")
    assert str(resolved).startswith("\\\\?\\")
    assert os.path.samefile(resolved, output)


def test_long_prediction_cache_completion_reuse_and_parquet(tmp_path):
    target = extended_path(tmp_path) / ("a" * 90) / ("b" * 90) / ("c" * 90)
    assert len(str(target / "features.bin")) > 300
    binding = {"model": "same", "subject": "fit", "session": "session"}
    cache = PredictionFeatureCache(target, binding)
    times = torch.arange(4, dtype=torch.int64) * 3000
    features = torch.arange(4 * 137, dtype=torch.float32).reshape(4, 137)
    cache.append(times, features)
    cache.finish()
    restored = PredictionFeatureCache(target, binding)
    restored_times, restored_features = restored.read(times.numpy())
    assert restored.reused
    np.testing.assert_array_equal(restored_times.numpy(), times.numpy())
    np.testing.assert_array_equal(restored_features.numpy(), features.numpy())
    restored.finish()
    frame = pd.DataFrame({"subject": ["fit"], "value": [1]})
    parquet = target / "sample.parquet"
    frame.to_parquet(parquet)
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), frame)
    del restored_times, restored_features, restored
    gc.collect()
    shutil.rmtree(extended_path(tmp_path) / ("a" * 90))


@pytest.mark.parametrize("already_extended", [False, True])
def test_powershell_helper_preserves_output_root_and_is_idempotent(tmp_path, already_extended):
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    script = Path(__file__).resolve().parents[1] / "scripts/enable_statsfusion_long_paths.ps1"
    escaped = str(script).replace("'", "''")
    command = (f". '{escaped}' 6>$null; . '{escaped}' 6>$null; "
               "[Console]::WriteLine(($env:BME_OUTPUT_ROOT | ConvertTo-Json -Compress))")
    environment = {**os.environ, "BME_OUTPUT_ROOT": str(extended_path(tmp_path) if already_extended else tmp_path)}
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-Command", command], env=environment, capture_output=True, text=True, check=True)
    resolved = Path(json.loads(result.stdout.strip()))
    assert str(resolved).startswith("\\\\?\\")
    assert os.path.samefile(resolved, tmp_path)
