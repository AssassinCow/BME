from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path

import pytest
from tqdm.utils import disp_len

from bme_eating.progress import TerminalProgress, progress


def runner_module(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("joint_progress_runner", root / "scripts/run_statsfusion_joint.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_log_mirror_preserves_terminal_descriptor_encoding_and_carriage_return(monkeypatch):
    runner = runner_module(monkeypatch)

    class Console(io.StringIO):
        encoding = "utf-8"

        def fileno(self):
            return 2

        def isatty(self):
            return True

    console, log = Console(), io.StringIO()
    mirror = runner.LogMirror(console, log)
    assert mirror.fileno() == 2 and mirror.encoding == "utf-8" and mirror.isatty()
    assert mirror.write("\rprogress") == 9
    assert console.getvalue() == log.getvalue() == "\rprogress"


@pytest.mark.parametrize("columns", [60, 80, 100, 120])
def test_progress_updates_stay_in_one_terminal_row(monkeypatch, columns):
    monkeypatch.setattr("bme_eating.progress.shutil.get_terminal_size",
                        lambda **kwargs: os.terminal_size((columns, 24)))
    output = io.StringIO()
    bar = TerminalProgress(total=3410, initial=2075, desc="partition_0/retrain epoch 21/38",
                           file=output, dynamic_ncols=True, ascii=False, colour="#FFFFFF", mininterval=0)
    bar.set_postfix(loss="0.24231", subject="e5076d6e8cbdd20b45eb")
    for _ in range(3):
        bar.update()
        bar.refresh()
    active = output.getvalue()
    assert "\n" not in active
    assert len(active.split("\r")) > 3
    assert all(disp_len(frame) < columns for frame in active.split("\r"))
    assert "#" not in active
    assert "\x1b[38;2;255;255;255m" in active
    bar.close()
    assert output.getvalue().count("\n") == 1


def test_default_progress_uses_white_unicode_blocks(monkeypatch):
    captured = {}

    def record(iterable, **kwargs):
        captured.update(kwargs)
        return iterable

    monkeypatch.setattr("bme_eating.progress.TerminalProgress", record)
    assert list(progress(range(3), total=3, description="selector")) == [0, 1, 2]
    assert captured["ascii"] is False and captured["colour"] == "#FFFFFF"
    assert captured["dynamic_ncols"] and captured["mininterval"] == .5
