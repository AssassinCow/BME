from __future__ import annotations

import runpy
import sys

sys.path.insert(0, r"C:\Users\LZX\.codex\skill-sources\math-agent-33cb044009d2\code")
sys.argv = [
    r"C:\Users\LZX\.codex\skills\research-model-literature\scripts\build_source_record.py",
    "--root",
    "docs/research/statsfusion_r3_literature",
    "--source-candidates",
    "docs/research/statsfusion_r3_frontier_sources.json",
]
runpy.run_path(sys.argv[0], run_name="__main__")
