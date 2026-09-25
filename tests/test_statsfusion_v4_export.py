from __future__ import annotations

import json
import os
import subprocess
import sys

from bme_eating.hierarchical_v4_export import (
    REQUIRED_MODEL_FILES,
    export_hierarchical_v4_bundle,
)


def test_v4_bundle_excludes_xgboost_and_imports_when_it_is_blocked(tmp_path) -> None:
    final_root = tmp_path / "final"
    final_root.mkdir()
    for name in REQUIRED_MODEL_FILES:
        path = final_root / name
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix in {".yaml", ".yml"}:
            path.write_text("project: {}\n", encoding="utf-8")
        else:
            path.write_bytes(b"model")
    (final_root / "logistic_verifier.json").write_text("{}", encoding="utf-8")
    (final_root / "final_manifest.json").write_text(
        json.dumps({"stage": "COMPLETE", "artifact_hashes": {}}), encoding="utf-8"
    )
    project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    bundle = export_hierarchical_v4_bundle(
        project_root, final_root, fresh=True, resume=False
    )
    names = {path.name.lower() for path in bundle.rglob("*") if path.is_file()}
    assert not any("xgboost" in name for name in names)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(bundle / "runtime")
    code = (
        "import builtins\n"
        "import numpy as np\n"
        "import torch\n"
        "original=builtins.__import__\n"
        "def guarded(name,*args,**kwargs):\n"
        "    if name == 'xgboost' or name.startswith('xgboost.'):\n"
        "        raise RuntimeError('xgboost import forbidden')\n"
        "    return original(name,*args,**kwargs)\n"
        "builtins.__import__=guarded\n"
        "from bme_eating.calibration_v4 import LogisticScoreCombiner, PlattCalibration\n"
        "from bme_eating.hierarchical_v4_pipeline import HierarchicalEatingDetectorV4\n"
        "from bme_eating.stats_features import FoldRobustScaler, STATS_FEATURE_COLUMNS\n"
        "from bme_eating.structured_decoder import TruncatedLogNormalDurationPrior\n"
        "class Stub(torch.nn.Module):\n"
        "    def forward(self,batch):\n"
        "        shape=batch['statistics'].shape[:2]\n"
        "        high=torch.full(shape,5.0,device=batch['statistics'].device)\n"
        "        zero=torch.zeros(shape,device=batch['statistics'].device)\n"
        "        return {'state_logit':high,'onset_logit':zero,'offset_logit':zero,'ppg_gate':zero,'statistics_gate':zero,'missing_fraction':zero}\n"
        "config={'hierarchical':{'maximum_event_latency_seconds':60},'decoder':{'grid_seconds':15,'fixed_lag_seconds':60,'use_semi_markov':False,'ema_half_life_seconds':12,'high_threshold':0.35,'low_threshold':0.15,'gap_merge_seconds':60,'transition_threshold':0.35,'jitter_seconds':[0],'maximum_variants_per_event':1,'maximum_candidates_per_hour':20,'deduplication_iou':0.9},'verifier':{'left_context_seconds':60,'right_context_seconds':60,'left_bins':4,'event_bins':16,'right_bins':4},'boundary':{'safety_gap_seconds':3}}\n"
        "logistic=LogisticScoreCombiner(tuple([0.0]*190),tuple([1.0]*190),tuple([0.0]*190),10.0)\n"
        "scaler=FoldRobustScaler(STATS_FEATURE_COLUMNS,np.zeros(12),np.ones(12),('s',))\n"
        "prior=TruncatedLogNormalDurationPrior(np.log(120.0),0.5,15.0,600.0)\n"
        "detector=HierarchicalEatingDetectorV4(state_models=[Stub()],state_calibration=PlattCalibration(1.0,0.0),verifier=None,logistic_verifier=logistic,proposal_calibration=None,boundary=None,statistics_scaler=scaler,duration_prior=prior,boundary_range=None,config=config,selection={'verifier_kind':'logistic','acceptance_threshold':0.1,'nms_iou_threshold':0.5,'boundary_enabled':False},device='cpu')\n"
        "steps=100\n"
        "events=detector.predict_session({'timestamp_ms':torch.arange(steps).mul(3000).unsqueeze(0),'statistics':torch.zeros(1,steps,24)},subject_key='s',session_id='d')\n"
        "assert events and events[0].start_ms < events[0].end_ms\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
