# 2026 生医工进食检测：StatsFusion v4.7.1

本仓库当前只支持 `statsfusion-r3.2` 主线。正式配置是
`configs/hierarchical_v4_r32_pooled_heads_early_select.yaml`，公共推理入口是
`HierarchicalEatingDetectorV4.predict_session()`。

旧 baseline、DTP fusion、hierarchical v3、StatsFusion r0/r1/r2/r3/r3.1 均已停止维护，
不得用旧 run 或 checkpoint 恢复当前训练。历史实验产物位于外部输出目录，不属于本仓库清理范围。

## 环境

- Python `3.11` 或 `3.12`
- 训练建议使用 CUDA 与 bf16；CPU 可运行测试和合成 smoke
- 原始数据路径由 `BME_DATA_ROOT` 提供
- 派生产物根目录由 `BME_OUTPUT_ROOT` 提供
- `.env`、原始数据、checkpoint 和实验输出不得提交

```powershell
conda activate bme-model
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

## 数据准备

已有冻结 v2 输入时，直接构建并核验 r3.2 canonical 输入：

```powershell
$config = "configs/hierarchical_v4_r32_pooled_heads_early_select.yaml"
python scripts/prepare_statsfusion_v4_inputs.py --config $config --workers 8 --resume
python scripts/audit_statsfusion_feature_provenance.py --config $config --fresh
```

若需要从原始下载数据重建 v2 输入，依次执行：

```powershell
python scripts/audit_data.py --config configs/base.yaml --schema-zips all --maximum-rows 1000000000 --workers 8
python scripts/audit_multisection.py --config configs/base.yaml --workers 8
python scripts/preprocess_data.py --config configs/base.yaml --workers 8 --overwrite
python scripts/validate_data.py --config configs/base.yaml
```

未知官方输入格式不得自行假设；当前冻结接口只接受 `statsfusion-raw-v2` 原始数组结构。

## 正式训练

```powershell
$config = "configs/hierarchical_v4_r32_pooled_heads_early_select.yaml"
$run = "hierarchical_v4_r32_pooled_heads_early_select_20260928a"

foreach ($fold in 0..4) {
  python scripts/train_hierarchical_v4_state.py --config $config --run-name $run --fold $fold --fresh
  python scripts/build_event_candidates_v4.py --config $config --run-name $run --fold $fold --resume
  python scripts/select_hierarchical_v4_pipeline.py --config $config --run-name $run --fold $fold --resume
  python scripts/evaluate_hierarchical_v4.py --config $config --run-name $run --fold $fold --resume
}

python scripts/train_hierarchical_v4_final.py --config $config --run-name $run --fresh
python scripts/export_hierarchical_v4_bundle.py --config $config --run-name $run --fresh
```

中断后只对同一 run 使用 `--resume`；不要同时传入 `--fresh` 和 `--resume`。Git commit、
dirty worktree 或运行时代码 SHA 变化不会阻断续跑，也不要求先提交；变化会记录到 manifest，
因此该 run 属于混合源码续跑证据。结果相关配置、模型张量结构、数据、subject folds、
truth/ignore 或协议变化仍必须换新 run name，checkpoint 不兼容时加载会立即失败。

完整说明见 `docs/hierarchical_v4_statsfusion_runbook.md`，统计特征证据边界见
`docs/statsfusion_feature_provenance.md`。

## 验收

```powershell
python -m pytest -q
python -m ruff check src tests scripts --no-cache
python -m compileall -q src scripts
git diff --check
python scripts/smoke_test_hierarchical_v4.py --config configs/hierarchical_v4_r32_pooled_heads_early_select.yaml
python scripts/replay_hierarchical_v4_raw_session.py --config configs/hierarchical_v4_r32_pooled_heads_early_select.yaml
```

正式结论必须同时保留数据哈希、配置、代码身份、随机种子、受试者划分、F1、边界 MAE、
同侧/异侧结果和局限性。当前五折结果属于 development/stress evidence，不宣称独立外部验证。
