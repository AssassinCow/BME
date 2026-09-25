# StatsFusion v4 运行手册

## 边界

- 输入只读复用 `outputs/v2/`，输出只写 `outputs/v4/`。
- v4 不加载 XGBoost 模型，不读取 XGBoost 概率、事件或候选，也不使用蒸馏目标。
- 唯一保留的是配置中固定顺序的 12 项 trailing 15 秒统计特征。
- `strict_resume_identity` 固定为 `true`；代码、配置、输入或 scaler 改变后必须使用新 run name。
- folds 2–4 需要先写入 `freeze_manifest.json`。
- 训练阶段对 outer-test anchors 只读取时间轴与文件定位列，并注入零值占位；真实窗口/事件标签
  只由 `evaluate_hierarchical_v4.py` 在 `SELECTED` 阶段读取。

## 首次审计

```powershell
python scripts/audit_statsfusion_feature_provenance.py `
  --config configs/hierarchical_v4_statsfusion.yaml `
  --fresh
```

当前配置按保守规则把 12 项特征视为查看过五折 gain 后固定，因此 folds 0–4 只能作为
development/stress evidence；真正独立证据来自官方隐藏测试或新增受试者。

## 单折顺序

```powershell
$run = "hierarchical_v4_statsfusion_20260927a"
$fold = 0
$config = "configs/hierarchical_v4_statsfusion.yaml"

python scripts/train_hierarchical_v4_state.py --config $config --run-name $run --fold $fold --fresh
python scripts/build_event_candidates_v4.py --config $config --run-name $run --fold $fold --resume
python scripts/train_event_verifier_v4.py --config $config --run-name $run --fold $fold --resume
python scripts/train_boundary_refiner_v4.py --config $config --run-name $run --fold $fold --resume
python scripts/select_hierarchical_v4_pipeline.py --config $config --run-name $run --fold $fold --resume
python scripts/evaluate_hierarchical_v4.py --config $config --run-name $run --fold $fold --resume
```

阶段固定为：

```text
CREATED -> STATE_COMPLETE -> PROPOSALS_COMPLETE -> VERIFIER_COMPLETE
        -> BOUNDARY_COMPLETE -> SELECTED -> EVALUATED
```

候选阶段会在 ECE、Brier、预测正例率或纯深度候选召回未过门禁时主动停止。boundary 每个
训练分区不足 60 个独立事件时自动禁用，保留 coarse boundary，不会用少样本强行训练。

状态阶段同时保存 `oof/gate_diagnostics.parquet` 和 JSON 汇总，逐受试者记录 PPG、统计与
长上下文 gate 的均值、标准差和分位数。若统计 gate 的受试者均值落到 `<=0.01` 或
`>=0.99`，审计文件会标出塌缩风险，但不会擅自放宽门禁。

## fold 0 消融

每个消融使用独立 run name，依次运行“状态训练 + 候选生成”；配置关系为：

| 代号 | 配置 | 变化 |
|---|---|---|
| S0 | `configs/hierarchical_v4_s0.yaml` | motion-only 短上下文 |
| S1 | `configs/hierarchical_v4_s1.yaml` | S0 + 统计门控 |
| S2 | `configs/hierarchical_v4_s2.yaml` | S1 + 1905 秒长上下文 |
| S3 | `configs/hierarchical_v4_s3.yaml` | S2 + PPG |
| S4 | `configs/hierarchical_v4_statsfusion.yaml` | S3 + semi-Markov |
| PPG-only | `configs/hierarchical_v4_ppg_only.yaml` | 独立模态诊断 |

所有 run 完成 `PROPOSALS_COMPLETE` 后执行：

```powershell
python scripts/check_hierarchical_v4_gates.py `
  --config configs/hierarchical_v4_statsfusion.yaml `
  --mode ablation `
  --run-name $run `
  --s0-run $s0Run --s1-run $s1Run --s2-run $s2Run --s3-run $s3Run `
  --ppg-only-run $ppgOnlyRun
```

该命令锁定统计分支、长上下文、纯深度候选和两类佩戴手门禁，并写入
`ablation/fold_0_report.json`。PPG 部分同时报告 motion-only、PPG-only、motion+PPG，
没有正向证据时不得在报告中声称 PPG 带来增益。

## 冻结与最终模型

S4 与 S0 完成 folds 0–1 的一次性 outer 评价后，先运行开发门禁；若存在可用 v3 基线，
同时传入 `--v3-run`，每折自动选择 S0/v3 中更强者：

```powershell
python scripts/check_hierarchical_v4_gates.py `
  --config $config --mode development --run-name $run --s0-run $s0Run
```

folds 0–1 通过开发门禁后：

```powershell
python scripts/freeze_hierarchical_v4_protocol.py --config $config --run-name $run --fresh
```

冻结命令会重新验证 fold 0 消融报告、folds 0–1 的 `EVALUATED` 状态、开发门禁和对应哈希；
任一证据缺失或失败都会阻止 folds 2–4。

完成冻结的 folds 2–4 后，先运行压力门禁：

```powershell
python scripts/check_hierarchical_v4_gates.py `
  --config $config --mode stress --run-name $run --s0-run $s0Run
```

完成并评价 folds 0–4 后：

```powershell
python scripts/train_hierarchical_v4_final.py --config $config --run-name $run --fresh
python scripts/export_hierarchical_v4_bundle.py --config $config --run-name $run --fresh
```

最终训练会再次要求 `stress_gate.json` 为 PASS，失败时不会读取全量标签或启动训练。

最终状态模型使用 `2026/2027/2028` 三个 seed，只平均 state/onset/offset logits。bundle
包含 scaler、校准器、duration prior、胜出 verifier、可选 endpoint refiner 和哈希清单，
不包含原始数据、标签、个人绝对路径、凭据或任何 XGBoost 工件。

## 验收

```powershell
python -m pytest tests/test_statsfusion_v4.py tests/test_statsfusion_v4_config.py `
  tests/test_statsfusion_v4_gates.py tests/test_statsfusion_v4_export.py
python -m ruff check .
python scripts/smoke_test_hierarchical_v4.py --config configs/hierarchical_v4_statsfusion.yaml
```

当前 RTX 4080 Laptop bf16 smoke 在 batch size 4、gradient accumulation 4、891 步输入下
峰值约 `1.269 GB`，重复推理最大概率误差为 `0.0`。正式训练若环境变化后超过 10.5 GB，
只降低 batch size 并等比例提高 gradient accumulation，保持有效 batch 为 16。
