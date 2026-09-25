# StatsFusion-r2 运行手册

## 边界

- 输入只读复用 `outputs/v2/`，输出只写 `outputs/v4/`。
- 旧实现统一标记为 `statsfusion-r0-blocked` / `statsfusion-r1-blocked`；r2 必须使用
  全新 run name，禁止 resume 任何旧 v4 产物。
- v4 不加载 XGBoost 模型，不读取 XGBoost 概率、事件或候选，也不使用蒸馏目标。
- 唯一保留的是配置中固定顺序的 12 项 trailing 15 秒统计特征。
- `strict_resume_identity` 固定为 `true`；代码、配置、输入或 scaler 改变后必须使用新 run name。
- folds 2–4 需要先写入 `freeze_manifest.json`。
- 训练阶段对 outer-test anchors 只读取时间轴与文件定位列，并注入零值占位；真实窗口/事件标签
  只由 `evaluate_hierarchical_v4.py` 在 `SELECTED` 阶段读取。

## 首次审计

```powershell
python scripts/prepare_statsfusion_v4_inputs.py `
  --config configs/hierarchical_v4_statsfusion.yaml `
  --workers 8 `
  --fresh

python scripts/audit_statsfusion_feature_provenance.py `
  --config configs/hierarchical_v4_statsfusion.yaml `
  --fresh
```

第一条命令会重新生成 session-wide 3 秒右端点网格和同相位 trailing 15 秒统计特征，写入
`outputs/v4/canonical_input/`。正式 r2 训练会校验 events/segments、每个 segment archive
完整 SHA-256、确定性聚合 SHA、preparation identity、anchor sidecar 与全部输出 SHA-256；
缺少该阶段或仍使用逐 segment 重启相位的旧输入时直接拒绝启动。中断后使用 `--resume`
复用按 archive 内容寻址的逐 session 缓存；源身份或 anchor sidecar 不一致时拒绝复用，
不得回退到 `outputs/v2/` 的旧相位 anchors。
当前数据的只读 v2 快照仍有 1,117 个 masked anchors；按真实原始首末时间重建后的 canonical
快照为 1,581,724 个 anchors，其中 1,125 个 masked。两者差异来自相位与尾端网格修正，
manifest 会记录实际计数，训练和 outer calibration 统一使用 canonical 计数，不硬编码任一数字。

当前配置按保守规则把 12 项特征视为查看过五折 gain 后固定，因此 folds 0–4 只能作为
development/stress evidence；真正独立证据来自官方隐藏测试或新增受试者。

## 单折顺序

```powershell
$run = "hierarchical_v4_statsfusion_r2_20260925a"
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

候选阶段会在 ECE、Brier、预测正例率或纯深度候选召回未过门禁时主动停止。duration
prior 只使用 `evaluable=True` 的 truth，valid ignore 不进入持续时间拟合。boundary 每个
训练分区不足 60 个独立事件时自动禁用，保留 coarse boundary，不会用少样本强行训练。

状态阶段同时保存 `oof/gate_diagnostics.parquet` 和 JSON 汇总，逐受试者记录 PPG、统计与
长上下文 gate 的均值、标准差和分位数。若统计 gate 的受试者均值落到 `<=0.01` 或
`>=0.99`，审计文件会标出塌缩风险，但不会擅自放宽门禁。

verifier/boundary 使用 `fully_nested_v1` meta-crossfit：每个 meta holdout 都会在其余受试者
内重新执行三折、三 seed state crossfit，meta holdout 不得出现在 scaler、state、Platt、
duration prior、verifier、calibrator、boundary 或阈值选择的训练集合中。该协议的 state
训练成本约为普通 outer crossfit 的 3 倍；nested cache 只在 subject 集合、配置、seed、epoch、
父 artifact 哈希和全部 inner lineage 均一致时复用，任何不一致都要求更换全新 run name。

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

先完成 S0、S1、S2、S3 与 PPG-only。S3 相对 S2 只有满足以下任一路径才允许 S4 使用
PPG：F1 至少提高 0.005；或候选召回至少提高 0.010 且 F1 下降不超过 0.005；同时
FP/h 不超过 1.05 倍、异侧召回下降不超过 0.02。若未通过，S4 必须将 `model.use_ppg`
设为 `false`，即从 S2 状态路径加 semi-Markov。不要手工改配置，使用：

```powershell
$config = (& python scripts/prepare_hierarchical_v4_s4.py `
  --config configs/hierarchical_v4_statsfusion.yaml `
  --run-name $run --s2-run $s2Run --s3-run $s3Run --fresh | Select-Object -Last 1)
```

该命令会哈希锁定 S2/S3 指标，自动写入 `model.use_ppg` 和晋级证据。后续 S4 的所有 fold、
freeze、stress、final 与 export 命令都必须使用这份 `$config`；门禁代码仍会拒绝与决定不一致
的 S4。

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
它还会锁定当前 Git worktree、resolved config、v2 输入、五折 raw logits、proposal score、
truth/ignore 哈希以及 state/verifier 三 seed checkpoint；任何父证据变化都会拒绝 resume。

最终状态模型使用 `2026/2027/2028` 三个 seed，只平均 state/onset/offset logits。bundle
中的 deep verifier 同样使用三个 seed 并平均 event/IoU logits。最终 Platt、proposal
calibration、logistic verifier、acceptance/NMS 与 boundary 参数都从 pooled outer OOF
重新拟合和选择，不复制 fold 0。bundle 包含 scaler、校准器、duration prior、胜出
verifier、可选 endpoint refiner 和哈希清单，
不包含原始数据、标签、个人绝对路径、凭据或任何 XGBoost 工件。

## 验收

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m compileall -q src scripts
python scripts/smoke_test_hierarchical_v4.py --config configs/hierarchical_v4_statsfusion.yaml
python scripts/replay_hierarchical_v4_raw_session.py `
  --config configs/hierarchical_v4_statsfusion.yaml
```

真实 replay 默认选择 fragment 数最多的完整 session，要求训练 dataset 与 raw-session
preprocessor 的 motion、motion validity、PPG、PPG quality、PPG validity、PPG-to-motion
映射、长上下文 block endpoint 索引和 24 维统计输入逐项最大误差均不超过 `1e-6`。
随后还必须在禁止 `import xgboost` 的环境中执行 bundle
replay；官方文件格式目前仍为 `UNKNOWN`，这里只冻结原始数组接口，不伪造官方 adapter。

当前 RTX 4080 Laptop bf16 smoke 在 batch size 4、gradient accumulation 4、895 步输入下
峰值约 `1.277 GB`，重复推理最大概率误差为 `0.0`。正式训练若环境变化后超过 10.5 GB，
只降低 batch size 并等比例提高 gradient accumulation，保持有效 batch 为 16。
