# 分层进食检测 v3 运行手册（2026-09-23）

## 1. 范围与硬边界

v3 实现“状态识别 → 候选事件 → 事件验证 → 边界精修”。它只读复用 `outputs/v2`
中的清洗数据、anchors、fold map、基础特征和冻结 XGBoost 预测；所有新产物写入
`outputs/v3`。正式命令要求 Git 工作树干净，开始训练前应先提交本次代码。

主指标固定为事件级 F1，匹配条件严格为 `IoU > 0.25`。所有状态、verifier、校准、
阈值和边界选择均按受试者交叉拟合。只有 `evaluate_hierarchical.py` 能读取 outer 标签。
folds 2–4 必须先生成不可变 `freeze_manifest.json`，且只能称为 frozen stress folds。

用户在后台确认的作品截止日期为 2026-10-20；具体截止时刻仍需登录官网截图归档。
官方预测文件字段和执行接口仍未确认，因此最终 bundle 暂不声称符合官方接口。

## 2. 代码验收

```powershell
conda activate bme-model
python scripts/check_environment.py
python -m pytest -q
python -m ruff check src scripts tests
python scripts/smoke_test_hierarchical.py --config configs/hierarchical_v3.yaml --batch-size 16
python scripts/smoke_test_hierarchical.py --config configs/hierarchical_v3_state_xgb.yaml --batch-size 16
git status --short
```

最后一条必须无输出。GPU smoke 同时检查 bf16 forward/backward、峰值显存不超过
10.5 GiB，以及同一模型重复推理误差不超过 `1e-6`。本机 2026-09-23 实测两种模式
batch 16 峰值约 1.24 GB，重复推理误差为 0。

## 3. 单折阶段命令

以下六条命令构成一折完整链路。`--fresh` 只用于创建状态阶段；其余阶段必须
`--resume`，并通过 `run-name/fold` 定位上游产物。

```powershell
$config = "configs/hierarchical_v3.yaml"
$run = "hierarchical_v3_verifier_only_20260927a"
$fold = 0

python scripts/train_hierarchical_state.py --config $config --run-name $run --fold $fold --fresh
python scripts/build_event_candidates.py --config $config --run-name $run --fold $fold --resume
python scripts/train_event_verifier.py --config $config --run-name $run --fold $fold --resume
python scripts/train_boundary_refiner.py --config $config --run-name $run --fold $fold --resume
python scripts/select_hierarchical_pipeline.py --config $config --run-name $run --fold $fold --resume
python scripts/evaluate_hierarchical.py --config $config --run-name $run --fold $fold --resume
```

状态、verifier、boundary 都有原子 checkpoint。阶段中断后使用相同命令和 `--resume`；
manifest 会拒绝修改后的上游文件、配置或 v2 输入。正式训练不要并行争用同一张 GPU。

## 4. fold 0 模式和消融

先分别运行以下两条主线到 `SELECTED`，不要提前读取 outer 标签：

- `configs/hierarchical_v3.yaml`：`verifier_only`；
- `configs/hierarchical_v3_state_xgb.yaml`：`state_and_verifier`。

每个 selection 自动输出 A2–A7、predicted-IoU 门禁和三随机种子方向证据。A0 使用冻结
XGBoost；A1 使用 v2 已登记的旧 hard-dyadic DTP 证据，不重新解释为 v3 同模型消融。

无 embedding 的 A7c 必须复用同一状态预测缓存，不能重训状态后冒充公平比较：

```powershell
python scripts/prepare_hierarchical_ablation.py `
  --source-config configs/hierarchical_v3.yaml `
  --config configs/hierarchical_v3_no_embedding.yaml `
  --source-run $run `
  --run-name hierarchical_v3_no_embedding_20260930a `
  --fold 0 --fresh
```

若 `state_and_verifier` 胜出，则把 `--source-config` 改为
`configs/hierarchical_v3_state_xgb.yaml`，把 `--config` 改为
`configs/hierarchical_v3_state_xgb_no_embedding.yaml`，并从对应 state-XGBoost run
复用状态缓存。门禁会拒绝 XGBoost mode 不同的 embedding 对比。

之后从 `build_event_candidates.py` 继续执行到 `SELECTED`。模式决策：

```powershell
python scripts/check_hierarchical_gates.py `
  --config configs/hierarchical_v3.yaml `
  --phase fold0-mode `
  --verifier-only-run hierarchical_v3_verifier_only_20260927a `
  --state-xgb-run hierarchical_v3_state_xgb_20260927a `
  --no-embedding-run hierarchical_v3_no_embedding_20260930a `
  --output decisions/fold0_mode.json
```

只有 `state_and_verifier` 的 OOF F1 至少增加 0.010、FP/h 不超过 1.05 倍且异侧召回
下降不超过 0.02 才晋级。embedding 增益不足 0.005 时最终移除 embedding。随后只对
胜出 run 执行一次 fold 0 outer evaluation。

## 5. 开发门禁与冻结压力测试

胜出配置完成 fold 1 后运行：

```powershell
python scripts/check_hierarchical_gates.py `
  --config $config --phase development --run-name $run `
  --output decisions/development.json
```

该命令检查两个 fold 均正向、平均 F1 增益至少 0.015、FP/h、异侧召回、2000 次受试者
配对 bootstrap 和三种子方向。只有 `passed=true` 才能冻结：

```powershell
python scripts/freeze_hierarchical_protocol.py `
  --config $config --run-name $run `
  --development-decision "$env:BME_OUTPUT_ROOT\v3\decisions\development.json"
```

随后按顺序完成 folds 2、3、4。代码会拒绝不同配置、Git commit 或缺少 freeze manifest
的压力折。结束后运行：

```powershell
python scripts/check_hierarchical_gates.py `
  --config $config --phase stress --run-name $run `
  --output decisions/stress.json
```

压力门禁要求至少 2/3 fold F1 不退化、pooled F1 高于 A0、单折最大下降不超过 0.03、
pooled FP/h 不超过 1.05 倍且任一手侧召回下降不超过 0.05。

## 6. 最终训练与导出

五折均完成且人工确认最终规格后：

```powershell
python scripts/train_hierarchical_final.py `
  --config $config --run-name $run --fold all --fresh

python scripts/export_hierarchical_bundle.py `
  --config $config --run-name $run --fold all --fresh
```

最终训练从五个 outer OOF 缓存拟合全数据 verifier、boundary、校准器和阈值，并训练
状态种子 2026/2027/2028 与全数据 XGBoost。三个状态模型只平均 logits，32D embedding
固定来自 seed 2026。输出位于 `outputs/v3/final/<run-name>/`。

导出采用白名单，仅包含模型、归一化、冻结配置和最小推理运行时代码；不含标签、原始数据、
OOF 预测、受试者映射、绝对个人路径或凭据。`model_bundle/SHA256SUMS.json` 和
`model_bundle.zip` 必须在干净环境复核。官方接口公布后只新增 I/O 适配层，不修改模型、
阈值或已冻结证据。
