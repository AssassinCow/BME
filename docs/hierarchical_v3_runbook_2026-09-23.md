# 分层进食检测 v3 运行手册（2026-09-23）

## 1. 范围与硬边界

v3 实现“状态识别 → 候选事件 → 事件验证 → 边界精修”。它只读复用 `outputs/v2`
中的清洗数据、anchors、fold map、基础特征和冻结 XGBoost 预测；所有新产物写入
`outputs/v3`。v3 命令允许在 Git 工作树存在未提交修改时运行，不要求每次修改后先提交。
manifest 会记录当前 commit、dirty 状态和完整工作树指纹 `worktree_sha256`。同一个 run
恢复时必须保持该指纹一致；若继续修改实现，应使用新的 run-name，或使用受控 checkpoint
迁移命令承接已经完成的状态分区。

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

最后一条用于人工记录当前修改，不再要求无输出。GPU smoke 同时检查 bf16
forward/backward、峰值显存不超过
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

状态模型在 epoch 间隔验证时使用 `event_stratified_complete_sessions`：默认目标为
32768 个 anchor，但采样原子是完整 session，不会随机删除 session 内时间点。采样器先
覆盖配置要求的受试者和可评估事件，再补充背景 session；由于完整 session 约束，实际行数
可能超过目标。若当前分区可用事件少于下限，下限按可用事件数折减，并在
`metadata.json` 的 `checkpoint_validation_metadata` 中记录实际覆盖情况。checkpoint
选择器只使用被采样 session 对应的 truth/ignore，避免把未采样事件错误计为 FN。最佳
checkpoint 产生后，训练流程仍会对完整 validation partition 做一次推理，并将完整预测
写入 `best_validation_predictions.parquet`，供后续 OOF 使用。

完整 validation 和 outer-test 推理使用 batch 64；本机 RTX 4080 Laptop 的 bf16 smoke
峰值约 4.92 GB，低于 10.5 GB 门限，重复推理误差为 0。全量时间轴按完整 session 聚合
成目标约 32768 anchor 的分块，每块完成后原子写入隐藏的 `.*_parts/` 目录。若推理被
`Ctrl+C`、关机或其他异常中断，使用同一命令和 `--resume` 时会校验 checkpoint、时间轴、
文件哈希和 AUPRC，然后仅重算未完成或损坏的分块；最终仍合并并严格核验完整时间轴，
不会把分块指标代替完整 OOF 预测。

若旧 run 已完成某个 state partition 的训练，但在全量推理阶段中断，可在修改推理配置
后把该 partition 迁移到新 run，而不重新训练。迁移只允许
`inference_batch_size`、`inference_num_workers`、`inference_resume_chunk_rows` 和
`early_stopping_patience_checks` 改变。patience 变化时，迁移器会按目标 patience 回放验证
历史；只有目标规则停止前已经产生同一个最佳 checkpoint 才允许迁移。模型、损失、采样、
候选、阈值、fold、v2 输入哈希或未完成训练仍会触发拒绝。示例：

```powershell
$sourceRun = "hierarchical_v3_verifier_only_20260927b"
$run = "hierarchical_v3_verifier_only_20260927c"

python scripts/migrate_hierarchical_state_checkpoint.py `
  --config configs/hierarchical_v3.yaml `
  --source-run $sourceRun --run-name $run `
  --fold 0 --partition 0 --fresh

python scripts/train_hierarchical_state.py `
  --config configs/hierarchical_v3.yaml `
  --run-name $run --fold 0 --resume
```

迁移器逐张量计算模型状态 SHA-256，确认迁移前后权重完全一致，只更新目标 selection
signature 和推理运行配置，并写入 `crossfit_0/state/checkpoint_migration.json` 以及目标
`run_manifest.json`。恢复命令会直接进入 partition 0 的分块全量推理；完成后才正常训练
partition 1 和 partition 2。

若一次迁移在创建目标 manifest 后失败，目标仍处于 `CREATED` 且对应 partition 目录为空，
可将上面的 `--fresh` 改成 `--resume` 继续迁移；迁移器仍会重新校验配置、输入哈希、
checkpoint 完整性和权重 SHA-256，不需要删除目标 run。

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
