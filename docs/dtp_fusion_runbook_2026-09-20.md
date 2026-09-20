# 冻结 baseline + DTP 受限残差融合运行手册

日期：2026-09-20。正式模型冻结日期：2026-09-26。

## 1. 路线冻结

- 保底模型固定为提交 `3ca55bb` 的五折 `baseline`，不重训、不重建特征。
- `baseline_boundary` 状态为 `rejected_active_search_boundaries`。
- `baseline_fastslow`、所有 XGBoost dyadic、TCN、future-context DTP 和原 DTP 全量后处理搜索均停止。
- 唯一候选为因果 DTP 对冻结 baseline 状态概率作受限残差融合。
- 官方 partial 计分、最终一对一匹配细节和提交接口仍为 `UNKNOWN`；本流程只产生本地评估。

## 2. 不会执行的步骤

本流程不调用以下脚本：

```text
audit_data.py
audit_multisection.py
preprocess_data.py
build_features.py
train_xgboost.py
retune_postprocess.py
```

它只读取已冻结的 v2 indices、预处理 NPZ、baseline OOF/test 预测和 baseline 后处理参数。

## 3. 预检与冻结哈希

在 4080 的 `bme-model` 环境、仓库根目录执行：

```powershell
python scripts/check_environment.py
python -m pytest -q
python -m ruff check .
python scripts/smoke_test_model.py --config configs/dtp_fusion.yaml --batch-size 1
python scripts/validate_data.py --config configs/base.yaml
python scripts/backfill_baseline_manifests.py --source-commit 3ca55bb
```

最后一条命令只为既有五折 baseline 写入数据、折分、模型、OOF/test 预测、后处理和指标哈希。
它不训练模型，也不改预测文件。`3ca55bb` 会作为团队声明的原始训练提交单独记录；回填本身
不能用密码学方法证明历史训练时的代码状态。fusion 启动时会逐文件复核产物哈希和这项声明；
不一致即停止。

## 4. fold 0 门禁

```powershell
$run = "baseline_dtp_fusion_clean_20260920a"
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $run `
  --fold 0 `
  --fresh
if ($LASTEXITCODE -ne 0) { throw "fusion fold 0 failed; keep frozen baseline" }
```

`$run` 必须是从未使用过的名字。`--fresh` 只允许用于 fold 0 且不能与 `--resume` 同时使用；
它会原子创建 `%BME_OUTPUT_ROOT%\v2\experiments\$run`，若目录已经存在则立即停止，绝不删除、
覆盖或读取其中的旧 DTP checkpoint、预测和 fusion trial。这里的“从零”仅指 DTP/fusion；
冻结 `3ca55bb` baseline 的 OOF/test 概率与后处理参数仍是当前融合定义的一部分，并会只读复用。

每个验证 epoch 只运行 9 个固定组合：

```text
alpha = 0.0, 0.25, 0.5
beta  = -0.25, 0.0, 0.25
```

`alpha=0,beta=0` 必须逐点和逐事件精确复现 baseline。选择过程只读 baseline OOF 中当前
DTP 内层验证受试者的预测；外层 test 预测在内部门禁通过前不会用于推理或评估。退出码 2 表示
预注册门禁失败，应立即停止并保留冻结 baseline，不是程序崩溃。

主要产物：

```text
%BME_OUTPUT_ROOT%\v2\experiments\<run-name>\fold_0\
  best.pt
  last.pt
  best_validation_predictions.parquet
  validation_predictions.parquet
  dtp_test_predictions.parquet
  test_predictions.parquet
  selected_postprocess.json
  selected_fusion.json
  fusion_trials.csv
  test_events.csv
  test_metrics.json
  test_failure_cases.csv
  run_manifest.json
```

`dtp_test_predictions.parquet` 是私有诊断产物；公共候选预测为 `test_predictions.parquet`。

## 5. 中断恢复

恢复路径必须属于当前 fold，且配置、固定 9 组合和 baseline 哈希必须保持一致：

```powershell
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $run `
  --fold 0 `
  --resume "$env:BME_OUTPUT_ROOT\v2\experiments\$run\fold_0\last.pt"
```

发现既有 `last.pt` 而未显式传 `--resume` 时脚本会停止，避免覆盖可恢复训练。训练 sampler 会把
epoch 写入每个数据索引，PPG 增强与模型随机过程均由 fold/epoch 确定；从完整 epoch checkpoint
恢复时不会额外插入验证轮次，因此恢复路径与不中断运行保持同一随机和验证日程。

## 6. 后续四折与最终比较

只有 fold 0 内部和外部门禁均通过，脚本才允许 fold 1 启动。后续折仍记录相同的内部诊断，
但不重复用 fold 0 的增益阈值阻断；任一外层折零召回时阻止下一折：

```powershell
1..4 | ForEach-Object {
  python scripts/train_fusion.py `
    --config configs/dtp_fusion.yaml `
    --run-name $run `
    --fold $_
  if ($LASTEXITCODE -ne 0) { throw "fusion fold $_ failed; keep frozen baseline" }
}
```

五折完成后运行唯一最终比较：

```powershell
python scripts/compare_models.py `
  --baseline "$env:BME_OUTPUT_ROOT\v2\experiments\baseline" `
  --candidate "$env:BME_OUTPUT_ROOT\v2\experiments\$run" `
  --output "$env:BME_OUTPUT_ROOT\v2\experiments\${run}_promotion.json"
if ($LASTEXITCODE -ne 0) { throw "fusion rejected; keep frozen baseline" }
```

比较器执行预注册的五折 F1、同侧/异侧召回、strict-no-ignore、FP/h、起止 MAE、折间稳定性、
数据/折分指纹和 2000 次受试者级配对 bootstrap 门禁。失败后不调 alpha/beta、不扩模型、
不启用未来上下文，也不重新搜索后处理。

fold 0 外层门禁决定是否继续 fold 1-4，所以最终五折结果是经过 fold 0 条件筛选后的本地比较，
不是完全无偏的独立测试估计。报告中必须披露该限制；官方计分接口到达前不得称为官方分数。

## 7. 隐私与交付边界

- 原始数据、受试者信息、凭据、NPZ、模型权重和私有失败案例不进入公开 Git 或提交包。
- `selected_fusion.json` 和 `run_manifest.json` 只记录匿名指标、哈希、版本、环境和参数。
- 获得官方测试接口前，不把当前输入输出格式或本地分数称为官方接口或官方分数。
