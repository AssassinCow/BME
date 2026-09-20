# 三折 cross-fit DTP 残差融合运行手册

日期：2026-09-20。正式模型冻结日期：2026-09-26。

## 1. 协议冻结

- 保底模型仍是完整五折 `baseline`；新 fusion 尚无真实五折结果，不能称为优化成功。
- 每个 outer fold 训练三个因果 DTP，分别留出 inner partition 0、1、2。
- 每个 DTP checkpoint 只按 masked window AUPRC 选择，不在训练 epoch 内搜索融合权重。
- checkpoint 选择固定使用可复现的 32,768 行分层验证子集；最佳 checkpoint 确定后再对
  完整 inner-validation 时间线推理一次，写入完整 OOF，最终覆盖不缩减。
- 三份互斥验证预测合成完整 outer-train OOF；三份 outer-test 概率逐点平均。
- 只在完整 cross-fit OOF 上执行一次固定 9 组 `alpha/beta` 搜索。
- 50% 正例采样固定使用 `focal_positive_alpha=0.5`，不再按原始正例率二次加权。
- 内部门禁要求含 DTP 候选优于最佳 beta-only 候选；外层单折结果只作诊断。
- 五个 outer folds 一旦开始就全部完成，最终只由五折比较器决定是否晋级。
- 官方 partial 计分、最终一对一匹配细节和提交接口仍为 `UNKNOWN`。

## 2. 预检

在 RTX 4080 的 `bme-model` 环境、仓库根目录执行：

```powershell
python scripts/check_environment.py
python -m pytest -q
python -m ruff check .
python -m compileall -q src scripts
python scripts/smoke_test_model.py --config configs/dtp_fusion.yaml --batch-size 1
python scripts/validate_data.py --config configs/base.yaml
git status --short
```

正式 clean-room 训练要求最后一条没有输出。冻结 baseline 路线可继续用已有 manifest，但历史
回填只能证明当前文件哈希，不能密码学证明历史训练时的代码状态；报告必须披露这一限制。

旧冻结 baseline 尚无 `run_manifest.json` 时，必须在确认 Git clean 后先执行一次：

```powershell
if ((git status --porcelain).Count -ne 0) {
  throw "Commit and synchronize reviewed changes before manifest backfill"
}
python scripts/backfill_baseline_manifests.py --source-commit 3ca55bb
if ($LASTEXITCODE -ne 0) { throw "baseline manifest backfill failed" }
```

回填只记录现有五折文件和当前数据/折分的哈希，不训练模型、不改预测，也不能证明历史训练代码
确实来自 `3ca55bb`；该提交仅作为团队声明写入 provenance。

## 3. 启动 fold 0

```powershell
$run = "baseline_dtp_fusion_clean_20260920a"
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $run `
  --fold 0 `
  --fresh
if ($LASTEXITCODE -ne 0) { throw "fusion fold 0 did not complete" }
```

`$run` 必须从未使用。 `--fresh` 只允许用于 fold 0，且不能与 `--resume` 同时使用。程序不会
删除或覆盖旧 checkpoint。单个 outer fold 的目录结构为：

```text
fold_0/
  crossfit_0/
    best.pt
    last.pt
    metadata.json
    normalization.json
    best_validation_predictions.parquet
    dtp_test_predictions.parquet
  crossfit_1/
  crossfit_2/
  dtp_oof_predictions.parquet
  dtp_test_predictions.parquet
  validation_predictions.parquet
  test_predictions.parquet
  selected_postprocess.json
  selected_fusion.json
  fusion_trials.csv
  test_events.csv
  test_metrics.json
  test_failure_cases.csv
  run_manifest.json
```

三个 `crossfit_k` 的验证受试者互斥，合并后必须恰好覆盖全部 outer-train 受试者，且不得含
outer-test 受试者。根目录 `dtp_test_predictions.parquet` 是三模型逐点平均，不是单个模型结果。
训练期间的 `best_checkpoint_validation_predictions.parquet` 仅用于 checkpoint 诊断；
`best_validation_predictions.parquet` 始终是在最佳权重上重新生成的完整 partition 预测。

## 4. 固定融合搜索与增量门禁

完整 OOF 只搜索一次：

```text
alpha = 0.0, 0.25, 0.5
beta  = -0.25, 0.0, 0.25
```

`alpha=0,beta=0` 必须逐点、逐事件和逐指标复现原始 baseline。程序从所有 `alpha=0` 行中选择
最佳 beta-only 对照，再从 `alpha>0` 行中选择最佳含 DTP 候选。含 DTP 候选必须同时满足：

- 相对原始 baseline 的 F1 增量至少 `0.015`；
- 相对最佳 beta-only 的 F1 增量至少 `0.005`；
- 相对原始 baseline 的异侧召回增量至少 `0.03`；
- 相对最佳 beta-only 的异侧召回不降低；
- 相对最佳 beta-only 的 FP/h 不增加；
- strict-no-ignore F1 不低于 baseline 和最佳 beta-only。

门禁结果写入 `selected_fusion.json`。失败的 fold 仍生成外层预测并继续完成后续 folds，但最终
`compare_models.py` 会拒绝晋级；不得因失败而看结果后扩网格。

## 5. 中断恢复

`--resume` 指向当前 outer fold 任一已有的 `crossfit_0..2\last.pt`：

```powershell
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $run `
  --fold 0 `
  --resume "$env:BME_OUTPUT_ROOT\v2\experiments\$run\fold_0\crossfit_0\last.pt"
```

该参数恢复整个 outer fold，而不只恢复所指 partition。程序逐一检查三个 partition：完整且签名
一致的自动跳过；存在 `last.pt` 的从下一 epoch 继续；有残留文件但无完整产物或 `last.pt` 的停止
并保留现场。签名包含 outer fold、inner partition、模型/训练/损失配置、baseline 哈希和协议版本。

## 6. 完成其余四折

fold 0 的内部或外部诊断结果不再控制后续折。按顺序完成全部 outer folds：

```powershell
1..4 | ForEach-Object {
  python scripts/train_fusion.py `
    --config configs/dtp_fusion.yaml `
    --run-name $run `
    --fold $_
  if ($LASTEXITCODE -ne 0) { throw "fusion fold $_ did not complete" }
}
```

串行约束只要求前折产物、run name、三折协议和 manifest 身份完整，不检查前折 F1 或召回。

## 7. 唯一最终比较

```powershell
python scripts/compare_models.py `
  --baseline "$env:BME_OUTPUT_ROOT\v2\experiments\baseline" `
  --candidate "$env:BME_OUTPUT_ROOT\v2\experiments\$run" `
  --output "$env:BME_OUTPUT_ROOT\v2\experiments\RUN_PROMOTION.json"
if ($LASTEXITCODE -ne 0) { throw "fusion rejected; keep baseline" }
```

将上面的 `RUN_PROMOTION.json` 替换为 `${run}_promotion.json`。比较器复核五折指标、每折内部
增量门禁、同侧/异侧召回、strict-no-ignore、FP/h、起止 MAE、折间稳定性、受试者级配对
bootstrap、Git/config 身份、输入哈希、三份 checkpoint 和预测哈希。任何一项失败都保留
baseline，不调 alpha/beta、不扩模型、不启用未来上下文。

## 8. 隐私与证据边界

- 原始数据、受试者信息、凭据、NPZ、模型权重和私有失败案例不进入公开 Git 或提交包。
- `selected_fusion.json` 和 `run_manifest.json` 只记录匿名指标、哈希、版本、环境和参数。
- 当前改动只有单元、静态和 smoke 验证，没有正式 fusion 五折实测结果。
- 获得官方测试接口前，不把本地输入输出格式或本地分数称为官方接口或官方分数。
