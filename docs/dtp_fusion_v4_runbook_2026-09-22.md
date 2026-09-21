# XGBoost + DTP 稳健融合 v4 分步运行手册

日期：2026-09-22。fold 0 是已观察开发折；fold 1 是首个未观察确认折。本文命令不覆盖旧实验。

## 1. 协议边界

- XGBoost 概率不校准；DTP 在每个 meta-train split 内做正斜率 Platt 校准。
- 三个 `calibration_fold` 轮流作为 held-out meta fold，任何 held-out 受试者不得参与该折参数选择。
- 删除全局 beta；仅在 XGBoost 犹豫区间用持续性门控加入截断后的 DTP 正/负残差。
- `start_probability`、`end_probability` 始终来自 XGBoost。
- 先用 baseline hysteresis 筛选三个非 identity 融合候选，再分别搜索融合专属双 EMA。
- 训练、调参、outer 评价和最终聚合是四个独立命令。
- 训练采用 `16 × 2 = 32`，`steps_per_epoch=1250`；只提高 micro-batch 吞吐，不改变
  有效 batch、每轮样本量或每轮更新次数。

## 2. 正式运行前

```powershell
conda activate bme-model
Set-Location "C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling"

python scripts/check_environment.py
python -m ruff check src scripts tests
python -m pytest -q
python -m compileall -q src scripts
python scripts/smoke_test_model.py --config configs/dtp_fusion.yaml --batch-size 1
python scripts/validate_data.py --config configs/base.yaml
git status --short
```

`git status --short` 必须无输出。正式命令会拒绝 dirty worktree。不要改写历史 run；每次协议或
配置调整都使用全新 run name。

## 3. run name 与四阶段产物

```powershell
$sourceRun = "baseline_dtp_fusion_source_20260922a"
$resultRun = "baseline_dtp_fusion_v4_20260922a"
```

`sourceRun` 只包含三个 DTP checkpoint 的冻结 OOF/test 预测；`resultRun` 包含 meta-OOF 选择、
outer 评价和最终门禁。若已有哈希完整的旧 DTP source，可以直接把 `$sourceRun` 指向它，先不重训。

## 4. fold 0：开发与冻结

没有可复用 source 时才运行训练：

```powershell
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $sourceRun `
  --fold 0 `
  --fresh
if ($LASTEXITCODE -ne 0) { throw "DTP source fold 0 failed" }
```

协议 v4 下该命令在写完 `dtp_oof_predictions.parquet`、`dtp_test_predictions.parquet`、
`dtp_source.json` 和 manifest 后立即结束，不生成 `selected_fusion.json` 或 outer 指标。

从冻结预测做 level-2 meta-crossfit：

```powershell
python scripts/tune_fusion.py `
  --config configs/dtp_fusion.yaml `
  --source-run $sourceRun `
  --run-name $resultRun `
  --fold 0 `
  --workers 16 `
  --fresh
if ($LASTEXITCODE -ne 0) { throw "v4 meta tuning fold 0 failed" }
```

中断后原命令改为 `--resume`；它复用双 EMA checkpoint，不覆盖已经完成的选择。meta 门禁通过后：

```powershell
python scripts/evaluate_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $resultRun `
  --fold 0
```

fold 0 outer 结果只作诊断。不得查看结果后修改同一 `$resultRun` 的参数；任何新假设必须换 run。

## 5. fold 1：首个确认折

先生成该折 source，再调参和评价：

```powershell
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $sourceRun `
  --fold 1
if ($LASTEXITCODE -ne 0) { throw "DTP source fold 1 failed" }

python scripts/tune_fusion.py `
  --config configs/dtp_fusion.yaml `
  --source-run $sourceRun `
  --run-name $resultRun `
  --fold 1 `
  --workers 16 `
  --resume
if ($LASTEXITCODE -ne 0) { throw "v4 meta tuning fold 1 failed" }

python scripts/evaluate_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $resultRun `
  --fold 1
if ($LASTEXITCODE -ne 0) { throw "fold 1 confirmation failed; stop folds 2-4" }
```

fold 1 必须同时通过 meta-OOF 和 outer 均衡门禁。根据 fold 1 改参时，原 result run 作废，下一折
成为新的确认折并使用新 run name。

## 6. folds 2–4：顺序运行

只有 fold 1 确认通过后执行。每一折的 tune 会验证 fold 1 的通过状态、前一折已完成 outer
评价，以及前折 `selected_fusion.json` 的 manifest 哈希。fold 2–4 单折门禁失败会返回退出码 2，
但结果已保存；最终协议允许最多一折小幅下降，因此记录后继续下一折，不在中途改参。

```powershell
2..4 | ForEach-Object {
  $fold = $_
  python scripts/train_fusion.py `
    --config configs/dtp_fusion.yaml `
    --run-name $sourceRun `
    --fold $fold `
    --confirmation-run $resultRun
  if ($LASTEXITCODE -ne 0) { throw "DTP source fold $fold failed" }

  python scripts/tune_fusion.py `
    --config configs/dtp_fusion.yaml `
    --source-run $sourceRun `
    --run-name $resultRun `
    --fold $fold `
    --workers 16 `
    --resume
  if ($LASTEXITCODE -ne 0) { throw "v4 meta tuning fold $fold failed" }

  python scripts/evaluate_fusion.py `
    --config configs/dtp_fusion.yaml `
    --run-name $resultRun `
    --fold $fold
  if ($LASTEXITCODE -notin 0, 2) { throw "v4 outer evaluation fold $fold crashed" }
}
```

## 7. folds 1–4 最终晋级

```powershell
python scripts/promote_fusion_v4.py `
  --config configs/dtp_fusion.yaml `
  --candidate-run $resultRun
```

脚本只读取冻结 folds 1–4，复核 manifest、Git clean 状态、文件哈希、数据/折分指纹、时间线和
受试者覆盖，并执行以下门禁：聚合 F1 至少 `+0.02`；FP/h 不超过 `1.2×`；Start/End MAE
分别不超过 `1.1×`；至少三折 F1 不下降；任一折下降不超过 `0.03`。结果写入
`promotion_v4.json` 和 `promotion_v4.json.sha256`。退出码 2 表示不晋级，应保留 XGBoost baseline。

## 8. 质量门控后备：只重推理

普通门控未通过时，不立即重训。用原 checkpoint 在新目录重推理四个诊断字段：

```powershell
$qualitySource = "baseline_dtp_fusion_quality_source_20260922a"
$qualityResult = "baseline_dtp_fusion_quality_v4_20260922a"

python scripts/export_dtp_quality.py `
  --config configs/dtp_fusion_quality.yaml `
  --source-run $sourceRun `
  --run-name $qualitySource `
  --fold 0 `
  --fresh `
  --workers 8

python scripts/tune_fusion.py `
  --config configs/dtp_fusion_quality.yaml `
  --source-run $qualitySource `
  --run-name $qualityResult `
  --fold 0 `
  --workers 16 `
  --fresh
```

`export_dtp_quality.py` 从 checkpoint 自带的模型和训练配置恢复结构，只做 CUDA 推理；不训练、
不覆盖 source。推理 batch 默认沿用 checkpoint，可用 `--batch-size` 只调整推理吞吐，不改变模型。

## 9. 产物与证据

`selected_fusion.json` 包含三个 meta fold 的训练/验证受试者和时间范围、各折校准器、融合与
后处理参数、逐折指标、meta-OOF 门禁、1000 次受试者配对 bootstrap 区间和最终全 OOF 参数。
manifest 哈希覆盖顶层指标/预测、cross-fit 预测及其 manifest、双 EMA checkpoint/trial 文件。

官方 partial 计分、一对一匹配细节、提交字段和测试执行接口仍为 `UNKNOWN`。本地通过不等于
官方分数；未通过任何门禁时不强推融合，baseline 继续作为提交候选。
