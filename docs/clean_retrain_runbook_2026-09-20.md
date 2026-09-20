# 全量 clean-room 分步骤重训手册

日期：2026-09-20。该手册是正式执行入口。`scripts/run_clean_retrain.ps1` 仅保留作历史
参考，不再推荐用于正式实验；不要用一个总脚本跨过人工质量检查、折级验收或模型路线决策。

本流程从原始下载数据重新生成审计、预处理、质量快照、baseline 特征与五折结果。原始数据
只读，唯一允许从旧输出复制的文件是私有 `subject_salt.hex`。所有分数均为本地验证结果，
不是官方测试分数。

## 0. 当前执行门禁

开始前必须满足：

- 已在 RTX 4080 电脑同步到准备训练的提交；
- `git status --short` 没有输出；
- 当前 Conda 环境是 `bme-model`；
- 原始数据路径由 `.env` 中的 `BME_DATA_ROOT` 提供；
- 旧 salt 文件存在且不进入 Git；
- 新输出根目录此前不存在。

当前实现已固定为三折 cross-fit DTP ensemble、50% 正例采样配 `focal_positive_alpha=0.5`、
最佳 beta-only 增量门禁和无外层早停的完整五折协议。正式训练前仍必须提交、同步并确认工作树
clean；这次代码修正本身没有产生新的 fusion 实测结果。

## 1. 激活环境并冻结代码身份

在仓库根目录逐条执行：

```powershell
conda activate bme-model
Set-Location "C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling"

if ($env:CONDA_DEFAULT_ENV -ne "bme-model") {
  throw "Activate bme-model first"
}

python -c "import sys; print(sys.executable); print(sys.version)"
git status --short
if ((git status --porcelain).Count -ne 0) {
  throw "Commit and sync all reviewed changes before formal training"
}

$codeCommit = (git rev-parse HEAD).Trim()
if ($codeCommit -notmatch '^[0-9a-fA-F]{40}$') {
  throw "Cannot resolve the training commit"
}
$codeCommit
```

验收：Python 路径位于 `anaconda3\envs\bme-model`，Git 状态为空，并保存本次 40 位
`$codeCommit`。不要在训练中途切换提交。

## 2. 创建隔离输出根目录

以下路径按当前电脑给出；若训练机路径不同，只修改 `$cleanRoot` 和 `$saltSource`：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$cleanRoot = "C:\Users\LZX\Desktop\University\Competition\生医工\outputs_clean_$stamp"
$saltSource = "C:\Users\LZX\Desktop\University\Competition\生医工\outputs\private\subject_salt.hex"

$cleanRoot = [System.IO.Path]::GetFullPath($cleanRoot)
$saltSource = [System.IO.Path]::GetFullPath($saltSource)
if ($cleanRoot -eq [System.IO.Path]::GetPathRoot($cleanRoot)) {
  throw "Output root cannot be a drive root"
}
if (Test-Path -LiteralPath $cleanRoot) {
  throw "Choose a new output root; existing outputs must not be reused"
}
if (-not (Test-Path -LiteralPath $saltSource -PathType Leaf)) {
  throw "Private subject salt is missing"
}
$saltFormat = (Get-Content -LiteralPath $saltSource -Raw).Trim()
if ($saltFormat -notmatch '^[0-9a-fA-F]{64}$') {
  throw "Subject salt must be 32 bytes encoded as 64 hexadecimal characters"
}
$saltFormat = $null

New-Item -ItemType Directory -Path $cleanRoot | Out-Null
New-Item -ItemType Directory -Path (Join-Path $cleanRoot "private") | Out-Null
Copy-Item -LiteralPath $saltSource `
  -Destination (Join-Path $cleanRoot "private\subject_salt.hex")
$env:BME_OUTPUT_ROOT = $cleanRoot
$env:BME_OUTPUT_ROOT
```

验收：新目录只含 `private\subject_salt.hex`。同一 PowerShell 会话中设置的
`BME_OUTPUT_ROOT` 优先于 `.env` 默认值；后续命令都在此会话执行。

## 3. 环境、测试和 CUDA 预检

```powershell
python scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment check failed" }

python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

python -m ruff check .
if ($LASTEXITCODE -ne 0) { throw "Ruff failed" }

python -m compileall -q src scripts
if ($LASTEXITCODE -ne 0) { throw "Compilation check failed" }

python scripts/smoke_test_model.py --config configs/dtp_fusion.yaml --batch-size 1
if ($LASTEXITCODE -ne 0) { throw "CUDA smoke test failed" }
```

验收：完整测试、Ruff、编译和 CUDA smoke 全部返回 0。正式 clean-room 运行不跳过 Ruff；
如果某项失败，应先修代码并重新提交，此输出目录不再作为正式结果使用。

## 4. 全量 schema 审计

```powershell
python scripts/audit_data.py `
  --config configs/base.yaml `
  --schema-zips all `
  --maximum-rows 1000000000 `
  --workers 12 `
  --no-resume
if ($LASTEXITCODE -ne 0) { throw "Full schema audit failed" }
```

验收：生成 `$cleanRoot\v2\indices\schema_audit.json`，并完整检查 1112 个当前纳入索引的
附件。正式从零运行使用 `--no-resume`；只有本步骤意外中断且代码、数据、参数均未改变时，
才去掉该参数继续检查点。

## 5. 重复表头审计

```powershell
python scripts/audit_multisection.py --config configs/base.yaml --workers 12
$multisectionExit = $LASTEXITCODE
$multisectionReport = Join-Path $cleanRoot "v2\indices\multisection_audit.json"
if (-not (Test-Path -LiteralPath $multisectionReport -PathType Leaf)) {
  throw "Multisection audit did not write its complete JSON report"
}
if ($multisectionExit -notin @(0, 1)) {
  throw "Unexpected multisection exit code: $multisectionExit"
}
Get-Content -LiteralPath $multisectionReport
```

退出码 1 表示发现不能自动精确去重的附件，不等于程序崩溃。当前冻结数据预期为 79 个重复
表头附件，其中 45 个精确重复恢复、17 个冲突重叠和 17 个非单调段，后两类共 34 个隔离。
数量不同必须停下排查，不能直接预处理。

## 6. 全量预处理与人工质量冻结

```powershell
python scripts/preprocess_data.py `
  --config configs/base.yaml `
  --workers 24 `
  --overwrite
if ($LASTEXITCODE -ne 0) { throw "Preprocessing failed" }

python scripts/validate_data.py --config configs/base.yaml
if ($LASTEXITCODE -ne 0) { throw "Unfrozen quality validation failed" }
```

人工核对匿名报告。当前数据快照应为：1112 records、39 subjects、1078 个预处理附件、
34 个隔离附件、2125 segments、1153 sessions、267 events、161 evaluable events；coverage
为 full 161、partial 77、none 23、invalid_duration 6。还要检查 77 个零有效 GYRO segment、
各模态有效率、PPG 槽位、数据/折分 digest 和 `preprocess_issues=0`。

只有确认这些数量及原因均可接受后，才逐条执行：

```powershell
python scripts/validate_data.py --config configs/base.yaml --write-expectations
if ($LASTEXITCODE -ne 0) { throw "Could not freeze quality expectations" }

python scripts/validate_data.py --config configs/base.yaml
if ($LASTEXITCODE -ne 0) { throw "Frozen quality gate failed" }
```

不要为消除门禁错误而盲目重写 expectations。39/40 人差异、34 个隔离附件和官方 partial
计分方式仍需在报告中披露。

## 7. 构建 baseline 特征

```powershell
python scripts/build_features.py --config configs/baseline.yaml --workers 16
if ($LASTEXITCODE -ne 0) { throw "Baseline feature build failed" }

python scripts/validate_data.py --config configs/base.yaml
if ($LASTEXITCODE -ne 0) { throw "Quality gate changed after feature build" }
```

验收：生成 `$cleanRoot\v2\features\baseline.parquet`，没有 `feature_issues.json` 中的失败项，
质量报告和折分 digest 未变化。

## 8. 五折 baseline：逐折运行、逐折验收

五折串行，全部使用 `--no-resume`。每折结束后检查该目录中的 `test_metrics.json`、
`selected_postprocess.json`、`test_predictions.parquet`、`model.json` 和 `run_manifest.json`；
manifest 必须记录 `$codeCommit`、`dirty=false`、正确 fold 和同一 resolved config hash。

```powershell
python scripts/train_xgboost.py --config configs/baseline.yaml --fold 0 --no-resume
if ($LASTEXITCODE -ne 0) { throw "Baseline fold 0 failed" }

python scripts/train_xgboost.py --config configs/baseline.yaml --fold 1 --no-resume
if ($LASTEXITCODE -ne 0) { throw "Baseline fold 1 failed" }

python scripts/train_xgboost.py --config configs/baseline.yaml --fold 2 --no-resume
if ($LASTEXITCODE -ne 0) { throw "Baseline fold 2 failed" }

python scripts/train_xgboost.py --config configs/baseline.yaml --fold 3 --no-resume
if ($LASTEXITCODE -ne 0) { throw "Baseline fold 3 failed" }

python scripts/train_xgboost.py --config configs/baseline.yaml --fold 4 --no-resume
if ($LASTEXITCODE -ne 0) { throw "Baseline fold 4 failed" }

python scripts/summarize_results.py `
  --config configs/baseline.yaml `
  --experiments baseline
if ($LASTEXITCODE -ne 0) { throw "Baseline summary failed" }
```

验收时把新五折与历史冻结 baseline 对照，但不要求数值完全相同；代码、数据、折分和配置均
可能已变化。重点检查五折均非零召回、阈值未触碰搜索边界、异侧召回、FP/h、起止 MAE、
逐受试者失败案例和折间波动。

## 9. 三折 cross-fit DTP 残差融合：逐折完整运行

每个 outer fold 内训练三个 DTP 模型，分别留出 inner partition 0、1、2。checkpoint 只按
DTP window AUPRC 选择；三份验证预测合成完整 outer-train OOF，三份 outer-test 概率平均。
融合权重只在完整 OOF 上选择一次。

```powershell
$run = "baseline_dtp_fusion_clean_$stamp"

python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $run `
  --fold 0 `
  --fresh `
  --baseline-source-commit $codeCommit `
  --require-clean-baseline
if ($LASTEXITCODE -ne 0) { throw "Fusion fold 0 failed; retain clean baseline" }
```

fold 0 完成后无论外层诊断或内部增量门禁是否通过，都继续完成 fold 1-4。门禁失败表示候选
最终不能晋级，不表示程序失败，也不授权看结果后改网格：

```powershell
python scripts/train_fusion.py --config configs/dtp_fusion.yaml --run-name $run --fold 1 `
  --baseline-source-commit $codeCommit --require-clean-baseline
if ($LASTEXITCODE -ne 0) { throw "Fusion fold 1 failed" }

python scripts/train_fusion.py --config configs/dtp_fusion.yaml --run-name $run --fold 2 `
  --baseline-source-commit $codeCommit --require-clean-baseline
if ($LASTEXITCODE -ne 0) { throw "Fusion fold 2 failed" }

python scripts/train_fusion.py --config configs/dtp_fusion.yaml --run-name $run --fold 3 `
  --baseline-source-commit $codeCommit --require-clean-baseline
if ($LASTEXITCODE -ne 0) { throw "Fusion fold 3 failed" }

python scripts/train_fusion.py --config configs/dtp_fusion.yaml --run-name $run --fold 4 `
  --baseline-source-commit $codeCommit --require-clean-baseline
if ($LASTEXITCODE -ne 0) { throw "Fusion fold 4 failed" }
```

中断只恢复当前折。路径可指向该折任一 `crossfit_0..2\last.pt`；命令会核对三个 partition
各自的 outer fold、inner partition、配置、baseline 哈希和协议签名，跳过已完整 partition：

```powershell
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $run `
  --fold 0 `
  --resume "$cleanRoot\v2\experiments\$run\fold_0\crossfit_0\last.pt" `
  --baseline-source-commit $codeCommit `
  --require-clean-baseline
```

## 10. 最终比较与结果固化

五折 fusion 全部完成后逐条执行：

```powershell
python scripts/summarize_results.py `
  --config configs/dtp_fusion.yaml `
  --experiments baseline $run
if ($LASTEXITCODE -ne 0) { throw "Fusion summary failed" }

python scripts/compare_models.py `
  --baseline "$cleanRoot\v2\experiments\baseline" `
  --candidate "$cleanRoot\v2\experiments\$run" `
  --output "$cleanRoot\v2\experiments\${run}_promotion.json"
if ($LASTEXITCODE -ne 0) { throw "Fusion rejected; retain clean baseline" }
```

比较器会重新校验五折 manifest、clean Git、单一 commit、单一 resolved config、三折 cross-fit
checkpoint 与预测哈希、实验名、fold 身份以及每名受试者恰好出现在一个测试折。五个内部
beta-only 增量门禁也必须全部通过；任何证据链不完整都不会晋级。

最后记录 `$cleanRoot`、`$codeCommit`、环境输出、质量 JSON、五折指标、promotion JSON、
随机种子和已知局限。fold 0 外层结果不再决定是否运行后续折；五折全部完成后才作唯一晋级
判断。官方 partial 计分、一对一匹配和最终接口仍为 `UNKNOWN`。
