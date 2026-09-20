# 新 baseline 与精简 dyadic 运行手册

日期：2026-09-20。以下命令在 RTX 4080 的 `bme-model` Conda 环境中运行。
所有分数均为训练数据内部的本地评估，不是官方测试分数。

## 实验定义

- 新 baseline：`baseline_boundary`。使用 15 秒局部 ACC/GYRO/PPG 特征、快慢双
  EMA 后处理、独立 start/end 边界头和 OOF 选择的异侧事件权重。
- 新 baseline_dyadic：`baseline_dyadic_lite`。在新 baseline 上仅增加
  `3/6/12/24/48/96` 秒运动历史桶；保留局部 PPG，但不使用 PPG 历史桶。
- 旧 `baseline_dyadic` 和 `baseline_dyadic_v2` 不再继续运行，也不覆盖其产物。

## 1. 同步、环境和代码验收

```powershell
git pull
conda activate bme-model
python scripts/check_environment.py
python -m pytest -q
ruff check src scripts tests
python -m compileall -q src scripts
```

确认 `.env` 中已有 `BME_DATA_ROOT` 和 `BME_OUTPUT_ROOT`。不要把 `.env`、原始数据、
身份信息或凭据加入 Git。

## 2. 数据门禁

已有完整 v2 预处理产物时只运行：

```powershell
python scripts/validate_data.py --config configs/base.yaml
```

必须确认仍为 1112 个附件、1078 个预处理、34 个隔离、2125 个 segment、1153 个
session、267 个事件，并且质量与折分指纹未变化。门禁失败时停止训练，不要自动写入
新的 expectations。

仅当原始数据、预处理代码或质量快照发生了预期变更时，才按 README 第 3 节重新执行
全量审计、预处理和人工冻结；正常代码同步不需要重复约一小时的全量 schema 审计。

## 3. 构建共享局部特征

```powershell
python scripts/build_features.py --config configs/baseline.yaml --workers 8
```

产物为 `%BME_OUTPUT_ROOT%\v2\features\baseline.parquet`。如果特征缓存签名和数据未变，
脚本会复用缓存；质量门禁仍会先执行。

## 4. 运行新 baseline

先跑 fold 0，确认无触边、无零召回且指标文件完整：

```powershell
python scripts/train_xgboost.py --config configs/baseline_boundary.yaml --fold 0
```

fold 0 通过后串行完成其余四折：

```powershell
1..4 | ForEach-Object {
  python scripts/train_xgboost.py --config configs/baseline_boundary.yaml --fold $_
  if ($LASTEXITCODE -ne 0) { throw "baseline_boundary fold $_ failed" }
}

python scripts/summarize_results.py `
  --config configs/baseline_boundary.yaml `
  --experiments baseline_boundary
```

不要并行跑五折。每折必须存在 `test_metrics.json`、`selected_postprocess.json`、
`run_manifest.json` 和预测文件。

旧冻结 baseline 若尚无 manifest，补录一次，然后比较新旧 baseline：

```powershell
if (!(Test-Path "$env:BME_OUTPUT_ROOT\v2\experiments\baseline\fold_0\run_manifest.json")) {
  python scripts/backfill_baseline_manifests.py --source-commit 3ca55bb
}

python scripts/compare_models.py `
  --baseline "$env:BME_OUTPUT_ROOT\v2\experiments\baseline" `
  --candidate "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_boundary" `
  --output "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_boundary_promotion.json"
```

比较器退出码 2 表示新 baseline 未满足完整晋级条件，不是程序崩溃。保留 JSON 结果；
由于 lite dyadic 是预先限定的最后一次低成本实验，可以继续 OOF 门禁，但最终候选仍须
再次与旧冻结 baseline 比较。

## 5. 新 baseline_dyadic 的 OOF 门禁

构建独立 lite 特征：

```powershell
python scripts/build_features.py --config configs/baseline_dyadic_lite.yaml --workers 8
```

它写入 `%BME_OUTPUT_ROOT%\v2\features\baseline_dyadic_lite.parquet`，不会覆盖 baseline
或旧 dyadic。随后只训练 fold 0 外层训练受试者并生成三折 OOF 校准结果：

```powershell
python scripts/train_xgboost.py `
  --config configs/baseline_dyadic_lite.yaml `
  --fold 0 `
  --oof-only

python scripts/screen_dyadic_oof.py --config configs/baseline_dyadic_lite.yaml
```

筛选脚本退出码为 0 才表示全部通过：OOF F1 相对 `baseline_boundary` 至少 `+0.015`、
异侧召回至少 `+0.03`、FP/h 不超过 `1.2x`，且后处理参数均未触边。退出码 2 或训练
因触边退出时，停止 dyadic，不运行任何外层测试或其他折。

## 6. 仅在 OOF 通过后运行 dyadic 五折

`--oof-only` 没有生成 fold 0 外层评估，所以先补跑 fold 0，再跑 1-4：

```powershell
python scripts/train_xgboost.py --config configs/baseline_dyadic_lite.yaml --fold 0
if ($LASTEXITCODE -ne 0) { throw "baseline_dyadic_lite fold 0 failed" }

1..4 | ForEach-Object {
  python scripts/train_xgboost.py --config configs/baseline_dyadic_lite.yaml --fold $_
  if ($LASTEXITCODE -ne 0) { throw "baseline_dyadic_lite fold $_ failed" }
}

python scripts/summarize_results.py `
  --config configs/baseline_dyadic_lite.yaml `
  --experiments baseline_boundary baseline_dyadic_lite
```

## 7. 五折最终晋级

```powershell
python scripts/compare_models.py `
  --baseline "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_boundary" `
  --candidate "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_dyadic_lite" `
  --output "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_dyadic_lite_promotion.json"

python scripts/compare_models.py `
  --baseline "$env:BME_OUTPUT_ROOT\v2\experiments\baseline" `
  --candidate "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_dyadic_lite" `
  --output "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_dyadic_lite_vs_frozen.json"
```

只有 dyadic 相对新 baseline 和旧冻结 baseline 的两个比较器均返回 0，才把 dyadic
作为最终候选。否则在通过门禁的 `baseline_boundary` 与旧 `baseline` 中选择；若新
baseline 也未通过，就保留旧冻结 baseline。不要因为某个外层 fold 表现好而回头修改
桶长、阈值或统计量。官方 partial 计分、最终一对一匹配、提交字段和测试执行接口仍
未知，当前结果不得称为官方分数。
