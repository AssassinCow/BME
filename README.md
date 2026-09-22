# 2026 生医工进食检测建模工程

本工程实现受试者级无泄漏的进食事件检测，包括：

- 15 秒统计/频域特征与 XGBoost 基线；
- 指数分桶历史（对数时间金字塔）XGBoost；
- PPG 质量门控、运动/PPG 晚期融合的 DTP-SQF；
- 统一的事件后处理、删失评估和五折结果记录。

正式预处理、训练和评估只在 RTX 4080 12GB 电脑执行。本仓库不包含原始数据、个人信息、访问凭据、模型权重或训练输出。

## 1. 当前规则边界

- 官方已确认主指标为事件级 F1，只有 `IoU > 0.25` 才算命中；起止时间 MAE 为次指标。
- 官方尚未完整说明一对一匹配、partial coverage、提交字段、时间单位和运行接口。
- 本地默认使用“最大合法匹配数量优先、总 IoU 次优”的一对一匹配，同时输出旧 Hungarian 和 greedy 敏感性结果。它不是官方评分器。
- 惯用手同侧和异侧必须分别报告。
- 疑似误写受试者 ID 默认隔离。得到官方确认后，只能通过 `data.subject_aliases` 显式映射并重建全部 v2 产物。
- `future_context_seconds` 默认是 `0`；官方确认允许整段离线读取前，不把未来上下文模型作为提交主模型。

## 2. 环境与目录

推荐目录：

```text
D:\BME2026\
|-- project\modeling\
|-- BME_Data_2026\
`-- outputs\
```

项目根目录创建不入库的 `.env`：

```dotenv
BME_DATA_ROOT=D:\BME2026\BME_Data_2026
BME_OUTPUT_ROOT=D:\BME2026\outputs
```

所有新版产物写入 `%BME_OUTPUT_ROOT%\v2`。旧输出不会被覆盖，也不得与 v2 混用。受试者 salt 仍保存在 `%BME_OUTPUT_ROOT%\private\subject_salt.hex`；必须随私有实验产物备份，不能提交到 Git。

安装：

```powershell
conda env create -f environment-gpu.yml
conda activate bme-model
python scripts/check_environment.py
```

安装完成后保存环境锁和 GPU 信息：

```powershell
conda env export --from-history > environment-history.lock.yml
pip freeze > requirements.lock.txt
nvidia-smi > nvidia-smi.txt
```

## 3. v2 数据流程

### 3.1 全量安全审计

```powershell
python scripts/audit_data.py --config configs/base.yaml --schema-zips all --maximum-rows 1000000000 --workers 8
```

审计严格检查标准 53 列表头。`--workers` 使用多进程并行扫描 ZIP；主进程仍在每个附件完成后原子更新 checkpoint，因此中断后可续跑。对于带不透明二进制前缀的附件，只允许找到一次完整标准表头，并从该偏移开始按 UTF-8 TSV 严格解析。不会猜测或输出前缀内容。

当前严格审计确认 1112 个附件中有 1029 个 `documented_text`、4 个单表头 `recovered_text_suffix` 和 79 个重复表头附件。重复表头既可能位于二进制前缀之后，也可能出现在一个正常文本段的中途；不得直接跳过表头拼接，先运行专项审计：

```powershell
python scripts/audit_multisection.py --config configs/base.yaml --workers 8
```

专项审计只保存包级指纹、相对持续时间和冲突计数，不保存原始传感器值、文件名、受试者 ID 或绝对时间戳，也不会修改原始数据。当前已复核结果固定为 45 个 `exact_duplicate_overlap`、17 个 `conflicting_overlap` 和 17 个 `nonmonotonic_section`。由于存在不安全附件，命令会以非零状态结束，这是数据门禁生效，不是脚本崩溃；只要已生成完整的 `indices\multisection_audit.json`，即可继续执行下一步预处理。

预处理不会任意选择冲突段：45 个精确重复附件先按原始包时间戳、样本数和包内数值严格去重，再统一展开采样时间；17 个数值冲突附件和 17 个时序回退附件显式隔离。隔离清单 `indices\quarantined_attachments.json` 只保存 ZIP SHA-256、实际分类和状态，不含文件名、路径、受试者或原始值。任何审计哈希、数量或分类变化都会在读取原始附件前停止。官方确认疑似误写 ID 后，应同时修改显式 alias 和期望值。

### 3.2 预处理、session 和 coverage

```powershell
python scripts/preprocess_data.py --config configs/base.yaml --workers 8 --overwrite
```

该步骤会：

- 以 ZIP SHA-256 生成跨电脑稳定的 `segment_id`；
- 对实际发生的降采样先执行分段抗混叠滤波；
- 保留 ACC、GYRO、PPG 的数值和有效性掩码；
- 将同一受试者、无重叠且间隔不超过 3 秒的 segment 连接为虚拟 session；
- 以真实 ACC 掩码计算 coverage、边界可见性、最大缺口和 `evaluable`；
- 生成状态、起止边界、删失掩码、全受试者事件距离和外层五折。

本次 schema 增加了分模态有效率、PPG 槽位、折分指纹和真实历史可用时长。已有 v2 产物不能增量复用，第一次验收必须带 `--overwrite` 全量重建。

任意允许附件仍无法解析、出现 session 时间重叠、数量不符或恢复/隔离计数不符时，流程停止。质量门禁要求 `1112 = 1078 已预处理 + 34 已隔离`，并要求 1029 个 `documented_text`、4 个 `recovered_text_suffix` 和 45 个 `recovered_multisection_deduplicated`。预处理会删除这 34 个隔离哈希对应的旧派生 NPZ，但不会修改或删除原始 ZIP。

主要产物位于 `%BME_OUTPUT_ROOT%\v2`：

```text
indices\records.parquet
indices\segments.parquet
indices\events.parquet
indices\anchors.parquet
indices\subject_folds.json
indices\quality_report.json
indices\quarantined_attachments.json
segments\*.npz
```

### 3.3 人工确认并冻结质量快照

```powershell
python scripts/validate_data.py --config configs/base.yaml
```

确认匿名汇总中的附件数、受试者数、恢复数、coverage、session 数和哈希合理后，显式冻结：

```powershell
python scripts/validate_data.py --config configs/base.yaml --write-expectations
python scripts/validate_data.py --config configs/base.yaml
```

特征和训练命令会重新计算报告。任何指纹变化都会停止，不会自动接受新状态。

## 4. XGBoost 基线

```powershell
python scripts/build_features.py --config configs/baseline.yaml --workers 8
python scripts/train_xgboost.py --config configs/baseline.yaml --fold 0
```

fold 0 通过后运行五折：

```powershell
0..4 | ForEach-Object {
  python scripts/train_xgboost.py --config configs/baseline.yaml --fold $_
}
```

五折保持串行，以免多个训练进程争用同一张 GPU。每折内部的 XGBoost CPU 辅助线程由 `xgboost.n_jobs` 控制，后处理网格搜索由 `postprocess_search.workers` 多进程并行；当前分别为 12 和 16。高阈值网格覆盖到概率域上限 `1.0`，并补充 0.99、0.995 和 0.999 自适应分位点，使常见的验证最优阈值保持在搜索区间内部。后处理 worker 会各自持有一份验证预测，若内存压力明显应先降低 `postprocess_search.workers`。

训练期间或五折完成后，可生成本地结果仪表板：

```powershell
python scripts/summarize_results.py --config configs/baseline.yaml --experiments baseline
```

输出位于 `outputs\v2\reports\baseline\`，包括 HTML 仪表板、总体/逐折/逐受试者/佩戴关系 CSV 和可追溯 JSON。未完成的折会标为临时结果，不会混充完整五折结论。比较基线与 dyadic 时使用 `--experiments baseline baseline_dyadic`。逐受试者文件属于私有实验诊断，不进入公开提交包。

训练规则：

- 外层按受试者固定五折；
- 外层训练集内部按事件数和同侧/异侧建立三折 Group OOF；
- 超参数用三折平均 AUPRC 选择；
- 正类按事件均衡加权，避免超长事件支配训练；
- 后处理只使用 OOF 概率，外层测试折不参与选择；
- 阈值同时搜索低绝对值和验证概率分位数；
- 最优高阈值仍落在搜索边界时停止并要求扩大验证网格。

## 5. 指数分桶历史

```powershell
python scripts/build_features.py --config configs/baseline_dyadic.yaml --workers 8
python scripts/train_xgboost.py --config configs/baseline_dyadic.yaml --fold 0
```

指数桶采用右闭区间 `(start, end]`，相邻桶不共享边界样本。每个桶保留统计量、按原始时间位置计算的趋势、最后有效表示和有效率；PPG 桶另外保留零值段质量特征。特征缓存包含实现版本，修复后运行 `build_features.py` 会自动生成新缓存，不会复用旧桶特征。

运动历史桶为 `1/2/4/8/16/32/64 x 3 秒`，总历史约 381 秒；PPG 为 `1/2/4/8/16 x 15 秒`，总历史约 465 秒。桶互不重叠，靠近当前时刻精细、远处压缩。所有历史通过 session reader 跨附件读取，但不跨 session。

只有 fold 0 相对局部基线有稳定收益时，才运行全部五折。

## 6. DTP-SQF

```powershell
python scripts/smoke_test_model.py --config configs/dtp_sqf.yaml --batch-size 1
python scripts/train_dtp_sqf.py --config configs/dtp_sqf.yaml --fold 0
```

DTP 历史长度、块数、桶时长和桶年龄均由 `motion_block_seconds`、`ppg_block_seconds` 及两组 `bucket_counts` 推导。`state_loss_mask=0` 的锚点不进入 batch 正负样本配额，也不进入窗口 AUPRC；它们仍可保留在完整时间轴预测中用于事件后处理。

DTP 与指数桶共享 session reader。每个 batch 精确满足 `positive_sampling_fraction`，正样本按事件均匀选择，负样本由近事件和远背景组成。验证覆盖完整验证受试者时间轴，并在每个验证 epoch 搜索事件阈值，不再用固定 `0.6/0.3` 选择模型。

中断恢复：

```powershell
python scripts/train_dtp_sqf.py `
    --config configs/dtp_sqf.yaml `
    --fold 0 `
    --resume "$env:BME_OUTPUT_ROOT\v2\experiments\dtp_sqf_causal\fold_0\last.pt"
```

## 7. 评估与晋级

```powershell
python scripts/evaluate_predictions.py `
    --config configs/base.yaml `
    --predictions 'D:\path\test_predictions.parquet' `
    --output 'D:\path\evaluation'
```

v2 预测必须包含：

```text
subject_key,segment_id,session_id,timestamp_ms,
state_probability,start_probability,end_probability
```

不可评估的已知事件作为 ignore interval：先匹配可评估真值，剩余预测与 ignore interval 相交时不计 FP；同时输出不忽略它们的 `strict_no_ignore` 结果。结果还包括总体、同侧/异侧、起止 MAE 和每可观测小时 FP。

每折另外生成 `test_failure_cases.csv`，列出匿名受试者键下的 FN/FP 时间区间，供错误分析使用；该文件属于私有实验输出，不进入公开仓库或最终提交包。

候选模型晋级要求：五折汇总 F1 至少提升 0.02，至少 3/5 折不退化，异侧召回下降不超过 0.02，FP/小时不超过基线 1.2 倍。未满足时保留简单模型。

```powershell
python scripts/compare_models.py `
    --baseline "$env:BME_OUTPUT_ROOT\v2\experiments\baseline" `
    --candidate "$env:BME_OUTPUT_ROOT\v2\experiments\baseline_dyadic" `
    --output "$env:BME_OUTPUT_ROOT\v2\experiments\dyadic_promotion.json"
```

## 8. 4080 验收顺序

```powershell
python scripts/check_environment.py
python -m pytest -q
python scripts/audit_data.py --config configs/base.yaml --schema-zips all --maximum-rows 1000000000 --workers 8
python scripts/audit_multisection.py --config configs/base.yaml --workers 8
python scripts/preprocess_data.py --config configs/base.yaml --workers 8 --overwrite
python scripts/validate_data.py --config configs/base.yaml
# 人工确认后：
python scripts/validate_data.py --config configs/base.yaml --write-expectations
python scripts/build_features.py --config configs/baseline.yaml --workers 8
python scripts/train_xgboost.py --config configs/baseline.yaml --fold 0
```

9 月 26 日后停止高风险模型重构；9 月 27-28 日完成干净环境复现、README 和报告证据链；9 月 29 日只做完整性、隐私、运行和打包审计。

## 9. 隐私与复现

- 不提交原始数据、`outputs`、`.env`、salt、密钥、原始 ID、个人信息或模型权重。
- 每个正式结果保存 Git commit、配置、数据聚合哈希、折分、随机种子、环境锁、GPU 信息、F1、MAE 和局限性。
- 不根据外层测试折手工调阈值或选择模型。
- 官方接口到达后只新增输入/输出适配层，不修改已冻结的折分和证据链。
- 官网提交、提前测评邮件或代表队伍联系组委会，必须由参赛者最终人工确认和执行。

## 10. 9 月 20 日冻结基线与限时核心候选

> 本节原有的 boundary/fastslow/dyadic 命令已停止执行，仅作为历史实验记录保留。
> 当前唯一候选是第 11 节的冻结 baseline + 因果 DTP 受限残差融合。

当前冻结基线提交为 `3ca55bb`，完整五折本地评估为：Micro F1 `0.4701`、
strict-no-ignore macro F1 `0.4350`、同侧/异侧召回 `0.6429/0.1538`、
平均 FP/h `0.023703`、加权起止 MAE `232.72/92.98` 秒。这些都是本地口径，
不得称为官方分数。当前 v2 数据状态是 1112 个附件中 1078 个预处理、45 个精确
多段恢复、34 个隔离，生成 2125 个 segment、1153 个 session 和 267 个事件，
其中 161 个可评估事件。

冻结 `baseline` 不覆盖。限时候选使用独立配置和目录：

```powershell
# 旧 baseline 产物没有 run_manifest 时，先补录当前质量/数据/折分指纹
python scripts/backfill_baseline_manifests.py --source-commit 3ca55bb

# 只复用已有 baseline OOF/test 概率，搜索快慢双 EMA 后处理
0..4 | ForEach-Object {
  python scripts/retune_postprocess.py --config configs/baseline_fastslow.yaml --fold $_
}

# 独立起止边界头 + OOF 异侧权重选择
0..4 | ForEach-Object {
  python scripts/train_xgboost.py --config configs/baseline_boundary.yaml --fold $_
}

# 最后一次精简指数桶实验：只生成 OOF，不查看 fold 0 外层结果
python scripts/build_features.py --config configs/baseline_dyadic_lite.yaml --workers 8
python scripts/train_xgboost.py --config configs/baseline_dyadic_lite.yaml --fold 0 --oof-only
python scripts/screen_dyadic_oof.py --config configs/baseline_dyadic_lite.yaml
```

`screen_dyadic_oof.py` 只读取 fold 0 外层训练受试者的三折 OOF 预测；不会读取 fold 0
test 预测。`baseline_dyadic_lite` 保留局部 PPG 特征，但只增加
`3/6/12/24/48/96` 秒互不重叠的运动历史桶；每个 ACC/GYRO 桶只保留
`mean/std/slope/last/valid_fraction`。旧版 192 秒运动桶、全部 PPG 历史桶和
`rms/maximum` 冗余桶统计不再使用。特征写到独立的
`features\baseline_dyadic_lite.parquet`，不会覆盖旧 dyadic 产物。

只有 OOF 同时达到 F1 `+0.015`、异侧召回 `+0.03`、FP/h 不超过新 baseline
`1.2x`，且所有后处理参数均未触碰搜索边界，才运行外层 fold 0 和后续四折。未通过
即永久停止指数桶路线，不再换桶长或扩大搜索。每折生成 `run_manifest.json`，记录
提交、dirty 状态、配置/数据/模型哈希、环境、GPU、命令和随机种子，但不记录原始
数据、身份、凭据或本机绝对路径。

完整的 4080 命令顺序见
[`docs/baseline_and_dyadic_runbook_2026-09-20.md`](docs/baseline_and_dyadic_runbook_2026-09-20.md)。

最终统一比较：

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

比较器执行计划中的全部硬门禁和 2000 次受试者级配对 bootstrap；数据、折分或质量
指纹变化会直接阻止晋级。官方 partial 计分、一对一匹配细节、提交字段和测试执行接口
仍为 `UNKNOWN`，因此继续同时保留本地 max-cardinality、旧 Hungarian、greedy 和
strict-no-ignore 结果。

## 11. 冻结 baseline + DTP 受限残差融合

当前注册协议是 v4。fold 0 是已观察开发折；参数只能由三块 meta-OOF 产生，不能根据 fold 0
outer 指标人工改参。训练采用 `batch_size=16 × gradient_accumulation=2 = 32`；同时把
`steps_per_epoch` 调为 `1250`，保持每轮样本量和更新次数不变。

```powershell
python scripts/check_environment.py
python -m pytest -q
python -m ruff check src scripts tests
python scripts/smoke_test_model.py --config configs/dtp_fusion.yaml --batch-size 1
python scripts/validate_data.py --config configs/base.yaml
git status --short
```

正式实验要求最后一条无输出。使用两个不同 run name：source run 只保存 DTP cross-fit 预测，
result run 只保存 meta 调参与 outer 评价。原目录只读：

```powershell
$sourceRun = "baseline_dtp_fusion_source_20260922a"
$resultRun = "baseline_dtp_fusion_v4_20260922a"

# 仅在没有可复用冻结 DTP 预测时运行。
python scripts/train_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $sourceRun `
  --fold 0 `
  --fresh

# 只读取冻结 OOF，执行三折 level-2 meta-crossfit 和融合专属双 EMA。
python scripts/tune_fusion.py `
  --config configs/dtp_fusion.yaml `
  --source-run $sourceRun `
  --run-name $resultRun `
  --fold 0 `
  --workers 16 `
  --fresh

# 只有 meta-OOF 硬门禁通过才允许读取 fold 0 outer 标签。
python scripts/evaluate_fusion.py `
  --config configs/dtp_fusion.yaml `
  --run-name $resultRun `
  --fold 0
```

若普通概率门控未通过，先用 `export_dtp_quality.py` 从原 checkpoint 重推理诊断字段，再使用
`configs/dtp_fusion_quality.yaml` 开新 source/result run；该命令不训练模型。fold 1 是首个未观察
确认折，失败时不得继续 folds 2–4。最终只聚合 folds 1–4：

```powershell
python scripts/promote_fusion_v4.py `
  --config configs/dtp_fusion.yaml `
  --candidate-run $resultRun
```

完整分步命令、失败退出码和质量重推理方法见
[`docs/dtp_fusion_v4_runbook_2026-09-22.md`](docs/dtp_fusion_v4_runbook_2026-09-22.md)。
旧的 [`docs/dtp_fusion_runbook_2026-09-20.md`](docs/dtp_fusion_runbook_2026-09-20.md)
只用于解释历史 v3 结果，不再用于新实验。

## 12. 从原始下载数据全量重新运行

第 11 节是复用旧 baseline 的限时路线。需要连审计、预处理、质量快照、特征和 baseline 五折
全部从零重建时，按阶段逐条执行，不再使用单一 PowerShell 总脚本。流程分为：clean Git 与
环境预检、独立输出根目录、全量 schema 审计、重复表头审计、预处理、人工质量冻结、baseline
特征、五折 baseline、候选模型和最终比较。任一步失败都先保留现场并排查，不自动跨过门禁。

新输出目录必须不存在，Git 工作树必须 clean。流程不会读取旧派生数据或训练产物；仅复制私有
salt 以保持匿名键和五折划分可比。DTP/fusion 已采用三折 cross-fit、neutral focal alpha、
beta-only 增量门禁和完整五折协议；正式运行前仍必须提交并同步当前代码，使 Git 工作树 clean。
逐条命令、验收条件和恢复方法见
[`docs/clean_retrain_runbook_2026-09-20.md`](docs/clean_retrain_runbook_2026-09-20.md)。

`scripts/run_clean_retrain.ps1` 仅保留作历史参考，不作为正式实验入口。

## 13. 纯 DTP 留出集诊断（不参与融合门禁）

使用已有的三折 cross-fit DTP 预测，无需重训。仅在外层训练集 OOF 上选择 DTP 自己的双 EMA
后处理，然后对 fold 0 留出集评价一次；原 source/baseline 目录保持只读。

```powershell
python scripts/evaluate_pure_dtp.py `
  --config configs/dtp_fusion.yaml `
  --source-run baseline_dtp_fusion_clean_20260920a `
  --run-name pure_dtp_fold0_20260921a `
  --fold 0 `
  --workers 4
```

新目录的 `diagnostics.json` 包含事件 F1、precision/recall、FP/h、起止 MAE、佩戴手关系、
coverage、strict-no-ignore、受试者分项和窗口 AUPRC/AUROC；`test_events.csv` 与
`test_failure_cases.csv` 供逐事件复核。`selected_postprocess.json`、搜索试验与
`run_manifest.json` 记录选择、参数、源文件及输入哈希。诊断必须使用新的 run name，
不可借 fold 0 留出结果重新选择阈值。fold 0 已在历史融合实验中被观察，故本报告仅作
开发诊断，不替代 v4 的 meta-OOF 门禁，也不能称为新的独立确认折。

## 14. 纯 DTP 后处理 v2（分步开发）

v2 将概率校准、候选事件生成、事件分数过滤、合并及边界修正拆为可否决的消融模块。
`evaluate_pure_dtp.py` 只保留为历史诊断；正式 v2 须先在 clean Git 状态下使用新 run name
运行 OOF 搜索，审阅 `selected_dtp_postprocess.json` 和 `module_ablations.csv`，
且仅在 `meta_gate_passed=true` 时执行外层评价：

```powershell
$sourceRun = "baseline_dtp_fusion_clean_20260920a"
$selectionRun = "dtp_postprocess_v2_20260922a"
python scripts/tune_dtp_postprocess.py `
  --config configs/dtp_postprocess.yaml `
  --source-run $sourceRun --run-name $selectionRun --fold 0 --workers 16
```

`--workers` 最多并行三个 meta 搜索；每个分区有独立签名断点，不能在代码、配置、
预测或标签哈希改变后续跑。搜索对候选的统一部署参数重新做三块 heldout 检查；若没有
单一参数组通过门禁，禁止外层评价。`meta_oof_events.csv`、
`meta_oof_per_subject_metrics.csv`、`meta_oof_hand_relation_metrics.json` 和
`meta_oof_failure_cases.csv` 可用于审阅训练侧表现；通过的边界消融会同步更新事件和门控。
评价使用另一全新目录，且需要该外折对应的冻结 DTP 预测：

```powershell
$fold1Source = "baseline_dtp_fusion_source_fold1_20260922a"
python scripts/evaluate_dtp_postprocess.py `
  --config configs/dtp_postprocess.yaml `
  --source-run $fold1Source `
  --selection-run $selectionRun `
  --run-name dtp_postprocess_eval_20260922a `
  --fold 1
```

上述 `$fold1Source` 仅是待实际训练并核验的示例名称。fold 1 参数由 fold 1 的 outer-train
OOF 重新拟合、fold 1 的留出预测只读取一次。正式 v2 不评价 fold 0；任一 fold 已有 v2
外层评价记录时不能换 run 名再次评价。但 fold 0 的开发 OOF 曾包含 fold 1 受试者，
因此这仍是**受试者重用的跨折诊断**，不是未见受试者的独立确认。不得据此宣称无偏泛化或
独立晋级；要做真正独立确认须有完全不参与 fold 0 调参的新受试者数据。未通过门禁时
保留冻结 XGBoost 作为提交候选。`dtp_event_gate.parquet` 仅在完成上述诊断后，作为
融合 v4 的正残差门控接口进行另一个独立消融；默认 v4 路径不变。
### DTP 互补性与融合消融（协议 2.1，开发阶段）

协议 2.1 恢复旧双 EMA 高召回生成器，按受试者不交叉的三个分区，仅在另两个训练分区拟合
事件分数阈值。它统计与 XGBoost 的共同 TP、两者各自独有 TP、DTP 救回的异侧事件，及
DTP 单独运行的 FP/每个救援。组件筛选**不要求纯 DTP F1 超过 XGBoost**；纯 DTP 的 FP
也不能直接等同融合 FP。两个冻结候选分别侧重救援和较低误报，它们还不是提交方案。

```powershell
$sourceRun = "baseline_dtp_fusion_clean_20260920a"
$selectionRun = "dtp_postprocess_21_20260922a"

python scripts/tune_dtp_postprocess.py `
  --config configs/dtp_postprocess_21.yaml `
  --source-run $sourceRun `
  --run-name $selectionRun `
  --fold 0 `
  --workers 16
```

完成后单独比较冻结 v4、原有乘法事件门控和受限加法救援分支：

```powershell
python scripts/compare_fusion_event_gates.py `
  --config configs/dtp_fusion_event_ablation.yaml `
  --fusion-run baseline_dtp_fusion_v4_20260922a `
  --component-run $selectionRun `
  --run-name fusion_event_ablation_20260922a `
  --fold 0
```

以上命令均要求先将代码提交到 clean Git 状态；新目录不能已存在。消融仅读取 fold 0
outer-train OOF，不读取 outer holdout，报告在 `fusion_event_gate_ablation.json`。受限加法
分支固定原 v4 校准器、权重及后处理，仅在 XGBoost 概率不超过 0.5、DTP 事件门控有效
且校准残差为正时额外补强，救援权重只在另外两个 meta 分区选择。原有正负残差不变。

当前开发探针表明单纯乘法门控不改变原 v4 事件；加法救援的三块训练侧选择可把
F1 `0.5597→0.5811`、异侧召回 `0.4167→0.4861`，但 End MAE `99.65s` 高于
XGBoost `82.06s × 1.1 = 90.27s`，故**原融合晋级门禁仍未通过**。
这不是未见数据的增益证明，不能据此运行 fold 1 或放宽门禁。协议 2.1 的纯 DTP
outer 评价入口已锁定；必须先修复失败项、预先冻结融合策略，再另行确认。搜索、标签、
bootstrap 都是离线操作；部署时只需要因果双 EMA、固定阈值和一次前向推理，不执行搜索。
本地只读计时：`339652` 个冻结 DTP 测试窗口，双 EMA 事件生成、固定阈值过滤和事件门控
映射共约 `0.75s`（不含 XGBoost / DTP 模型前向推理，机器与数据规模变化时需重测）。

### 加法救援的结束边界保留（开发消融）

冻结预测只读、Git 工作树干净后，使用全新 run name 单独运行；该步骤仅使用 fold 0 的
outer-train 三块 meta-OOF，不读取 outer holdout，也不重训 DTP：

```powershell
python scripts/compare_fusion_end_preservation.py `
  --config configs/dtp_fusion_end_preservation.yaml `
  --fusion-run baseline_dtp_fusion_v4_20260922a `
  --component-run dtp_postprocess_21_20260922a `
  --ablation-run fusion_event_ablation_20260922a `
  --run-name fusion_end_preservation_20260922a `
  --fold 0
```

先断言旧报告各块和汇总的 TP、FP、FN、F1、起止 MAE 与 strict F1 完全重放，之后才执行
同受试者、同 session 内 IoU 严格大于 `0.25` 的一对一无标签事件匹配。仅复制匹配事件的
原 v4 结束点，若新边界无效或产生相邻事件重叠则跳过；起点、分数、事件数不变。
`paired_errors.csv`、`paired_summary.csv`、`alignment.csv` 记录手别、逐折配对误差和
对齐/跳过原因；三路事件与 `end_preservation_report.json`、`run_manifest.json` 保留
哈希、参数、门禁、1000 次受试者配对 bootstrap 和固定 `339652` 窗离线后处理计时。

模块门禁需 End MAE 相对未修正救援改善至少 `5%`、各块不恶化、F1 与 strict F1 不降，
再按原 v4 融合门禁相对 XGBoost 判断。均通过时才在完整 outer-train OOF 按既有
训练侧选择规则冻结唯一部署阈值；`apply_frozen_end_preservation` 推理接口只用
冻结数值参数和预测，执行既有 DTP 门控生成、两次固定融合事件解码及一次事件对齐，
不读取标签、不搜索、不 bootstrap，且必须声明推理范围；`outer`
范围会拒绝与拟合受试者重叠的预测。fold 0 已被观察，结果属于
开发假设，不是独立确认。fold 1 必须使用自身无泄漏的冻结 DTP 预测；因其受试者与
fold 0 开发 OOF 重合，不能直接拿 fold 0 拟合的部署参数评估 fold 1；先在 fold 1
自己的 outer-train OOF 重做同一冻结协议，再对其 outer 留出作跨折压力测试。
失败即停止后续折并保留 XGBoost。
目前没有自动 outer 评价入口，不得把本次开发门禁通过当作可提交证明。
