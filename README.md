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
python scripts/audit_data.py --config configs/base.yaml --schema-zips all --maximum-rows 1000000000
```

审计严格检查标准 53 列表头。对于带不透明二进制前缀的附件，只允许找到一次完整标准表头，并从该偏移开始按 UTF-8 TSV 严格解析。不会猜测或输出前缀内容。

当前门禁为 39 名可确认受试者、1112 个附件、16 个 `recovered_text_suffix`。官方确认疑似误写 ID 后，应同时修改显式 alias 和期望值。

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

任意附件仍无法解析、出现 session 时间重叠、数量不符或恢复附件数不符时，流程停止。

主要产物位于 `%BME_OUTPUT_ROOT%\v2`：

```text
indices\records.parquet
indices\segments.parquet
indices\events.parquet
indices\anchors.parquet
indices\subject_folds.json
indices\quality_report.json
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

运动历史桶为 `1/2/4/8/16/32/64 x 3 秒`，总历史约 381 秒；PPG 为 `1/2/4/8/16 x 15 秒`，总历史约 465 秒。桶互不重叠，靠近当前时刻精细、远处压缩。所有历史通过 session reader 跨附件读取，但不跨 session。

只有 fold 0 相对局部基线有稳定收益时，才运行全部五折。

## 6. DTP-SQF

```powershell
python scripts/smoke_test_model.py --config configs/dtp_sqf.yaml --batch-size 1
python scripts/train_dtp_sqf.py --config configs/dtp_sqf.yaml --fold 0
```

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
python scripts/audit_data.py --config configs/base.yaml --schema-zips all --maximum-rows 1000000000
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
