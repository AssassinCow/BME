# 2026 生医工进食检测建模工程

本目录实现两套可独立训练、统一评估的方案：

1. **强基线**：15 秒统计/频域特征 + XGBoost + 因果事件后处理。
2. **创新方案 DTP-SQF**：ACC/GYRO 与 PPG 独立编码、指数分桶历史、PPG 信号质量门控、轻量 Transformer、状态与边界多任务输出。

全部代码设计为在队伍自己的 **RTX 4080 12GB** 电脑运行。当前目录不包含原始数据、个人信息、密钥、训练缓存或模型权重。

## 1. 已落实的数据事实

- 下载目录应包含 1150 个有效传感器 ZIP；规范化 ID、排除异常和平台测试记录后，正式监督训练对象为 40 名受试者、1140 个传感器附件。
- 强制排除：`HNU21007`、`HNU21026j`、`HNU21030`。
- 原始文本共有 53 列：三个包时间戳、`PPG1–PPG44`、三轴 ACC、三轴 GYRO。
- 实际样本中 ACC/GYRO 通常每包约 10 行；PPG 通常每包约 15 行，每行前 20 个 PPG 字段有值。
- 本实现根据实际包结构，将 `PPG1–PPG20` 按行优先顺序展开为**单路连续 PPG 波形采样点**，不是 20 个独立 PPG 通道。`PPG21–PPG44` 是否始终为空必须通过全量审计再次确认。
- 包内多行共享同一个包时间戳。本实现使用相邻包时间戳在线性时间轴上展开包内采样点；默认把包时间戳解释为包起点。若官方文档以后确认是包终点，只需修改 `packet_timestamp_anchor` 并重新预处理。
- 内部 100 Hz 运动网格和 50 Hz PPG 网格是建模选择，不是对官方采样率的声明。

## 2. 目录与环境变量

在 4080 电脑上建议放置为：

```text
D:\BME2026\
├── project\modeling\        # 本工程
├── BME_Data_2026\           # 原始下载数据
└── outputs\                 # 审计、缓存、模型和结果
```

项目根目录已经提供 `.env`，默认内容为：

```dotenv
BME_DATA_ROOT=D:\BME2026\BME_Data_2026
BME_OUTPUT_ROOT=D:\BME2026\outputs
```

把工程复制到 4080 电脑后，只需按实际盘符修改一次 `.env`，后续命令会自动读取，不必每次在 PowerShell 设置。标准 `.env` 使用 `名称=值`，不写 `$env:` 前缀。

如果需要临时覆盖，可以在 PowerShell 中设置：

```powershell
$env:BME_DATA_ROOT = 'D:\BME2026\BME_Data_2026'
$env:BME_OUTPUT_ROOT = 'D:\BME2026\outputs'
```

代码只通过这两个变量访问数据和输出，不依赖当前电脑的绝对路径。

优先级为“当前进程已有环境变量 > `.env`”。`.env` 已被 `.gitignore` 排除；可提交的模板是 `.env.example`。

受试者 ID 使用本地盐值做 HMAC 映射。首次运行会在 `$env:BME_OUTPUT_ROOT\private\subject_salt.hex` 生成盐值；它不会进入 Git，但必须随私有实验产物备份，否则重新生成的匿名键无法与旧实验对应。也可以在运行前设置 `BME_SUBJECT_SALT`。

## 3. 在 RTX 4080 上安装

先确认显卡和驱动：

```powershell
nvidia-smi
```

进入本目录后创建环境：

```powershell
conda env create -f environment-gpu.yml
conda activate bme-model
pip install -e .[dev]
python scripts/check_environment.py
```

`environment-gpu.yml` 默认使用 CUDA 12.4。若目标电脑驱动不支持，应先按 PyTorch 官方兼容矩阵调整 `pytorch-cuda`，不要盲目降级驱动。`python scripts/check_environment.py` 必须显示 `cuda_available: true` 和 RTX 4080，再开始数据处理。

安装后固定实际环境：

```powershell
conda env export --from-history > environment-history.lock.yml
pip freeze > requirements.lock.txt
nvidia-smi > nvidia-smi.txt
```

### 3.1 Python 脚本命令与可选参数

每个任务使用独立 Python 脚本，没有总入口或子命令分发。每个脚本均支持 `--help`。

| 脚本 | 必选参数 | 可选参数与默认值 | 用途 |
|---|---|---|---|
| `scripts/check_environment.py` | 无 | 无 | 输出 Python、PyTorch、CUDA、XGBoost、GPU 型号和显存，并在 CUDA 不可用时中止。 |
| `scripts/audit_data.py` | 无 | `--config configs/base.yaml`；`--schema-zips 40`，可设为整数或 `all`；`--maximum-rows 100000` | 建立匿名安全索引，审计 PPG 字段占用和基础数据规模。 |
| `scripts/preprocess_data.py` | 无 | `--config configs/base.yaml`；`--workers <整数>`，默认读取配置；`--overwrite`，默认关闭 | 解析 ZIP、展开包时间戳、切连续段、重采样、生成标签和五折划分。 |
| `scripts/build_features.py` | 无 | `--config configs/baseline.yaml`；`--workers <整数>`，默认读取配置 | 为普通或指数桶 XGBoost 生成特征文件。 |
| `scripts/train_xgboost.py` | `--fold {0,1,2,3,4}` | `--config configs/baseline.yaml` | 训练一个外层折，完成参数搜索、困难负样本、后处理搜索和测试评估。 |
| `scripts/train_dtp_sqf.py` | `--fold {0,1,2,3,4}` | `--config configs/dtp_sqf.yaml`；`--resume <检查点路径>`，默认从头训练 | 训练因果或未来上下文 DTP-SQF，并输出检查点、预测和事件指标。 |
| `scripts/evaluate_predictions.py` | `--predictions <Parquet>`；`--output <目录>` | `--config configs/base.yaml` | 对已有窗口概率重新进行事件后处理和双匹配器评估。 |
| `scripts/smoke_test_model.py` | 无 | `--config configs/dtp_sqf.yaml`；`--batch-size 1` | 在 CUDA 上构造合成输入，检查模型输出形状和峰值显存。 |

常见示例：

```powershell
python scripts/audit_data.py --help
python scripts/train_xgboost.py --help
python scripts/train_dtp_sqf.py --help
```

布尔参数 `--overwrite` 只需写出开关本身，不带 `true/false`。所有独立脚本都会自动定位项目根目录，因此可以从其他目录使用脚本的绝对路径调用。

## 4. 推荐执行顺序

### 4.1 安全索引和格式审计

先检查每名受试者的一个附件：

```powershell
python scripts/audit_data.py --config configs/base.yaml --schema-zips 40
```

确认输出中的 `maximum_observed_nonzero_ppg_slot` 不大于 20，再进行全量审计：

```powershell
python scripts/audit_data.py --config configs/base.yaml --schema-zips all --maximum-rows 1000000000
```

如果发现 `PPG21–PPG44` 中有非零数据，程序会中止。此时先核实它们是额外连续采样点还是其他物理通道，再修改配置；不要直接丢弃。

主要产物：

```text
outputs\indices\records.parquet
outputs\indices\events.parquet
outputs\indices\schema_audit.json
```

索引只保存匿名受试者键，不把食物名称、照片、健康 ID 或用户表字段送入模型。

### 4.2 解析、重采样、标签和五折划分

```powershell
python scripts/preprocess_data.py --config configs/base.yaml --workers 8
```

该命令会：

- 流式读取 ZIP，不生成完整解压副本。
- 展开 ACC、GYRO 和 PPG 包内时间轴。
- 按时间戳逆序和大缺口切连续段。
- 重采样并生成有效性掩码。
- 重新计算事件的完整、部分和无覆盖状态。
- 每 3 秒生成状态、开始和结束标签。
- 创建固定的受试者级 5 折划分。

主要产物：

```text
outputs\segments\*.npz
outputs\indices\segments.parquet
outputs\indices\anchors.parquet
outputs\indices\subject_folds.json
```

若修改 PPG 展开规则、时间戳锚点、采样网格或异常名单，必须使用 `--overwrite` 重新生成，旧缓存不得混用。

## 5. 方案一：XGBoost 强基线

### 5.1 局部 15 秒特征

```powershell
python scripts/build_features.py --config configs/baseline.yaml --workers 8
```

特征包括：

- 六轴稳健统计量、有效率、加速度和角速度幅值。
- jerk、轴间相关性、主频、频谱熵和分频带能量。
- 单路 PPG 统计、脉动频段集中度、自相关、平线、削顶、缺口和 SQI。

训练一个外层折：

```powershell
python scripts/train_xgboost.py --config configs/baseline.yaml --fold 0
```

确认折 0 正常后运行全部五折：

```powershell
0..4 | ForEach-Object {
    python scripts/train_xgboost.py --config configs/baseline.yaml --fold $_
}
```

训练包含 30 组内层参数搜索、餐前后近端负样本、最多 5:1 的远端背景负样本和一次困难负样本挖掘。随后仅使用内层验证预测搜索 729 组有界事件后处理参数，冻结后才应用到外层测试折。默认使用 GPU XGBoost；若目标环境的 XGBoost GPU 构建不可用，可把配置中的 `device` 改为 `cpu`，但仍在 4080 电脑上运行并记录环境变化。

### 5.2 指数桶特征基线

```powershell
python scripts/build_features.py --config configs/baseline_dyadic.yaml --workers 8
0..4 | ForEach-Object {
    python scripts/train_xgboost.py --config configs/baseline_dyadic.yaml --fold $_
}
```

它在同一个 XGBoost 框架中增加互不重叠的指数历史桶，用来低成本检验“近处精细、远处压缩”的历史表示是否独立带来收益。

## 6. 方案二：DTP-SQF

### 6.1 模型结构

- 运动基础块：3 秒，ACC/GYRO 六路数值 + 六路有效性掩码。
- PPG 基础块：15 秒，单路 PPG 数值 + 有效性掩码。
- 运动历史桶：1、2、4、8、16、32、64 个基础块，总计约 381 秒。
- PPG 历史桶：1、2、4、8、16 个基础块，总计约 465 秒。
- 每桶保留嵌入均值、标准差、最大值、末端表示、趋势和有效率。
- SQI 使用有效率、最大缺口、削顶、差分异常、脉动频段集中度、自相关峰、平线比例和稳健信噪指标。
- PPG 门控公式：`g·PPG_token + (1-g)·missing_token`。
- 13 个历史/查询 token 加两个预留未来 token，进入两层、四头、192 维 Transformer。
- 同时预测进食状态、开始边界和结束边界。

### 6.2 GPU 冒烟测试

```powershell
python scripts/smoke_test_model.py --config configs/dtp_sqf.yaml --batch-size 1
```

先确认输出形状正确且显存可用，再训练折 0：

```powershell
python scripts/train_dtp_sqf.py --config configs/dtp_sqf.yaml --fold 0
```

折 0 可以完整训练、保存检查点并输出测试预测后，再运行全部五折：

```powershell
0..4 | ForEach-Object {
    python scripts/train_dtp_sqf.py --config configs/dtp_sqf.yaml --fold $_
}
```

中断恢复：

```powershell
python scripts/train_dtp_sqf.py `
    --config configs/dtp_sqf.yaml `
    --fold 0 `
    --resume "$env:BME_OUTPUT_ROOT\experiments\dtp_sqf_causal\fold_0\best.pt"
```

默认参数为 BF16、批量 4、梯度累积 8、每轮 5000 个同连续段平衡批次。训练集按连续段组织批次并在每个 worker 内缓存最近段，避免反复读取压缩文件。早停代理集包含全部事件邻近连续段及每名验证受试者的固定背景段，每两轮评估一次；最佳模型最后仍对完整外层测试折推理。若显存超过约 10.5GB，先把训练批量改为 2、累积改为 16，并相应降低 `inference_batch_size`；仍然不足时再冻结局部编码器或降低 DataLoader 工作线程，不要缩短历史桶作为第一反应。

### 6.3 有限未来上下文消融

因果主线不读取未来信号。离线消融分别增加 30 秒和 60 秒未来摘要 token：

```powershell
python scripts/smoke_test_model.py --config configs/dtp_sqf_future30.yaml --batch-size 1
python scripts/train_dtp_sqf.py --config configs/dtp_sqf_future30.yaml --fold 0

python scripts/smoke_test_model.py --config configs/dtp_sqf_future60.yaml --batch-size 1
python scripts/train_dtp_sqf.py --config configs/dtp_sqf_future60.yaml --fold 0
```

只有折 0 显示稳定收益后才运行未来版本的全部五折。未来版本是否可作为提交模型，必须等官方确认推理接口允许读取整段文件。

## 7. 评估与结果位置

每个训练命令都会输出：

```text
outputs\experiments\<方案>\fold_<折号>\
├── metadata.json
├── selected_postprocess.json
├── postprocess_trials.csv
├── test_predictions.parquet
├── test_events.csv
├── test_metrics.json
└── 模型文件或检查点
```

也可以对任意内部预测文件重新评估：

```powershell
python scripts/evaluate_predictions.py `
    --config configs/base.yaml `
    --predictions 'D:\path\test_predictions.parquet' `
    --output 'D:\path\evaluation'
```

评估器严格使用 `IoU > 0.25`，同时输出：

- 最大权重一对一匹配。
- IoU 降序贪心匹配。
- 事件 F1、灵敏度、阳性预测率。
- 正确命中事件的开始和结束 MAE。
- 同侧、异侧和未知佩戴关系的灵敏度及边界误差。

这两个匹配器都是本地复核实现。官方尚未公布完整匹配细节，因此不能把其中任何一个称为官方最终评分器。

## 8. 测试

所有测试在 4080 电脑运行：

```powershell
pytest
```

测试覆盖：

- 包时间戳展开和 PPG 行内采样点展开。
- `IoU=0.25` 不命中以及重复预测的一对一匹配。
- 迟滞阈值事件生成。
- 受试者只属于一个外层折。
- DTP-SQF 输出形状和因果模型的确定性。

全数据训练前至少依次通过：

```powershell
python scripts/check_environment.py
pytest
python scripts/audit_data.py --config configs/base.yaml --schema-zips 40
python scripts/smoke_test_model.py --config configs/dtp_sqf.yaml --batch-size 1
```

## 9. 隐私与复现要求

- `src/bme_eating/data/` 是可复现所需的数据读取、预处理、标签和划分源码，必须随工程同步；它不是原始数据目录。
- 不把 `BME_Data_2026`、`outputs`、盐值、密钥、日志中的原始 ID 或检查点提交到公开 Git。
- 原始数据目录 `BME_Data_2026` 只在 4080 电脑本地按 `.env` 配置，`outputs/`、缓存和模型权重继续保持忽略，不进入代码同步包。
- 用户信息表中的年龄、性别、身高、体重和设备标识不作为首版模型输入。
- 每个正式实验保留代码版本、配置、数据清单哈希、划分、随机种子、环境锁和 GPU 信息。
- 不得根据外层测试折结果手工调阈值；阈值和后处理只能由内层验证确定。
- 总体结果必须同时报告同侧与异侧佩戴场景，不能只用一个总分掩盖场景失效。

## 10. 当前明确未实现的内容

- 官方测试集适配器和最终提交列名：官方接口仍未知。
- 官方可执行文件封装：必须等测试接口、时间单位和运行环境确认。
- 全天原始波形 Transformer、大规模自监督、MaskCAE 和基础模型迁移：不进入首轮双轨 MVP。
- 自动发送提前测评邮件或官网提交：必须由参赛者最终人工确认和执行。

获得官方测试接口后，应只新增输入/输出适配层，保持内部事件格式：

```text
subject_key,start_ms,end_ms,score
```

不要为了适配官方文件而修改已冻结的数据划分、模型逻辑或本地评估证据链。
