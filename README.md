# StatsFusion：可穿戴进食事件检测

用于2026生医工竞赛的 **v4.14** 联合检测实现。共享传感器编码器同时学习时间线、候选事件、质量和边界；训练按受试者划分，推理按已观测数据因果提交。

| 身份 | 值 |
| --- | --- |
| 模型 / 协议 | `statsfusion_joint_event_v414` / `statsfusion-joint-event-v9` |
| 修复标识 | `v414-r4` |
| 唯一配置 | [`configs/statsfusion_v414_full_data.yaml`](configs/statsfusion_v414_full_data.yaml) |
| Seed | `2026` |
| 流程 | `audit → dev → final39 → export → replay` |

## 快速开始

Python 3.11或3.12。先安装适合本机的[PyTorch CUDA版本](https://pytorch.org/get-started/locally/)，再安装项目；CPU环境可运行单元测试和合成验收。

```powershell
conda activate bme-model
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

复制`.env.example`为`.env`并设置`BME_DATA_ROOT`与`BME_OUTPUT_ROOT`。训练使用核验通过的`$BME_OUTPUT_ROOT/v2`数据索引与原始session缓存、`$BME_OUTPUT_ROOT/v4` canonical特征。只有原始下载材料时，先在新输出目录生成输入，并沿用原私有subject salt：

```powershell
conda run --no-capture-output -n bme-model python scripts/prepare_statsfusion_inputs.py `
  --config configs/statsfusion_v414_full_data.yaml --stage all --workers 8 --fresh
```

仅生成v2时使用`--stage v2`；已有合格v2时使用`--stage canonical`。恢复准备任务使用`--resume`，禁止覆盖已有共享输入。材料格式、salt、冻结配方及数据身份衔接见[数据准备说明](docs/data_preparation.md)。canonical协议为`statsfusion-r3.2`，与模型版本分开；训练期间输入及冻结划分保持只读，审计失败时停止。

GPU空闲后，在仓库根目录运行：

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v414_full_data.yaml `
  --run-name statsfusion_v414_28_11_then_39_repair_r4_20261014a `
  --evaluation-mode dev28_11 --stage all --fresh
```

中断后使用相同命令，将`--fresh`改为`--resume`。只需开发模型时使用`--stage dev`；开发完成后以`--evaluation-mode final39 --stage all --resume`继续。可用阶段为`audit/dev/final39/export/replay/all`，详细产物和失败处理见[运行说明](docs/training.md)。

## 数据与结果边界

| 集合 | 受试者 | Session | 有效事件 | 自然GYRO全缺失事件 |
| --- | ---: | ---: | ---: | ---: |
| 开发训练 | 28 | 799 | 116 | 6 |
| 开发验证 | 11 | 354 | 45 | 4 |
| 最终重训 | 39 | 1153 | 161 | 10 |

划分冻结于[`splits/v414_dev28_11.json`](splits/v414_dev28_11.json)。开发阶段选择epoch `E`；最终39人模型从头训练至`E+4`，每4轮（4、8、12…）保存权重，并保留`E−4/E−2/E/E+2/E+4`；重复轮次只保存一次，默认校准与导出仍使用`E`。v9保留端点跨session平衡及早停原子终止，完整保留原三个回退候选并附加最多一个学习起点。正式训练每组最多四个版本，负例预评分仍为32个family、每family最多三个、总计96个；模型容量及16038个操作点保持。

最终校准使用已参与39人训练的11名开发受试者，标记为`development_on_final_model`和`independent_calibration=false`。所有产物`promotable=false`；合格最终bundle为`official_test_candidate=true`。开发F1、bootstrap和邻近checkpoint结果均不能作为独立测试证明。Boundary继续学习，部署固定`off`。

当前修复、基线及验收入口见[v414-r4说明](docs/v414_r4_repair.md)，此前修复保留于[v414-r3记录](docs/v414_r3_repair.md)、[v414-r2记录](docs/v414_r2_repair.md)和[v414-r1记录](docs/v414_r1_repair.md)。实施验收不自动启动正式训练；旧协议产物不能resume到v9。

## 仓库结构

```text
configs/                  唯一完整训练配置，无继承链
splits/                   冻结受试者划分
src/bme_eating/
  models/                 共享编码器、定位表示和联合候选头
  training/               自然回放、采样、贡献归一化与恢复
  data/                   session读取、派生特征与一致增强
  workflow.py             开发选模与最终39人工作流
  protocol.py             身份、资格、资源与恢复约束
  streaming.py, bundle.py  因果推理与部署bundle
scripts/                  数据准备、训练、验收、审计和环境入口
tests/                    当前主线及输入语义回归测试
docs/                     架构、运行和验收说明
manifests/                源码来源及本次整理验收
```

候选、评分、缓存、搜索和诊断按功能组织，旧模型、迁移和五折入口已退役。历史实现与旧验收报告可从Git历史查阅，当前入口仅支持v4.14。相邻v4.13项目独立运行。

## 验证与推理

```powershell
python -m pytest -k "not cuda and not gpu"
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
python scripts/validate_statsfusion_joint.py --device cpu
```

GPU验收需要资源空闲：`python scripts/validate_statsfusion_joint.py --device cuda`。真实短训练审计使用`python scripts/audit_statsfusion_v414_training.py --device cuda --long-session --epochs 8`，覆盖5个session与完整任务启用安排，其权重不进入正式run。真实资源测试需显式设置`BME_RESOURCE_TESTS=1`，见[验收说明](docs/validation.md)。CI只执行CPU检查。

部署入口：`bme_eating.bundle.StatsFusionJointFinalDataDetector`。加载导出的bundle后调用`predict_session(raw)`，其中`raw`为`bme_eating.data.stats_fusion_preprocess.RawSessionInput`；接口、因果约束见[架构说明](docs/architecture.md)。

本次整理改变源码身份：正式训练使用全新run；旧源码身份的run不能resume。原始数据、权重、输出、`.env`和subject salt均不提交Git。AI使用及最终提交须遵守赛事规则。
