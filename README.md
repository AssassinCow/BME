# 2026 生医工进食检测：StatsFusion v4.14

本仓库的开发主线为 **v4.14**：28名受试者训练、11名开发验证选择epoch，再用全部39名有效受试者从头重训。模型沿用v4.13共享编码器与联合检测训练，增加独立的全量数据工作流、冻结划分、阶段核验和最终开发校准。

| 项目 | 当前主线 |
|---|---|
| 版本 / schema | `v4.14` / `v414` |
| 模型 | `statsfusion_joint_event_v414` |
| 协议 | `statsfusion-joint-event-v5` |
| 配置 | `configs/statsfusion_v414_full_data.yaml` |
| 默认run | `statsfusion_v414_28_11_then_39_20261012a` |
| Seed | `2026` |
| Runner | `scripts/run_statsfusion_joint.py` |
| 最终bundle接口 | `StatsFusionJointFinalDataDetector.predict_session()` |

完整流程、产物及推理示例见[v4.14运行说明](docs/statsfusion_v414_full_data.md)。旧版源码、配置和测试作为历史实现及兼容依赖保留；v4.14运行必须使用自己的配置和全新run。相邻v4.13项目独立运行，不受本仓库提交和推送影响。

## 环境与输入

- Python `3.11`或`3.12`；训练使用CUDA与BF16，CPU可运行相应验收。
- 原始输入根目录由`BME_DATA_ROOT`提供，产物根目录由`BME_OUTPUT_ROOT`提供。
- 只读复用已核验的原始输入、canonical features及subject split；冻结开发划分位于`splits/v414_dev28_11.json`。
- `.env`、凭据、subject salt、原始数据、模型权重和运行输出保存在仓库外或Git忽略目录。

```powershell
conda activate bme-model
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

## 训练与恢复

GPU资源空闲后，在v4.14仓库根目录执行：

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v414_full_data.yaml `
  --run-name statsfusion_v414_28_11_then_39_20261012a `
  --evaluation-mode dev28_11 --stage all --skip-s0 --fresh
```

执行顺序为`audit → dev → final39 → export → replay`。开发训练上限64轮，保留资格检查与早停。最终39人训练采用开发阶段冻结的E轮，不再次选epoch；实际训练到E+4，保存`E−4、E−2、E、E+2、E+4`五档checkpoint，默认校准和导出仍使用E轮。学习率沿用64轮时间轴，额外轮次保持既有最低学习率。

中断后保持相同命令，将`--fresh`改为`--resume`。恢复仅允许worker、prefetch和既有缓存运行参数调整；模型、loss、源码、seed、数据或成员名单变化必须新建run。v4.14拒绝旧版本run及checkpoint身份。未完成阶段按恢复游标继续，损坏或不完整证据不能标记为完成或用于导出。

若先仅执行开发训练，将`--stage all`改为`--stage dev`；完成后在同一run用`--evaluation-mode final39 --stage all --resume`继续最终训练、导出和replay。

## 数据与结论范围

| 集合 | 受试者 | Session | 有效事件 | 自然GYRO全缺失事件 |
|---|---:|---:|---:|---:|
| 开发训练 | 28 | 799 | 116 | 6 |
| 开发验证 | 11 | 354 | 45 | 4 |
| 最终训练 | 39 | 1153 | 161 | 10 |

训练与开发按受试者互斥，两名自然GYRO全缺失受试者各分配一组。开发成绩用于选epoch、校准和完整16038操作点搜索，属于开发证据。最终39人模型在原11人上重新校准；这些标签已参与最终训练，因此该结果明确登记为`development_on_final_model`、`independent_calibration=false`。

主线身份不表示效果已通过正式晋级。最终bundle标记`official_test_candidate=true`、`promotable=false`；Boundary部署固定为`off`。官方测试仅使用锁定配置推理，不能据其标签重新选epoch、拟合校准器或搜索操作点。

## 验收与记录

```powershell
python -m pytest -k "not cuda and not gpu"
python -m ruff check src scripts tests --no-cache
python -m compileall -q src scripts tests
git diff --check
python scripts/validate_statsfusion_joint.py --config configs/statsfusion_v414_full_data.yaml --device cpu
```

GPU验收必须等待已有训练释放资源。实施检查、GPU/真实数据验收、开发与最终训练完成、正式晋级分别记录，不能相互替代。已执行检查与待验项见[实施验收记录](docs/v414_implementation_acceptance_20261012.md)及[完整性修复验收](docs/v414_integrity_repairs_20261012.md)。

每run保存源码、配置/data/runtime身份、冻结划分、训练贡献、候选漏斗、搜索、资源、失败报告和逐文件hash。仓库通过`.gitattributes`固定源码checkout的换行，并保留冻结划分及验收证据的原始字节，以便核验已有hash。历史r3.2运行说明见[历史runbook](docs/hierarchical_v4_statsfusion_runbook.md)，不作为当前v4.14入口。
