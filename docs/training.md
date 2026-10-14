# 训练与恢复

## 运行依赖

`.env`或环境变量提供输入及输出根目录。`JointInputs.audit()`核验v2索引、session缓存、v4 canonical特征及其来源；冻结28/11划分与当前数据hash必须一致。`scripts/generate_v414_split.py`仅打印确定性生成结果供核对，运行时不会重新划分。

从原始下载材料生成输入使用`scripts/prepare_statsfusion_inputs.py`，完整命令、私有salt要求及恢复方式见[数据准备说明](data_preparation.md)。该入口使用冻结v2配方，不改变训练配置或模型。

v4.14的训练、源码快照、缓存及诊断写入`$BME_OUTPUT_ROOT/v414/runs/<run>`。旧版输入只读复用；不写相邻v4.13源码或产物。GPU忙时返回PAUSED，保留状态并等待资源释放后resume。

## 两次训练

1. `dev`：28人从头训练，11人开发验证选择E。normalization仅在28人自然数据上拟合。最大64轮，epoch8起每2轮验证，保留操作点资格、clipping与两类早停规则。
2. `final39`：39人从头训练，采用相同64轮LR时间轴，不重新选epoch。运行至E+4并保存五档邻近checkpoint，默认最终模型为E。
3. 最终模型在11名开发受试者上重新拟合Platt校准器和搜索操作点，明确记录这些标签已参与最终训练。16038个操作点全部保留；部署Boundary固定off。
4. `export/replay`：资格及证据hash通过后原子导出bundle，验证锁定因果推理与无XGBoost重放。

AdamW、BF16、256个新步及实际感受野warmup、每session权重固定、累计4个session、梯度裁剪5.0保持。无资格checkpoint不能进入最终重训或导出。邻近模型保存用于审计和后续明确授权的实验，默认流程不会选择邻近模型或用官方测试标签调参。

## 阶段命令

```powershell
python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v414_full_data.yaml `
  --run-name statsfusion_v414_28_11_then_39_repair_r3_20261014a `
  --evaluation-mode dev28_11 --stage all --fresh
```

`--stage dev`只执行开发选模；完成后使用同一run及`--evaluation-mode final39 --stage all --resume`执行最终训练、导出与重放。`--skip-s0`与`--reference-run`保留作命令兼容与身份记录；本流程不进行正式参考晋级比较。

中断恢复必须完成当前阶段，禁止绕过失败阶段导出。resume只允许worker、prefetch、reader cache、derived cache等已登记运行参数调整；模型、loss、源码、数据、seed和划分变化均要求新run。提交相同源码的Git commit不改变源码身份。

## 恢复提交

每chunk先持久化缓存和校验追加日志，再原子提交checkpoint。v8恢复状态版本为4，保存RNG、模型/optimizer、LR位置、累计梯度、端点计数表及hash、逐版本和逐排序对梯度、几何所有权、纯负贡献、自然回放hash、候选及重复诊断释放游标、增强描述及早停计数。停止原因、完成epoch和终止标记与下一轮游标在同一个checkpoint提交；恢复终止状态只返回选模结果或重建失败报告，不进入训练循环。恢复截断未提交日志尾部、核验缓存，只接受完整chunk。旧协议checkpoint、梯度缓冲、候选缓存与校准器不得加载或resume。

自然回放按subject/session/start/end合并几何，首次可用decision和phase的稳定代表提供完整metadata。同decision/phase合并来源审计；晚到重复按自身时间释放诊断，不改写早期输入。预算几何去重不依赖CDF fallback，即使首次预算不足，后续重复也不能重试。

每epoch使用独立计数视图计算训练session的合法峰值和背景资格，不消耗训练RNG。令W为所有session权重和，W_peak/W_bg为具有对应监督的session权重和；端点贡献为`w_s * ((W/W_peak)*P_s + 20*(W/W_bg)*B_s)/21`，onset/offset平均后乘0.1。分母为零时对应项关闭。28人和39人各自统计，计数只读取本阶段训练标签。

合法训练父候选为空时记录专用无资格原因，CDF、校准和操作点不可用；该验证计入已有无资格规则。数据损坏、身份错误、非finite或OOM继续失败。验证结果与早停计数原子提交，中断恢复不会重训完成epoch或重复计数。

Ctrl+C为PAUSED；OOM及其他异常为FAILED。未完成epoch不标记完成；失败报告和诊断权重保留，资格失败不生成部署bundle。资源上限为GPU峰值10.5 GiB、进程树private RAM 26 GiB，超限失败而不静默减小训练协议。

## 主要产物

| 产物 | 用途 |
| --- | --- |
| `identity.json`, `resolved_config.json`, `source_snapshot/` | 数据、源码、运行与配置身份 |
| `split_manifest.json`, `dev_selection.json` | 冻结成员与开发选模证据 |
| `dev/model/`, `final39/model/` | 模型、resume状态和训练日志 |
| `final39/model/nearby/` | E−4/E−2/E/E+2/E+4权重 |
| `final39_training_manifest.json` | 固定epoch、从头训练及资格 |
| `calibration_manifest.json`, `operating_point.json` | 最终开发校准与锁定操作点 |
| `final_manifest.json`, `SHA256SUMS.json` | 导出资格与证据完整性 |
| `resource_log.json`, `failure_report.json` | 资源诊断与失败原因 |
| `bundle/`, `bundle_replay.json` | 推理包与重放结果 |

训练、自然回放、CDF、验证和select均显示单行进度。训练分别显示chunk计算目标及最近提交update在实际保留集合上的归一化目标；两者含义不同。日志保存峰值/背景有效数量、分项梯度、实际任务贡献和VJP耗时，零梯度夹角记录为不可定义。验证输出F1、TP/FP/FN、FP/h、候选覆盖、边界误差、操作点、clipping、最佳epoch和早停计数。
