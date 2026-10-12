# v4.14 全量数据最终模型

本目录是 v4.13 工作树的独立源码快照，包含当时的 staged、unstaged 和相关 untracked 修复。Git 对象通过 `clone --no-hardlinks` 独立复制；没有对 v4.13 设置文件属性、ACL 或写入其训练输出。来源 HEAD、两类差异、初始逐文件 hash 保存在 `manifests/v414_source_provenance.json` 和相邻 patch 中。

模型数学与 v4.13 一致。新身份为 `v4.14/v414`、`statsfusion_joint_event_v414`、`statsfusion-joint-event-v5`；缓存布局仍为 201 维时间特征和 259 维 ROI。新模型不接受旧 run 身份的 checkpoint、optimizer、候选、CDF 或校准器。评分登记为 `calibrated_event_probability_v414`，提交路径使用事件概率与原有 quality 版本选择。

## 冻结的数据划分

`splits/v414_dev28_11.json` 绑定核验后的原始索引、canonical features、39人身份和完整成员名单。确定性多起点受试者级搜索仅使用 session、有效事件、手关系、真实事件时长和自然缺失统计，没有读取模型成绩。原 fold0 的8人进入训练；两名自然 GYRO 全缺失受试者各进入一组。最终 tie-break 使用排序后的稳定 subject ID。

| 集合 | 受试者 | Session | 有效事件 | 同手/异手 | 自然 GYRO 全缺失事件 |
|---|---:|---:|---:|---:|---:|
| 开发训练 | 28 | 799 | 116 | 52/64 | 6 |
| 开发验证 | 11 | 354 | 45 | 18/27 | 4 |
| 最终训练 | 39 | 1153 | 161 | 70/91 | 10 |

相对约801/352个 session 的目标偏差为−2/+2；事件数和手关系目标恰好匹配，时长分布也纳入目标函数。算法和各项偏差已经写入冻结文件。运行时只核验该文件，禁止重新动态划分。划分文件 SHA256 是配置身份的一部分；更改内容需要新 run。

## 两次独立训练

`all` 按 `audit → dev → final39 → export → replay` 执行。

1. `dev`：28人从头训练，11人选择合格 epoch；最大64轮，epoch8开始每2轮验证。保留 v4.13 的自然回放、困难负例、正组替换、梯度贡献归一化、物理增强、早停与恢复协议。normalization/scaler 只拟合28人的自然数据。保存 `selected.pt`；该文件可能不同于训练结束时的 `latest.pt`。锁定结果写入 `dev_selection.json`。
2. `final39`：39人从头训练，以所选E轮为默认部署权重，再连续训练4轮以保留邻近checkpoint。没有selector，不重新选择epoch，不复用开发模型权重或optimizer。LR保持原64轮时间轴；若E=64，额外4轮使用既有调度的最低LR。完成、finite、源身份与clipping失败仍阻止导出。
   完整epoch额外保留`E−4、E−2、E、E+2、E+4`轮推理checkpoint，路径为`final39/model/nearby/epoch_NNN.pt`。例如E=20时保存16/18/20/22/24轮，训练到24轮；`latest.pt`在E轮后保持不变，`resume.pt`继续保存实际训练进度及optimizer。默认校准和导出仍使用E轮权重，不自动评估或选择邻近模型。manifest分别记录默认部署轮数`fixed_epoch`与实际训练轮数`executed_epochs`。
3. 最终39人模型在原11人上重新推理、拟合非负斜率 Platt，并搜索完整16038个操作点。calibration/scaler 的已见受试者例外只对明确授权的 `development_on_final_model`、恰好11人开发集和恰好39人预处理身份开放；模型推理数据路径仍不读取监督标签，标签仅用于随后开发诊断与校准。

Boundary 继续按照 v4.13 训练，但所有开发搜索与部署强制 `off`，跳过六组修正搜索。45个开发 truth 达不到60个独立配对事件门槛。没有放宽该门槛。

完整搜索、固定 decoder 曲线、正负 logit/ROC-AUC/AP、来源覆盖、端点峰诊断、near-optimal 操作点稳定性与按受试者重采样的锁定操作点区间均保存。bootstrap 不重新搜索，不校正搜索或选模偏差。

## 结果证据边界

28人模型的开发成绩用于 epoch、校准和操作点选择，属于受试者排除的开发证据。39人模型的11人开发标签已经参与训练，结果属于训练内开发校准。`calibration_manifest.json` 与 bundle 明确记录：

- `calibration_kind=development_on_final_model`
- `labels_participated_in_training=true`
- `independent_calibration=false`
- `independent_test_evidence=false`
- `official_test_candidate=true`、`promotable=false`

没有运行正式五折门禁；绝对161事件/125覆盖门槛转换为比例诊断，不据此放行正式晋级。总体、GYRO 可用和 GYRO 全缺失组按完整受试者及其曝光计算 F1/FP/h/覆盖/MAE。手关系与时长报告完整事件的召回和覆盖；其 F1/FP/h 只在 truth 全属于该层的事件 session 中计算，并显式标注范围，因为背景 session 无法按 truth 分配手关系或时长。

`development_comparison.json` 对照28人模型和39人模型在同11人上的结果，不将该比较称为独立测试。官方测试使用冻结配置的因果接口，不读取标签，不重新拟合、搜索或选择 epoch。本实现没有假设新的官方输入文件格式，也没有执行官方测试。

## 运行命令

当前 GPU 任务释放资源后，在新的终端运行：

```powershell
Set-Location -LiteralPath "C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v414"

conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v414_full_data.yaml `
  --run-name statsfusion_v414_28_11_then_39_20261012a `
  --evaluation-mode dev28_11 --stage all --skip-s0 --fresh
```

中断后使用相同命令，将 `--fresh` 改为 `--resume`。仅允许 worker、prefetch、reader/derived cache 等既有运行参数调整。源码、loss、seed、模型、数据或成员名单变化需要新 run。未完成阶段必须先完成，不能跳到另一个阶段恢复。

可以先完整开发训练并保存28人模型：将 `--stage all` 改为 `--stage dev`。之后在同一 run 继续：

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v414_full_data.yaml `
  --run-name statsfusion_v414_28_11_then_39_20261012a `
  --evaluation-mode final39 --stage all --skip-s0 --resume
```

`--evaluation-mode final39` 是执行阶段别名，配置冻结身份仍为整套 `dev28_11` 流程；它执行 `final39 → export → replay`，要求已有合格 `dev_selection.json`，不会重新划分或更改 run 身份。新 run 不能以 final39 跳过 epoch 选择。

GPU 忙时阶段写入 `PAUSED`，等待现有任务结束后 resume。开发训练前自动执行模型 forward/backward、batch1/128、13022秒级长事件和原始流式 replay 预检，预检权重不进入训练。训练和验证继续使用白色单行进度条。GPU/private RAM 上限仍为10.5/26 GiB。

## 产物与使用

run 位于原输出根目录的 `v414/runs/<run-name>`。开发和最终训练分别保存在 `dev/model` 与 `final39/model`；阶段 manifest 以 SHA256 绑定 checkpoint、划分、搜索和评估。完成阶段 resume 也重新核验 hash，不能仅依据状态跳过检查。导出先在 `bundle.partial` 构造并独立加载验收，随后原子提交 `bundle`。

`final39_training_manifest.json` 为五档邻近权重逐一记录路径、epoch、SHA256、完成状态、clipping 与资格；所有五档都进入阶段证据和 bundle 存档。完成训练、resume 和导出时核验完整训练到 E+4 的 history、邻近权重的模型/数据/源码/受试者身份、finite，以及默认 E 权重的一致性。邻近权重可以只具备诊断资格，默认部署仍须通过原有资格检查，并继续使用 E。

前置数据审计、完成阶段校验及运行失败均进入统一状态处理；Ctrl+C/GPU 忙为 `PAUSED`，OOM、文件损坏或协议校验失败为 `FAILED`。已有 run 的前置审计失败也写入失败报告；新 run 尚未建立时直接报告审计错误。报告写入发生次生错误时保留并重新抛出原始异常。

final39 扩展轮次中断时，失败报告的 `final39_training_progress` 分别记录 E 权重是否已完整保存、其资格、`last_recorded_epoch`、目标 E+4 以及扩展训练是否经 manifest 核验。`last_recorded_epoch` 是 history 已记录的轮数，不替代恢复 checkpoint 的提交游标；E 已保存不代表整个 final39 阶段完成。保留 checkpoint，resume 按原恢复游标继续，未完成扩展或后续资格失败时不导出最终 bundle。

bundle 用独立的 `StatsFusionJointFinalDataDetector` 加载；原正式五折 detector 继续拒绝非晋级 bundle。

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(bundle_path) / "src"))
from bme_eating.joint_v414_bundle import StatsFusionJointFinalDataDetector

detector = StatsFusionJointFinalDataDetector(bundle_path, device="cpu")
events = detector.predict_session(raw_session)
```

在新的 Python 进程中执行该示例，避免已经导入的其他版本占用同名模块。loader 也核验实际导入源码与 bundle 的冻结源码一致。`raw_session` 沿用既有 `RawSessionInput` 契约。官方测试只执行锁定方案，不基于其结果二次调参。部署 bundle 同时保存模型、normalization/scaler、source CDF、操作点、阶段身份和源代码。独立 no-XGBoost 进程校验与重复/不同 chunk/native stream replay 由 `replay` 完成。

CPU 检查可独立执行：

```powershell
python -m pytest -k "not cuda and not gpu"
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
git diff --check
python scripts/validate_statsfusion_joint.py --config configs/statsfusion_v414_full_data.yaml --device cpu
```

GPU 检查必须等待现有训练释放资源，然后执行同一个验证脚本并改为 `--device cuda`。实施验收、GPU/真实数据运行验收、开发/最终训练完成、正式晋级分别记录，不能互相替代。

真实生产训练审计使用独立临时权重，可执行 `python scripts/audit_statsfusion_v414_training.py --device cpu --workers 1 --long-session`；GPU 空闲后可改为 `--device cuda`。它只取冻结28人训练池的5个真实 session，从头跑3轮，包含自然GYRO缺失与长session；检查正负贡献、实际 optimizer 更新、head 梯度、96候选上限和资源。报告保存在 `manifests/v414_production_loop_<device>.json`。这个小样本早期审计不选择正式epoch，也不能替代epoch8以后的clipping及校准资格。

本次实现、已完成检查及待验项见[实施验收记录](v414_implementation_acceptance_20261012.md)。CPU真实短审计的三轮clipping fraction为0.50，尚未证明正式checkpoint满足≤0.20；GPU验收等待v4.13训练结束，没有启动正式v4.14训练。
