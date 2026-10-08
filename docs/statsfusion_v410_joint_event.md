# v4.10 共享编码器联合事件检测

## 使用范围与隔离

实施目录：`C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v410`。
从 Git HEAD `7e2ce03642bceaeeb115d50a161951636ea5586e` 创建 detached worktree，不修改原 `modeling` 目录或当前训练，不提交 Git commit。

新架构 `statsfusion_joint_event`，协议 `statsfusion-joint-event-v1`，输出独立写入 `BME_OUTPUT_ROOT/v410/runs/<run-name>`。
共享原始数据、canonical 输入和五折 subject split 的只读来源；不加载旧 State、Deep、Boundary、候选或校准产物。

本目录的 `.env` 是原项目运行路径的私有副本，不纳入 Git 或导出 bundle。
数据审计验证 canonical preparation、原始 archive、特征文件和 subject split 的身份。
本地事件表已核对：161 个可评估 truth，140 个 ignore，truth 的 `source_event_key` 唯一。

canonical 身份审计保留原始 SHA-256，不改写已冻结的 preparation/manifest/sidecar。Git `core.autocrlf=true` 创建隔离 checkout 时可能改变 LF/CRLF；只有经逐字节换行归一化核实的等价版本、已注册的历史混合换行哈希以及既有非 canonical 迁移可以复用。实际实现或数据变化仍拒绝通过。
2026-10-07 启动问题已用真实 JointInputs.audit 复核：39 名受试者、五折、161 个 truth、140 个 ignore；所有 canonical 文件哈希前后相同，未创建训练 run。

## 模型与训练

`StatsFusionStateModel.forward_features()` 公开 semantic/local 两种特征；旧 `forward()` 的键及权重格式保留。
共享 ACC/GYRO、方向不变运动、PPG 质量门控及统计分支，hidden=64；3/15/60 秒 proposal 路径只使用完成时间块。
各 proposal 路径采用两层 kernel=3 的深度可分离因果卷积，低分辨率概率保持最近完成块的输出；不会把未完成块当作背景零概率。
24 个候选 ROI bins 聚合 masked mean/max、有效率与位置，接两层轻量时序块、4-head learned-query pooling 和双输出。
事件 BCE 的标签严格为实际候选 IoU>0.25；质量 QFL 的目标为最大实际 IoU，beta=2。
边界改变后重新计算训练标签并重评事件分数；坐标转换停止梯度，不宣称端到端可微定位。
固定最终排序分数为非负斜率 Platt 校准的事件概率乘候选质量。

每个 chunk 为 256 个新时间步，warmup 根据原编码器、跨尺度注意力和新增 proposal 路径的实际感受野推导。
按 session 顺序流式训练，session 内权重固定；四个完成 session 后更新 AdamW。
历史特征 detach 后写入 CPU memmap；当前特征通过 CPU ROI 聚合保留梯度。
历史候选只将固定数量的 ROI tokens 传到 GPU，Boundary 仅传最多 ±900 秒的局部支持，不保留长事件全部原始编码计算图。
训练是截断反向传播；长事件历史编码器不接收完整跨小时反传。

状态及 proposal warmup 为 epoch 1–2；事件、质量和合格边界损失在 3–5 升权；truth-jitter 在 6–8 降至零。
对比学习从 epoch 6 开始，独立 32 维 projection、温度 0.1、最多 128 对。
同事件其他阶段和边界附近不作伪负；已观测背景可作为负对；无有效正负对时记录零损失。
`loss.contrastive: 0` 是关闭对照，必须使用新配置身份和新 run。
保持 session 一致的旋转及模态 dropout，关闭依赖 clip 全局统计的 MixStyle。

独立正事件均衡贡献，负样本按 subject、wear hand、hand relation、IoU、GYRO 缺失、时长与来源分层。
unknown 独立保留，同 family 去重，候选不足时记录缺口。
正事件贡献按独立事件数归一化；负 family 在 chunk 内均衡，再按有效时间贡献归一化，避免长 session 误报数把梯度总量放大。
训练采样权重与自然分布校准权重分别记录。
随机背景候选单独从已观测的有效时间生成，排除 truth/ignore，并保留真实模型分数；高 proposal threshold 不会使背景类永久为空。

## 因果候选与流式接口

来源经验分布在模型训练 subject 上拟合，按来源和尺度冻结，每个 family 只贡献一个代表分数。
在线预算只看累计已观测有效小时：`ceil(hours * budget)`，不读取 session 最终长度。
配额共享预算，重叠类别可同时满足配额，但每个候选只消耗一个预算名额；第一轮优先 family 多样性。
保留 17 个基础/jitter 变体和两个 600 秒扩展来源，transition 关闭。
结束后最多 30 秒提交缓冲，gap 仅搜索 0/15/30 秒；已提交事件不可撤销或重新排序。
每个候选保存 proposal/family/ancestor、64 位 parent SHA-256、父边界、来源、尺度、参数与模型身份。

公共接口仍返回 `bme_eating.types.Event`：subject_key、start_ms、end_ms、score、event_id、hand_relation。
诊断另存 coarse/refined、decision/max-input time、事件概率、质量、rank score 及来源身份。

```python
from bme_eating.joint_pipeline import StatsFusionJointDetector

detector = StatsFusionJointDetector(bundle_path, device="cpu")
events = detector.predict_session(raw_session)
diagnostics = detector.last_event_diagnostics

stream = detector.new_raw_session(subject_key, session_id)
new_events = stream.push(native_raw_chunk, final=False)
last_events = stream.push(last_native_raw_chunk, final=True)
```

增量输入须按时间前进，不允许重叠样本、修改已编码前缀或向已结束 session 追加输入。
也可通过 `new_session(...).push_batch(...)` 输入完成预处理的因果 chunk；此低层接口返回诊断字典。
每次事件决策验证 `max_input_time <= decision_time` 和 `decision_time - end <= 60s`，不附加旧 decoder lag。
session 尾部缺少支持时采用 mask 和 coarse fallback。

## 受试者隔离、校准与晋级

保留现有五折，整个联合模型 seed=2026。每折另外训练三个内部选择模型、三个全内部训练集重训模型，再从头训练 outer 模型。
selector 为内部训练 subject 的 35%，采用已有事件分层算法。
内部预测 subject 和 outer holdout 排除于对应 fit、选模、传感器/统计标准化、来源映射和模型重训。
三内部 epoch 中位数决定 outer 训练时长；内部完整联合模型 OOF 用于校准、操作点和 Boundary 锁定。
部署另建专用三分区内部 OOF，再从头训练全训练集模型；不从 pooled outer 标签选择阈值。

操作点搜索使用批准的 high/budget/NMS/gap/acceptance 网格，校准用自然候选而非训练采样分布。
最早选模 epoch 8，每两 epoch 验证，patience=3，F1 min delta=0.003。
clipping 按完整 epoch 的 optimizer 更新数统计，包含末尾不足四个 session 的更新。epoch>=6 时，只有已裁剪次数除以本轮计划更新数超过 0.50（即后续更新全部不裁剪也不能通过）才提前失败；epoch 完成时该比例等于实际整轮裁剪率。晋级 checkpoint 上限仍为实际整轮裁剪率 0.20，另外校验 finite、完成状态、配置、data/source 与 subject lineage。
每轮保存 clipping_gate、计划/实际更新数和最大裁剪前梯度范数；裁剪失败另存 clipping_failure_report.json，不把未完成 epoch 作为晋级权重。

2026-10-08 修复：旧统计在每轮第一次更新被裁剪时会用 1/1=100% 直接终止。`statsfusion_v410_joint_event_20261007a` 第八轮因此失败；第六、七轮完整裁剪率分别为 1/87 和 5/87，失败前权重与梯度 finite。修复不提高裁剪范数或放宽整轮 50%/晋级 20% 阈值。

用户随后明确授权旧 run 恢复，因此增加一次性 `--resume --migrate-clipping-gate` 迁移。登记文件 `configs/statsfusion_v410_clipping_resume_migration.json` 锁定旧身份、修复源码集合及该断点 SHA-256；仅支持此 run 的第八轮、已提交 29 个 chunk 的断点。其它源码、模型、配置、数据、划分或环境改变仍拒绝恢复。
迁移保留原 `identity.json` 和断点，备份放入 `resume_migrations/clipping_epoch_gate_v1/`，另外生成 `active_identity.json` 与 `resume_migration.json`。模型、optimizer、累计梯度、RNG、session/cache 游标原样加载；未提交的第 30 个 chunk 重新执行。原验证记录备份后重验，五折/最终 manifest 与 bundle 保留迁移证据。迁移准备完成后普通 `--resume` 即可继续，无需重跑前七轮；一般源码修改仍要求新 run。
无可行操作点时保留诊断最优的训练合格 epoch，标记不可晋级，不把诊断选择变成正式资格。

Boundary 须覆盖>=0.70 且至少 60 独立正事件才训练。内部 cross-fit 决定是否启用；启用时使用实际 refined 标签重新校准。
起止 MAE、成对 TP、F1、TP→FP、有效 bins、熵、残差 clipping、冲突和 fallback 分别保存。
成对 MAE 同时锁定 truth 与 proposal 身份；TP→FP 追踪原先命中的同一 proposal。clipping/fallback 统计覆盖完整预算后候选池，包含阈值和 NMS 拒绝的候选。
全部 entropy 与同一个固定 coarse 操作点比较，不根据前一个 entropy 的结果更新基线。修正与已提交邻居产生新冲突时恢复实际未修正边界及其重新校准的分数。
任何声明启用的正式 Boundary 门禁失败均阻止导出。

五折门禁包含 F1>=0.5475、FP/h<=0.0609237、覆盖>=125/161、每折 recall>=0.70、手关系、配对 bootstrap、matching 和 stress 检查。
Bootstrap cohort 采用 truth/prediction/ignore/window 受试者并集，保留纯 FP 和仅有曝光的主体。
fold 4 独立保存误报、操作点和起点 signed error。

历史参考只读审计，不自动补训。其当前 final manifest 声明 `time_constrained_single_holdout_v1`，不能据此推定合格五折配对参考。
完整比较证据需要原始 truth/ignore、曝光时间、预测、匹配、seed 和模型/校准排除身份及哈希；缺失时 `comparison_performed=false`、`promotable=false`。
可审计的补充证据文件为历史实验目录的 `joint_reference_evidence.json`，字段在 `joint_workflow.audit_reference()` 明确列出；不得手填分数或伪造排除证明。
参考审计要求五个 outer model 和五个 calibration lineage 恰好覆盖原 split，保存 artifact 文件及 SHA-256，并证明校准使用完整排除的内部 OOF。
参考 identity 文件及全部证据哈希进入协议锁；final/export 前再次校验，证据变化立即关闭门禁。不可读取的参考仍允许新模型完成诊断。
S0 显式跳过，不宣称优于 sensor-only。

## 恢复和失败语义

状态为 CREATED/RUNNING/PAUSED/FAILED/COMPLETE，outer 另记录逐折状态。单折完成不会把五折 outer 标成完成。
每个完整提交 chunk 保存模型、optimizer、累计梯度、全部 RNG、session/proposal 状态、缓存提交行和恢复游标。
未提交尾部在恢复时截断；Ctrl+C 为 PAUSED，OOM 和异常为 FAILED，仍只能用 resume。
同一 run 使用操作系统文件锁，禁止并发 resume。训练恢复同时校验 epoch 目标、selector/exclusion 身份与游标范围。
内部 OOF 最后原子提交 `internal_manifest.json`，逐折最后提交 `completion.json`；只有报告或 checkpoint 文件存在不代表流程完成。
恢复、gate、final/export 重查完整证据哈希及实际内部/outer checkpoint 的 finite、clipping、训练/排除 subject 和配置/data/source 身份。
运行期间检查 canonical 文件及当前 session 原始 archive 的变更；源码身份检查包含新增、删除文件。
worker、prefetch、reader-cache 可修改后 resume；模型、loss、数据、源码、协议、seed 或其他配置改变需要新 run。
入口检测已有 Python GPU 进程，资源被占用时记录 PAUSED，不启动联合训练。
门禁失败保存独立 failure report 和诊断；final/export 逐项验证锁及 OOF，失败不产生正式 bundle。
正式阶段失败时，已有 bundle 和未完成 staging 原子移入 `failures/nondeployable_*`，保存诊断而不保留部署入口。
部署 checkpoint 只导出必要模型/预处理元数据，以 `weights_only=True` 加载；不导出 optimizer、RNG、恢复游标或任意 pickle 状态。
bundle 检查五折正式门禁结构、内部 OOF 身份、完整文件清单、SHA-256、训练 cohort 和冻结源码。恢复导出需重新验证现有 bundle。

## 运行命令

当前旧训练完成并释放 GPU 后，在隔离目录执行：

```powershell
Set-Location -LiteralPath 'C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v410'
$config = 'configs/statsfusion_v410_joint_event.yaml'
$run = 'statsfusion_v410_joint_event_20261007a'
$reference = 'hierarchical_v4_r32_deep_frontier_20260930a'

conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config $config --run-name $run --stage all `
  --skip-s0 --historical-reference $reference --fresh
```

当前参考若不能补齐合格证据，all 在五折诊断及 gates 后停止，不运行 final/export。

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config $config --run-name $run --stage all `
  --skip-s0 --historical-reference $reference --resume
```

只做身份审计可使用 stage=audit。先 audit 创建 run 后，outer/其他 stage 使用 resume。
如果在 JointInputs.audit 报错且 run 尚未创建，修复后应继续使用 fresh；不要手工修改 preparation_identity、manifest 或 checksum 来跳过检查。
单折执行：`--stage outer --fold 0`；之后 `--stage outer --resume` 自动补其余折。
不要把五折单 seed 理解成只训练五次。

## 验证与证据边界

代码验证不启动五折训练，不宣称模型 F1、起止 MAE 或正式晋级已得到证明。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTORCH_NVML_BASED_CUDA_CHECK = '0'
conda run -n bme-model python -m pytest -q
conda run -n bme-model python -m ruff check src tests scripts --no-cache
conda run -n bme-model python -m compileall -q src scripts
git diff --check
conda run -n bme-model python scripts/validate_statsfusion_joint.py --device cpu
```

CPU 测试覆盖共享事件/质量梯度、完成块、长事件、原始整段/增量回放、前缀、不合格 checkpoint、family、OOF、参考、恢复及操作点。
恢复测试注入 Ctrl+C/OOM 并启用 dropout，比较恢复与连续训练权重，包含部分累计梯度和 session cache。
不同 chunk 的浮点特征允许 1e-5 数值舍入差异，但事件身份必须相同；同 checkpoint 重复推理及事件分数误差上限仍为 1e-6。
CPU/GPU 的训练、推理和 smoke 共用确定性设置，关闭 TF32，并固定 float32 matmul 精度，避免 GPU 不同 chunk 形状引入明显的特征漂移。

GPU 空闲后移除 CPU 测试用的 CUDA mask，再验证完整感受野 smoke：

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
conda run -n bme-model python scripts/validate_statsfusion_joint.py --device cuda
```

训练资源已于本次验收释放，真实 Windows worker=10 的恢复探针及 GPU 完整感受野 forward/backward、长事件和 raw replay 已执行。
GPU 验收分别记录 allocated/reserved 峰值，正式训练按较大值执行 <10.5 GB 门禁；小规模验收不替代真实五折训练的资源记录。
资源释放后也可设置 `$env:BME_V410_RESOURCE_TESTS = '1'`，执行 `python -m pytest -q tests/test_statsfusion_joint_resources.py`。
正式 bundle 仅在门禁通过后产生，然后运行 stage=replay：校验 bundle 哈希、原始会话、长事件及禁止导入 XGBoost 的隔离子进程回放。
导出证据包含 joint_gate_manifest、final_manifest、SHA256SUMS、内部 OOF、逐折与 pooled 指标、搜索记录、来源/边界/资源日志及失败报告。

独立 smoke 的本次实测证据保存为 `docs/statsfusion_v410_cpu_validation.json` 和 `docs/statsfusion_v410_gpu_validation.json`。
无 XGBoost replay 使用测试生成的临时 synthetic bundle；当前没有正式训练 run、五折成绩或通过门禁的部署产物。

## 本次验收记录

- 深入修复后的完整 pytest（启用 GPU 与真实 worker=10 资源探针）：484 passed，1 skipped，101.44 秒。
- 唯一 skip 是既有 canonical 输入缺少 clarified mask-count diagnostics 的旧测试，不属于新联合模型资源测试。
- Ruff、compileall、git diff --check 通过；原 modeling checkout 的 git status 仍为空。
- CPU/GPU 长事件均完成 13023 秒、17/256/整段 chunk 对齐，提交事件身份一致。
- GPU 同输入重复推理误差为 0，native raw 不同 chunk 特征误差 2.3841858e-7、事件分数误差 0。
- FP32 与 BF16 backward 均通过；BF16 事件/质量损失对共享编码器的梯度范数分别为 0.044413、0.010587。
- 完整感受野 GPU smoke 峰值 allocated=1.1961 GiB、reserved=1.3887 GiB、进程 RAM=1.9229 GiB；这些是合成验收工作负载的峰值。
- 新增真实 NPZ→canonical Dataset→训练器的三 epoch 小规模 GPU 测试，验证 BF16、GYRO 部分缺失、正候选贡献、参数更新、安全 checkpoint 加载与完成恢复；输入是测试合成数据。
- 实际 exporter 生成的临时测试 bundle 通过加载、恢复审计、无覆盖和不明文件拒绝测试；恶意 pickle 未执行。
- 当前没有启动正式五折训练，没有证明 F1 提升或满足正式晋级门禁。

具体缺陷、修复与证据边界见 `docs/statsfusion_v410_deep_audit.md`。
