# v414-r4完整回退保护与真实验收

基线为`c7d6be3d63b8684b5b8a15014a08e5487a02af52`。版本/schema/架构保持；修复标识`v414-r4`、协议`statsfusion-joint-event-v9`，checkpoint状态结构继续为4。新run为`statsfusion_v414_28_11_then_39_repair_r4_20261014a`，必须fresh，禁止续接旧协议权重、optimizer、梯度银行及候选缓存。

## 修复内容

原r3仅保护一个兼容版本，真实流式合成反例中132–177秒事件的最大IoU由0.306122下降至0.230769。r4完整保留相同输入下的原三版本回退集合，再附加最多一个独有学习起点，总计最多四个；重复几何合并来源。无峰回退沿用原排序，no_onset消融保持原路径。

正式训练的正负组版本上限同步为4，避免下游仍截成3。困难负例预评分与正式监督分开：仍最多32个family、每family预评分3个，总计96个；选择完成后正式loss重新评分最多4个版本。损失权重、版本平均、事件/session贡献归一化、模型容量、增强调度、4-session累计及早停原子终止均保持。

完整回退集合不被学习峰删除是生成阶段的保证；预算仍按唯一几何计数，增加第四版本可能竞争其他名额，quality选择及流式NMS也可能改变结果。实际覆盖收益、背景FP和耗时需分别测量，不能用生成池集合扩大证明F1改善。

## 验收安排

先添加失败回归，复现兼容几何丢失及训练三版本截断；随后检查完整pytest、ruff、compileall、diff及GPU前后向、batch1/128、故障恢复和资源上限。

重新运行5个真实session、8个可丢弃epoch，覆盖正例、背景、自然GYRO缺失和长session，核验实际任务贡献、端点计数、自然回放、增强、对比任务、optimizer和恢复。审计权重不进入正式run。

真实长短session使用当前审计checkpoint重放。真实短session另导出可丢弃审计bundle，在独立CPU进程禁止XGBoost进行加载/回放；它不具有最终39人bundle身份，正式导出契约由fixture测试另行覆盖。完整16038搜索点比较同权重、候选下的参考与缓存执行路径，报告搜索阶段加速及其适用范围。

候选对照读取修复前不可变`source.zip`中的hash锁定候选生成器，使用同一r4权重和自然特征。五个真实session、18个decoder及10/20/30每小时预算分别记录生成覆盖、预算覆盖、固定诊断指标、评分量及耗时；只作为实现及成本诊断，不用于选epoch、调整阈值或证明泛化收益。

## 保持的训练与证据边界

冻结划分仍为799/354个session、116/45个事件，开发28人选择E，最终39人从头训练至E+4，默认部署E。16038操作点、Boundary off、非独立最终开发校准及`promotable=false`保持。

修复前HEAD、源码zip、暂存/未暂存diff及逐文件hash位于仓库外审计目录；旧run、v4.13和历史验收报告保留。实际验收结果登记于`manifests/v414_r4_repair_validation.json`及对应r4报告。本次不自动启动正式训练，也不commit/push。

## 实际验收结果

- 新回归修复前6失败、2通过；修复后完整CPU回归273通过、18跳过，这18项另行使用GPU和资源探针全部通过。Ruff、compileall和diff检查通过。
- CPU/GPU共享事件和质量梯度、BF16反向、历史detach及恢复检查通过；candidate batch1/128最大误差分别为`4.10e-8`和`2.61e-8`。早停终止提交的Ctrl+C/OOM回归仍通过。
- 5个真实session、10个worker、8轮可丢弃生产审计通过，包括自然GYRO缺失和39375秒session。每轮2次optimizer更新，端点实测数量与独立计数表一致；第6–8轮均有820个对比配对，合法任务梯度及参数更新非零。CUDA allocated/reserved峰值为1.446/1.961 GiB，进程树private RAM峰值17.136 GiB。资源结论适用于该审计，不保证所有正式session具有相同峰值。
- 真实29秒事件和13022秒长事件所在session的128/256步回放通过，提交身份一致、事件分数误差为零。长session包含3557275个motion样本和1778641个PPG样本，两条路径分别耗时402.50/225.42秒。真实短session的独立CPU审计bundle回放禁止XGBoost，事件分数与GPU最大误差`3.70e-9`；最终39人bundle契约另由fixture回归验证。这些审计bundle没有最终训练模型身份。
- 参考与缓存路径逐项执行全部16038个搜索点，指标零差异。四个真实短session子集上搜索阶段79.547秒降至3.450秒，约23.06倍；它只证明搜索阶段加速，完整开发验证加速至少2倍尚未测量。长短原始回放使用一致性诊断CDF/校准器，搜索及候选对照使用自然训练父CDF和固定诊断校准器，均不作为效果证明。
- 五个真实session、4个事件、18个decoder和3档预算共54条r3/r4对照中，两者各评分3606个候选、调用候选头108次；生成与预算后覆盖均未增减。该8轮审计权重没有产生额外的第四个独有几何，不能据此测得学习峰增多后的完整训练开销或证明F1改善。合成反例及18个decoder回归已验证生成阶段的回退集合保护。
- 执行前后核验旧run的11988个文件hash和mtime、v4.13的151个源码hash及冻结划分hash均未改变，历史r1/r2/r3验收文件未修改。审计子集身份额外包含5个session清单，其hash与全量输入hash不同；两者已分别与checkpoint、当前输入及回放清单核对一致。

首次真实bundle审计暴露了新增审计脚本按Windows默认GBK读取UTF-8清单的问题。已经明确指定UTF-8，并重新执行8轮生产审计、CPU/GPU检查和真实回放；旧审计报告另存为`before_utf8_audit_fix`记录，不用于当前源码资格。

当前状态为实现验收通过；正式28/11开发训练、39人重训及正式晋级均未执行，`promotable=false`。第四版本仍可能竞争几何预算、改变版本平均和排序配对数量；正式run应继续检查预算后覆盖、FP/h、各任务贡献及耗时。此修复没有将开发校准变成独立泛化证据。

## 正式新run入口

本次不执行下面的命令。使用新run和`--fresh`，不能续接r3及更早协议的checkpoint；同一r4新run中断后才改为`--resume`。

```powershell
Set-Location -LiteralPath "C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v414"
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v414_full_data.yaml `
  --run-name statsfusion_v414_28_11_then_39_repair_r4_20261014a `
  --evaluation-mode dev28_11 --stage all --fresh
```
