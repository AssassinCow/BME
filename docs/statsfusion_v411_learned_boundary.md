# v4.11 共享编码器、学习定位与联合判别

本目录是 v4.10 工作树的隔离快照。`manifests/v411_source_snapshot.json` 保存来源 HEAD、未提交差异和逐文件 SHA256；包含原有恢复、clipping 和性能修复。旧 `modeling_joint_v410` 与其运行产物不由本版本修改。

## 注册身份

|字段|值|
|---|---|
|architecture|statsfusion_joint_event_v411|
|protocol|statsfusion-joint-event-v2|
|schema|v411|
|seed|2026|
|config|configs/statsfusion_v411_learned_boundary.yaml|
|run|statsfusion_v411_learned_boundary_20261008a|

仅复用经身份审计的 v2 输入、v4 canonical features 和既有受试者划分。全部模型、候选、CDF、校准与操作点重新生成。每个新 run 保存源码副本、原快照证据、resolved config 和 source/data/runtime 身份。

## 模型与因果语义

共享编码器保留隐藏维度64以及原传感器、质量、统计和缺失分支；3/15/60秒 proposal 路径只聚合已完成块。候选头包含24个全事件ROI token和起止各8个局部token，共40个，五个区域embedding。候选头输出事件logit、质量logit与64维embedding。Boundary输入为local64、candidate64、relative/validity/onset/offset共132维，MLP132→64→2。

occupancy仍负责活动区间。已确认的onset峰阈值0.30、左右各9秒，较早同值峰优先；起点±900秒内保留两个最高峰，间隔至少15秒。每个父proposal最多粗版本加两个学习版本；无峰时才补最多两个兼容版本。几何去重合并source flags。`no_onset`消融保留完整兼容池。

CDF按训练受试者上的父来源与尺度冻结；版本共享父分数。预算为累计有效曝光下允许进入候选头的唯一几何，先分配不同粗父proposal，再补版本。主线没有强制来源配额，仍记录来源与缺额。Boundary重评分的额外调用另计。

接受分数固定为非负Platt校准的事件概率，v411的rank_score等于该概率。已接受的同父版本先比较质量、Boundary置信度、事件概率、稳定ID；跨父版本按概率、质量、ID执行流式NMS。family不强制只能输出一个事件。所有最终输入早于decision；提交后不可撤销；decision与报告终点相差不超过60秒。

这是离散候选坐标和历史detach下的截断联合训练，不能称为完全可微定位。

## 训练与OOF

每chunk256个新步、实际感受野warmup、串行session、累计4个session。session内权重固定。CPU/memmap历史特征detach；当前chunk保持梯度。每chunk最多8组，配额2正、2near-miss、3hard-FP、1background；每组最多3个不同几何。正事件每epoch最多3组，组内版本平均，CPU梯度缓冲在session结束时按实际组数和独立事件数平均。负family重复受控。每个版本、每次refine均重新计算严格IoU>0.25标签。质量排序仅在IoU差至少0.10时参与，权重0.05。

配额由`training.group_quotas`显式注册，v411拒绝旧`sample_composition`及未注册的组数、版本数配置。同一粗父proposal的版本若匹配不同真实正事件，按真实事件拆分监督组，分别计数和归一化。Boundary逐端点判断是否已经观测及实际有效支持是否覆盖；未出现的终点不会阻止合法起点监督。

State/proposal/起止/event/quality/Boundary/smooth/contrastive权重依次0.5/1/0.1/1/0.5/0.1/0.02/0.003。对比学习第6轮启用，历史正对detach，排除ignore、边界±30秒以及同事件其他阶段伪负；无合法正负对返回零。优化器、AMP、最大32轮、clipping与早停门禁保持既有协议。

Boundary从第3轮学习，不要求fit分区达到60事件；严格正候选、端点已出现、支持范围覆盖且有效bin至少3才监督。正式启用独立依赖完整内部OOF：off/start_only/both。先锁粗proposal、预算、NMS；每种修正模式独立自然校准，entropy三值×99接受阈值，始终比较原锁定粗基线。start_only保证终点不变。启用要求至少60独立配对事件、相应端点MAE改善10%、F1下降≤0.005、TP损失≤1%、clipping≤5%。报告分母与精确95%区间；零失败不证明真实风险≤1%。

五折沿用既有subject split。每折3内部排除分区，每个分区35%selector选模，再从头retrain；outer标签不参与选择。局部消融固定fold0/internal0，分别从头训练no_onset、no_boundary、single_candidate，仅作开发诊断。

## 验证、速度与恢复

来源CDF专用路径不调用候选头或Boundary。ROI采用二分时间索引与精确缓存；metadata始终重新进入候选头。验证batch128，在9个decoder与3个预算的入选并集上评分。缓存ignore、IoU矩阵、校准概率和decision分组；最多4进程按注册顺序归并全部8019个粗操作点，各阈值独立执行流式NMS。

训练历史不拼接复制完整137维session；使用分块视图与memmap。sampling/gradients/candidate_history/training_history写入链式SHA256追加日志。恢复按checkpoint记录的字节游标校验并截断未提交尾部。每chunk先fsync缓存、日志，再原子提交RNG、optimizer、累计梯度、正事件缓冲和session游标。worker池在验证前与异常退出时释放。

worker10、prefetch4、reader cache4；resume只允许这三个运行参数变化。模型、损失、数据、源码与协议变化必须新run，无v410式迁移。GPU峰值限制10.5GB，进程树private RAM限制26GB，超限失败。执行日志包含时间、资源、worker、写盘量、吞吐及head调用。

流式入口暂存每次非final调用最后一个anchor的候选决策，收到下一anchor时在增加新曝光前完成它，或在空final调用时与EOF候选一起决策。这样最后一个非空chunk直接final与后续空chunk final共用相同预算和NMS顺序；评分始终受原decision time遮罩。调用方必须发送final以完成最后一组候选提交。已提交事件及其嵌套来源信息保存独立副本。

完整验证2倍加速是待测目标，不是已证实结果。`scripts/benchmark_statsfusion_v411.py`对同一v411 checkpoint和受排除subject运行参考与优化路径，保留完整搜索，核对全部操作点匹配与资格，再保存耗时报告。参数为`--checkpoint --subjects <JSON受试者列表> --output <全新目录>`；使用配置中的device并检查GPU空闲。尚无已训练v411 checkpoint时不能给出真实完整验证加速倍数。

## 门禁与产物

正式门禁保持F1≥0.5475、FP/h≤0.0609237、覆盖≥125/161且各折≥0.70，以及手别、bootstrap、matching和stress要求。优先只读审计已完成v410五折的completion、checkpoint、内部OOF、校准、cohort与曝光；不要求参考先通过其旧晋级门禁。参考不完整时只允许诊断，不能导出。S0显式跳过，comparison_performed=false。

门禁失败写failure_report并保留诊断，隔离任何部署bundle。正式导出必须通过五折锁、专用部署内部OOF与Boundary模式锁，最终从头训练。验收分为实现验证、五折诊断与正式晋级三种不同状态。

## 执行

单 fold0、64 轮上限与已完成 32 轮 selector 的新 run 接续见 [单折接续说明](statsfusion_v411_fold0_continuation.md)。模型、损失、优化器、划分、校准搜索及早停规则保持不变；单折产物不替代完整五折晋级。

在本隔离目录中，等当前v410释放GPU后执行：

```powershell
$config = "configs/statsfusion_v411_learned_boundary.yaml"
$run = "statsfusion_v411_learned_boundary_20261008a"
$reference = "statsfusion_v410_joint_event_20261007a"
conda run -n bme-model python scripts/run_statsfusion_joint.py `
  --config $config --run-name $run --stage all `
  --skip-s0 --reference-run $reference --fresh
```

中断后仅把`--fresh`改为`--resume`。all执行audit→validate→outer→ablation→gates→final→export→replay。若GPU忙，状态落为PAUSED，待资源空闲使用resume。

CPU测试应设置本目录src为PYTHONPATH，避免可编辑安装指向旧项目。隐藏GPU后执行pytest、ruff、compileall及diff检查；GPU bfloat16、真实session资源、真实完整验证速度和正式bundle replay需在资源释放且相应产物存在时执行，不可把synthetic测试等同正式性能。

pytest配置已设置本目录`src`优先。资源释放后可设置`$env:BME_V410_RESOURCE_TESTS="1"`，运行`tests/test_statsfusion_joint_resources.py`；该兼容开关现同时覆盖v410和v411配置。2026-10-08深度审计记录见`docs/statsfusion_v411_deep_audit_20261008.md`。
