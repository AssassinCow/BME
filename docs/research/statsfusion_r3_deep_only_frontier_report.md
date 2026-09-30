# StatsFusion v4 Deep-only 与前沿优化复核报告

日期：2026-09-30  
代码代际：v4.7.1 / `statsfusion-r3.2`
证据等级：当前结果属于 development/stress evidence，不等同于隐藏测试或新增受试者验证。

## 1. 结论

当前最合理的部署链路是：

```text
ACC/GYRO/PPG 原始输入
  -> StatsFusion state model
  -> causal decoder / transition candidates
  -> proposal features
  -> mask-aware Deep verifier
  -> acceptance + NMS
  -> optional boundary refinement
```

本次将部署选择收敛为 Deep-only：

- `state` 仍保留，因为它是候选生成器，删除它会失去候选召回能力。
- `state-only` 仅保留为内部诊断基线，不进入 bundle，不作为 fallback。
- `Logistic verifier` 不再由 Deep-only final 流程训练、选择、导出或加载。
- Deep promotion gate 失败时直接终止 Deep-only final，不静默回退到 Logistic 或 state-only。
- Boundary 仍是可选后处理；只有独立匹配事件数和端点门禁通过才进入 bundle。

新增配置为 [`hierarchical_v4_r32_deep_only_transition.yaml`](../../configs/hierarchical_v4_r32_deep_only_transition.yaml)，使用 transition candidates、单 state seed `2026` 和单 Deep seed `2026`。

## 2. 当前架构复核

### State backbone

- ACC 与 GYRO 使用独立局部 CNN/causal TCN，另有方向不变模长/变化量分支。
- PPG 使用 session-phased 15 秒分块；统计分支使用 12 项稳定统计量及 missing indicators。
- 统计分支经过 causal TCN 与 GMU/gate，再与 motion/PPG 融合。
- 五个 3 秒 token 聚合为 15 秒 long-context token；short/long 分支保留不同时间尺度。
- mask-aware temporal blocks 在卷积、归一化、残差后清零无效位置；最终输出 `state/onset/offset` logits。
- 当前配置的短上下文约 381 秒、长上下文约 1905 秒，未来信息限制仍为 60 秒。

### Deep verifier

- 输入为候选前后上下文、候选内部 16 bins、mask、统计/质量/gate 标量。
- 使用 mask-aware depthwise temporal blocks、masked mean/max pooling。
- 同时预测 eventness 与 IoU，再由 ProposalCalibration 生成 `final_score`。
- 训练和选择按 subject 隔离；五折 pooled final 使用外层 holdout cross-fit，避免预测 fold 参与 verifier 训练。

本次额外修正了 final pooled 路径：此前 `pooled_heads` 在调用 Deep cross-fit 前没有构建完整的隔离 `PooledFoldProposalData`；现在 Deep-only 会先按“其余四折校准/decoder/候选，当前折预测”的方式构建五份数据，再进行 Deep cross-fit。

## 3. 现有实验到底证明了什么

已有单折 diagnostic OOF 的记录显示：

| head | F1 | recall/sensitivity | FP/h |
|---|---:|---:|---:|
| state-only | 0.4638 | 0.64 | 0.1545 |
| Logistic | 0.4480 | 0.56 | 0.1296 |
| Deep | 0.4964 | 0.68 | 0.1462 |

在该单折样本上，Deep 相对 state-only 的 F1 提升 `+0.0326`，异侧 sensitivity 从 `0.4643` 提升到 `0.5357`，同时 FP/h 略低于 state-only。这支持将 Deep 作为主部署 head。

但不能把结论扩大为“Deep 在所有指标都优于 Logistic”：Logistic 的 FP/h 和边界误差更好。并且目前证据仍是单折/开发诊断，完整五折 pooled OOF 才能决定最终门禁是否通过。Deep 也不能补回 state 阶段没有生成的候选，因此候选召回必须单独报告。

## 4. 为什么保留这些模块

### 不能删除 state-only 生成器

这里的 state-only 不是一个可替换的部署 head，而是 state model 的 generator score。候选生成依赖它的概率、hysteresis/decoder 和 transition hints；删除会让 Deep 没有 proposal 输入，直接降低上限。

### 可以删除 Logistic

Logistic 只对已生成 proposal 的标量特征做线性重加权。Deep 已经使用同一候选集合的序列、mask、上下文和 eventness/IoU 联合预测；在当前 diagnostic 中 Deep F1 更高，因此 Logistic 不再有部署价值。旧 Logistic 代码和历史产物保持只读兼容，不会被新的 Deep-only bundle 选入。

## 5. 最值得尝试的模块内优化

下面按“理论收益/实现风险/当前适配度”排序。它们都应先做单折或两折 subject-disjoint 消融，未通过门禁不并入主线。

### P0：proposal-aligned latent bridge

当前 verifier 主要看到 proposal-level 概率、统计和 gate，state backbone 的隐藏表示在 state logits 后被压缩。建议把 state hidden `h_t` 沿 proposal 区间做 masked pooling，并加入：

1. proposal 内 `mean/max/last(h_t)`；
2. proposal 前后上下文差分；
3. onset/offset 附近 hidden summary；
4. `h_t` 与 PPG/GYRO/statistics gate 的乘积或差分交互。

实现上可把 pooled latent 拼到 Deep scalar branch，序列 branch 保持不变。它直接改善“候选来自哪个表征状态”的可辨识性，理论收益高于继续扩大 Logistic 特征。风险是参数增多和 state/verifier 特征对齐错误；必须用 `proposal_id`、timestamp 和 mask 做一对一校验。

### P1：boundary-quality joint score

将 onset/offset probability、state probability derivative、候选持续时间、候选与 transition hint 的距离，以及 Deep predicted IoU 组成 boundary-quality 分支，作为 Deep scalar 输入或轻量辅助 head。该方案借鉴 BMN 的 boundary matching 思路，但不允许新增、删除或合并事件；Boundary 只负责端点精修。

它能让 Deep 在“状态像事件但端点不可信”的 proposal 上降分，理论上可同时减少异侧误检和边界 MAE。必须避免把后处理选出的 truth 信息反灌入 verifier。

### P1：gate-conditioned cross-modal fusion

保留 ACC/GYRO/PPG 独立编码，在融合层增加低初始化的 cross-modal gate：

```text
q_motion = MLP([acc_hidden, gyro_hidden, valid_fractions])
q_ppg    = MLP([ppg_hidden, ppg_quality, ppg_valid_fraction])
alpha    = sigmoid(MLP([q_motion, q_ppg, statistics_gate]))
fused    = alpha * q_motion + (1-alpha) * q_ppg
```

当前已有 gate，因此建议只增加质量条件和 modality-specific residual，不重写整个网络。GYRO 缺失时 gate 必须只使用 ACC 有效比例和 missing token，不能把 ACC 有效率伪造成 0.5。

## 6. 最值得尝试的模块间连接

### P0：state → proposal → verifier 的 latent 传递

这是当前最可能带来明显收益的跨模块改动。连接应是单向、可审计的：state hidden 只按 proposal 的观测区间池化，Deep 不反向改变候选生成；这样保持候选 identity、IoU 规则和 60 秒因果边界不变。

### P1：decoder uncertainty → Deep verifier

除 generator score 外，把 decoder 的连续状态置信度、hysteresis 进入/退出 margin、transition hint 距离、片段碎片数作为结构化输入。它能使 Deep 区分“单点高概率噪声”和“连续稳定状态”。这些量必须只来自预测轴，不可读取 truth 边界。

### P2：Deep → boundary 的质量传递

将 calibrated eventness、calibrated IoU、boundary-quality logit 作为 EndpointRefiner 的 scalar conditioning。EndpointRefiner 仍只输出 offset 和 entropy；高熵、全 mask、`start >= end` 继续回退 coarse endpoint。这样可以减少低质量 proposal 上的过度精修。

## 7. 外部方法的适配性

以下来源的 DOI、题名、作者、年份和相关性元数据已通过本地 `validate_source_records` 校验；具体结论只依据来源摘要/公开元数据，尚未把它们当作本项目性能证据。

| 方法 | 可迁移设计 | 推荐接入点 | 当前优先级 | 主要风险 |
|---|---|---|---|---|
| MixStyle，Zhou et al., IJCV 2023，DOI [10.1007/s11263-023-01913-8](https://doi.org/10.1007/s11263-023-01913-8) | 混合实例 feature statistics 做 domain generalization | state short-fusion 或 statistics branch 输出后 | P1 | 不得混合 mask/quality；subject 数少时可能破坏个体节律 |
| TS-TCC，Eldele et al., IJCAI 2021，DOI [10.24963/ijcai.2021/324](https://doi.org/10.24963/ijcai.2021/324) | 弱/强增强视图的 temporal/contextual contrastive pretraining | supervised state 前的 outer-train-only 预训练 | P2 | 训练成本较高；增强必须保持 ACC/GYRO 同步 |
| TinyHAR，Zhou et al., ISWC 2022，DOI [10.1145/3544794.3558467](https://doi.org/10.1145/3544794.3558467) | 轻量多模态协同与 saliency | 当前 ACC/GYRO/PPG gate 的协同残差 | P1 | 与现有 gate 功能重叠，可能只增加复杂度 |
| BMN，Lin et al., ICCV 2019，DOI [10.1109/iccv.2019.00399](https://doi.org/10.1109/iccv.2019.00399) | proposal confidence 与 boundary matching 联合建模 | Deep verifier 的 boundary-quality 分支 | P1 | 不应把 BMN 的 proposal 生成器直接替换当前 decoder |
| ActionFormer，Zhang et al., ECCV 2022，DOI [10.1007/978-3-031-19772-7_29](https://doi.org/10.1007/978-3-031-19772-7_29) | 多尺度局部 temporal attention 与边界预测 | long-context 与 short-context 间的轻量 local attention | P2 | 完整 Transformer 对当前小样本和显存预算不友好 |

## 8. 不建议现在直接加入的方案

- 完整 Transformer/Mamba 或时间序列 foundation model：会同时改变容量、优化和时延，无法在当前样本量下区分收益来源。
- adversarial domain adaptation：容易把真实佩戴差异当成应抹除的噪声，并增加 nested protocol 风险。
- 直接用 `wear_hand`/`hand_relation` 作为推理输入：会造成场景依赖，且不符合未知测试场景稳健性目标。
- 在 raw sensor 上做未经验证的镜像：ACC/GYRO 物理方向、PPG 质量和佩戴关系可能被错误改变。
- 立即重写 Semi-Markov 或 Boundary proposal 逻辑：当前首要瓶颈是跨受试者状态泛化，复杂后处理不能找回漏生成事件。

## 9. 推荐实施顺序

1. 先使用 Deep-only 配置跑通五折 state、pooled Deep cross-fit、bundle loader 和 raw replay。
2. 在完全相同的候选集合上加入 latent bridge；只比较 proposal recall、Deep F1、异侧 recall、FP/h 和 calibration。
3. 若 latent bridge 未提升，再试 boundary-quality scalar 分支；保持 Boundary 不改变事件 identity。
4. 最后才做 MixStyle 小消融；仅在 state latent 上启用，推理关闭，按 subject-disjoint fold 验证。
5. TS-TCC 和 local attention 属于有时间余量时的二期方案。

建议的晋级门禁：Deep F1 至少 `+0.010`，或 F1 下降不超过 `0.005` 且 FP/h 降低至少 `10%`；同侧/异侧 recall 均不得下降超过 `0.03`，候选召回不得下降，所有校准和阈值必须有限。

## 10. 已知限制

- 现有 Deep 优势主要来自单折 development/stress 证据；完整五折 pooled OOF 仍是最终判据。
- 当前报告没有宣称 Boundary 必然有效；独立匹配事件不足时必须保持 `boundary_enabled=false`。
- 统一 pooled OOF 联合选择 decoder、阈值和 head 会带来开发集乐观偏差，报告中必须明确 `joint_tuning=true`。
- 竞赛官方一对一匹配规则仍为 `UNKNOWN`；内部继续以 max-cardinality 为主、greedy 为敏感性诊断。

## 参考与本地证据

- 来源候选与检索/验证字段：[`statsfusion_r3_frontier_sources.json`](statsfusion_r3_frontier_sources.json)。
- 当前问题范围：[`statsfusion_r3_problem.md`](statsfusion_r3_problem.md)。
- 当前选定模型说明：[`statsfusion_r3_selected_model.md`](statsfusion_r3_selected_model.md)。
- Deep 单折报告（若产物存在）：`outputs/v4/experiments/hierarchical_v4_r32_pooled_heads_early_select_20260928a/fold_0/diagnostics/single_fold_heads/verifier_report.json`。
