# StatsFusion v4 开发、排错与实验简史

> 截至 2026-10-03。本文按本次对话和本地运行产物整理；“提出/计划”不等于“已验证有效”。当前导出 run 为 `hierarchical_v4_r32_deep_frontier_20260930a`，代码 `v4.7.1`、协议 `statsfusion-r3.2`。

## 时间线

1. **早期 v4 / r1：先修评估可信度。** 初始路线是因果多尺度 motion state 网络、统计特征融合与可选 PPG，再由候选、Verifier、Boundary 输出事件。审查发现 masked bin 影响卷积、高熵 Boundary 回退仍改写边界、meta-OOF 不完全嵌套、3/15 秒网格右端点不一致、校准忽略 loss mask、候选预算跨 session 竞争等。随后加入 mask-aware 处理、统一时间语义、嵌套受试者 lineage 和 session 级预算。连续标签 AUPRC 报错、缺少进度显示也在此阶段修复。
2. **r2 深审：修时间轴和缓存身份。** Semi-Markov 逐点 fixed-lag 回溯可能拼成超过时长上限的事件；训练与 raw 推理 anchor 相位、15 秒 PPG 块及长上下文的 5-token 分组相位随 clip 改变。另有 selector 种子、历史统计 padding 缺失位、jitter 越界、pooled FP/h 加权、bundle 哈希/隐私和 canonical cache 只凭文件大小/mtime 的问题。修复方向是 session-wide 3 秒右端点网格、固定 session-relative 分组相位、内容/源码哈希身份及严格重复 anchor 核查；旧 r0–r2 产物不再混用。
3. **r3 正确性协议：把标签、分数和候选对齐。** truth/ignore 按受试者及 session 分区，state/onset/offset/smooth 分别使用监督 mask；平滑项改为非饱和 Huber。保留软 occupancy target，用 Soft-Platt 与软 Brier/ECE 校准；Semi-Markov 改为一致的合法 segment 解码，候选硬时长与 duration prior 分离，decoder 在训练 OOF 上选择。补充 raw-v2 输入契约、GYRO/PPG 缺失诊断、重复 anchor 冲突失败和缓存/产物来源校验。跨受试者、异侧佩戴和 GYRO 缺失被识别为主要泛化风险；独立模态编码、方向不变分支和物理旋转等当时属于待门禁消融，不能仅凭方案认定有效。
4. **r3.1–r3.2：时间约束下改变训练。** 三 seed、每折三 partition 的成本过高，转为单 seed `2026`、每折一个受试者隔离 holdout；selector 最多 32 轮、至少训练 3 轮、连续 3 次无改进早停，epoch 1 起可参与 checkpoint 选择，再按所选轮数从头重训。沿用每轮验证、学习率 `1.5e-4`、warmup `0.10`、batch `8` × accumulation `4`。训练中出现高梯度裁剪、chunk 重叠 logits 不一致、snapshot/feature provenance 冲突和 CPU decoder 搜索耗时，分别补充诊断、时间轴一致性与可恢复身份管理。早期 partition 0 的 **11/31 人诊断性 partial OOF** 候选召回 `29/44=0.6591`、异侧 `0.5833`；48 组 decoder 未提高召回，提示 state 跨人泛化瓶颈，但它不是完整五折结果，也不能证明 retrain 单独导致过拟合。
5. **后端路线：由诊断走向 pooled Deep。** 讨论过 state-only、Logistic、Deep Verifier 与 Boundary；Verifier 对完整候选池打分，不能找回候选池没有覆盖的 truth。用户基于阶段性结果选择 Deep-only 主线；引入 learned-query Verifier pooling、轻量 state temporal contrastive regularization，以及 proposal-conditioned Boundary。五折 outer OOF 后为 pooled head 构造排除预测折的 nested state OOF，选择阈值并做跨折 Deep 评分；Boundary 仅在通过门禁时启用。单折 Deep 结果不能代表五折总体。
6. **最终训练故障与恢复。** nested state 推理曾发生 GPU OOM，调整 bf16/autocast、推理 batch 与缓存释放；随后 fold-local 校准遗漏 `onset_probability` / `offset_probability` 导致候选生成失败，补字段并定向迁移缓存后用 `--resume` 继续。最终训练完成后，首次导出误用 `--resume`（bundle 尚不存在）而报错，改 `--fresh` 导出成功。代码/配置改动允许记录源码身份后恢复，但结果相关数据、划分、标签和父 artifact 哈希仍需核验；混合源码 run 不能被当作严格同一代码版本复现实验。

## 当前实验结果（压缩版）

| 口径 | 候选召回 | 事件 F1 | 说明 |
| --- | ---: | ---: | --- |
| fold-local state-only，fold 0–4 | `0.8125 / 0.6250 / 0.6061 / 0.8750 / 0.7500` | `0.4932 / 0.3860 / 0.5357 / 0.6316 / 0.5172` | 各折独立选择；state epoch 为 `3/3/8/4/4`。 |
| pooled OOF state-only 对照 | — | `0.4491` | TP/FP/FN=`75/98/86`；precision/recall=`0.4335/0.4658`；FP/h=`0.07435`。 |
| pooled OOF **Deep（入选）** | — | **`0.5375`** | TP/FP/FN=`86/73/75`；precision/recall=`0.5409/0.5342`；FP/h=`0.05539`。 |

Deep 命中事件的起点/终点 MAE 为 **`218.20/153.66 s`**；pooled state-only 对照为 `224.25/179.67 s`，但两者命中事件集合不同，不能把差值当成严格配对边界改善。Deep 同侧/异侧召回 `0.6429/0.4505`（state-only 为 `0.6429/0.3297`）。最终阈值 `0.12926`、NMS IoU `0.3`、Semi-Markov 关闭。Boundary 虽有 115 个独立正事件，但 fold 0 的 residual range 裁剪过多，**未通过门禁，最终保留粗边界**。导出 manifest 为 `EXPORTED`，bundle ZIP 与清单哈希一致。

**证据边界与来源。** 以上 pooled 分数是按受试者隔离预测、但在完整五折 OOF 上联合选择 decoder/head/阈值的 **development/stress evidence**，可能偏乐观；不是官方隐藏测试或独立外部验证。内部以严格 `IoU > 0.25` 和 max-cardinality 一对一匹配为主；官方具体一对一规则仍为 `UNKNOWN`。可复核原始记录见 `${BME_OUTPUT_ROOT}/v4/final/hierarchical_v4_r32_deep_frontier_20260930a/` 的 `selected_pipeline.json`、`boundary_crossfit.json`、`final_manifest.json`，以及 `${BME_OUTPUT_ROOT}/v4/experiments/hierarchical_v4_r32_deep_frontier_20260930a/fold_{0..4}/evaluation/metrics.json`。
