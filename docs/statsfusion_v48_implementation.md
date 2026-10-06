# v4.8 分阶段实现与证据边界

旧版 `hierarchical_v4_r32_deep_frontier_20260930a` 的模型与五折文件只读保留；旧 ZIP 的 SHA-256 是 `8240110408f31cb9ad6a7797f9984bcdddd4b9630791cbc143f2601613c2de7d`。v4.8 使用新 run name，不能 resume 旧版。数据协议和 raw-v2 契约仍是 `statsfusion-r3.2`；代码代际、候选协议、提名头和可选 IMU 分支分别在 manifest 中标记。赛事隐藏集及官方匹配细节仍未知，以下全部是开发集/stress 证据。

## 已完成的低成本候选回放

只读旧版五折 outer 窗口、truth、候选和 duration prior，针对六种预注册机制与 `0.06/0.08` 高阈值回放，并对每个目标折仅用其余四折选机制。含按事件窗口判定 GYRO 缺失的修正结果在 `outputs/v4/diagnostics/v48_candidate_replay_20261003d/`：五折均选 `backtrack_600 + expansion_600 + gap_120 + high_0.06`；直接对照相同 outer 候选文件，覆盖从 `118/161` 升至 `128/161`，同侧 `58→62`、异侧 `60→66`，10 个 GYRO 全缺失事件从 `5→7`，所有折候选密度 `≤11.59/h`。旧版最终 pooled 候选的 `115/161` 是另一套候选池，不可与 outer 文件的 `118/161` 混为同一对照。旧版 transition 候选独有覆盖为 `0`；9 个小于两分钟的 truth 在这次候选回放中仍命中 `0`，因此才需要单独测试提名头。

## 分阶段配置

| 配置 | 只改变的实验因素 |
|---|---|
| `hierarchical_v4_v48_candidate_repair.yaml` | 17 种对称 jitter、回溯/扩展、关闭 transition、候选预算轮转 |
| `hierarchical_v4_v48_proposal_head.yaml` | 在上一行基础上增加 `proposal_logit` 和 0.1 提名损失 |
| `hierarchical_v4_v48_proposal_head_no_mixstyle.yaml` | 仅消融 MixStyle |
| `hierarchical_v4_v48_proposal_head_no_contrastive.yaml` | 仅消融 temporal contrastive |
| `hierarchical_v4_v48_raw_imu.yaml` / `hierarchical_v4_v48_proposal_head_raw_imu.yaml` | 在对应锁定候选池上测试低门控原始 IMU Deep 分支 |

所有 state 对照仍为 seed 2026、32 轮上限、至少 3 轮、连续 3 次无改进早停、epoch 1 起参与选择、学习率 `1.5e-4`、warmup `0.10` 和 `8×4` 有效 batch。新增提名标签在线按 `subject_key + session_id` 生成，不改变 canonical 特征提取文件，因而可复用旧 canonical 传感器输入；启用提名头时不能加载旧 state 权重。候选阶段若相对冻结 pooled 基线不足 5 个新增 truth、任一佩戴关系下降超过 2 个，或提名头未达到 `126/161`、异侧 `63/91`，最终训练会停止在 Deep 前。

Deep 一律在新候选池上重训，旧权重和校准不得复用。v4.8 Deep 显式读取 7 位候选来源，区分回溯、扩展与提名头。最终 Deep 还须满足 F1 `≥0.5475`、FP/h `≤0.0609237`、同/异侧召回相对冻结基线下降 `≤0.02`；报告 greedy 敏感性与 seed 2026 的 1000 次受试者 bootstrap。原始 IMU 分支读取起点/中点/终点附近 30 秒信号，最晚不超过候选结束后 15 秒，缺失 GYRO 用 mask 表示。它只有在相同候选 ID、边界且除 IMU 开关外配置相同的 Deep 对照中 F1 增加 `≥0.010`，或 F1 下降 `≤0.005` 且 FP/h 降低 `≥15%`，并生成绑定配置、分数与报告 SHA-256 的 `v48_raw_imu_gate.json` 后才允许导出。

Boundary 仅在 Deep 晋级后尝试。新增几何可修正比例和计数诊断；只有至少 60 个正事件在起终点均可修正时才训练。起点搜索按候选时长限幅至 900 秒，终点最远向前 900 秒、向后 60 秒；MAE 晋级只比较粗/细两版共同匹配的 truth，原有事件身份、熵、无有效 bin、冲突和 F1/佩戴关系门禁仍有效，失败保持 coarse boundary。

## 后续执行

单折后端诊断按完整 `oof/window_predictions.parquet` 时间轴确定评估受试者，而不是按候选表确定。没有候选的受试者仍保留 truth、ignore 和观测时长，其有效事件计为漏检；Verifier 与 Boundary 报告均记录 `evaluation_cohort`。旧版单折诊断使用原命令 `--resume` 时自动归档到 `single_fold_heads_before_cohort_fix_*`，复用兼容的 verifier checkpoint，重新计算校准、阈值、模型晋级与 Boundary。旧报告不覆盖，state 和候选无需重训；这些单折结果仍是开发集诊断，不是五折 pooled 结果。

先对候选修复版和提名头版分别筛查 fold 1、3。只对通过候选及分层门禁的配置开全新 run 跑五折：每 fold 顺序为 `train_hierarchical_v4_state.py --fresh`、`build_event_candidates_v4.py --resume`、`select_hierarchical_v4_pipeline.py --resume`、`evaluate_hierarchical_v4.py --resume`。五折完成后运行 `train_hierarchical_v4_final.py --fresh`，通过绝对门禁后再 `export_hierarchical_v4_bundle.py --fresh`。IMU 实验先跑无 IMU 的完全相同候选配置，再用 `compare_v48_raw_imu.py --reference-final-root ... --candidate-final-root ...` 生成比较证据。不要将候选回放的 128/161 误认为最终 Deep 命中数；五折新训练、GPU smoke、真实 raw-session 与无 XGBoost bundle replay 尚待执行。
