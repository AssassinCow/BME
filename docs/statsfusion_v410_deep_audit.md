# v4.10 全链路深入审查与修复

## 结论与范围

本次发现并修复了会影响边界晋级判断、OOF 恢复、产物身份、失败状态和部署安全的实际缺陷。
完整测试为 **484 passed / 1 skipped**；CPU/GPU 独立 smoke、长事件、native raw replay、BF16 backward 和无 XGBoost 回放通过。
这证明本次代码验收通过，不证明正式五折 F1、覆盖、边界 MAE 或晋级目标已达到。

修改仅在隔离 checkout `C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v410`。
原 `modeling` checkout 的 git status 仍为空；没有创建正式 run、启动正式训练、覆盖冻结产物或提交 Git commit。
模型容量、共享编码器路线、单 seed=2026、原五折 subject split 与三分区完整联合模型内部 OOF 维持已批准协议。

## 已发现的缺陷与修复

| 范围 | 原缺陷及影响 | 修复 |
|---|---|---|
| Boundary 成对证据 | 仅按 truth ID 配对；另一 proposal 替代原 TP 会被误认为同一候选改进，并低估 TP→FP | MAE 同时锁定 subject/session/truth/proposal；TP→FP 追踪原 matched proposal |
| Boundary 诊断总体 | clipping/conflict/valid-bin 只统计最终提交事件，遗漏被阈值或 NMS 拒绝的异常候选 | 完整预算后候选池进入诊断；提交器保存所有接受/拒绝候选的 fallback 记录 |
| Boundary 改变量 | `changed_endpoints` 实际统计变更候选数 | 分开记录 changed candidates 与实际 start/end 改动数 |
| Boundary entropy 搜索 | 每轮改写 point，后续 entropy 相对前一轮结果比较，产生顺序依赖 | 所有 entropy 使用固定 coarse 基线，循环结束后一次锁定 winner |
| Boundary 最终影响 | 只用 refined calibration/threshold 的 coarse 对照，可能掩盖相对原锁定操作点的退化 | 保留纯修正对照，另检查相对固定 coarse 操作点的最终 F1 降幅 |
| Boundary 邻居冲突 | 起点修正可能侵入原先不重叠的已提交邻居 | offline OOF 与在线部署共用提交器，恢复实际未修正区间及 coarse event/quality 的校准分数后再次执行接受和 NMS |
| 内部 OOF 恢复 | point 与 `.pt` 存在就复用，没有完成标志、哈希或新 run 绑定 | 原子提交 internal manifest；复用前检查 run/config/data/source/cohort 与所有证据哈希 |
| OOF 排除身份 | 未拒绝重复 partition/model、空 fit/selector；仅信任 JSON 的 qualified 字段 | 检查唯一性、完整排除，并读取实际 retrain/selector checkpoint 验证 finite、clipping、epoch、训练/排除与预处理主体 |
| outer 恢复 | `evaluation.json` 存在就视为完成，prediction/truth/windows 等可来自部分流程 | 最后提交 completion manifest；恢复和 gates 校验完整 fold 证据和实际 checkpoint |
| cohort 校验 | 总 truth 数正确不能证明 truth/ignore 是当前 canonical cohort | 逐折比较完整 canonical truth/ignore，另验证候选/预测 model 身份和一人一份时间轴 |
| 晋级锁 | `promotable=true` 和少量文件即可通过 | 检查全套正式 checks、五折证据、合格且实际执行的配对参考；缺证据即拒绝 |
| 部署加载 | `weights_only=False` 可执行 checkpoint 中的任意 pickle；训练状态也不应进入部署包 | exporter 生成精简部署 checkpoint，运行时只允许安全 weights-only 加载；回归测试证明恶意 pickle 未执行 |
| bundle 身份 | 文件哈希清单可能漏掉额外源码，实际最终训练 cohort 未与五折锁核对 | 校验完整文件清单、冻结源码、数据、内部 lineage、最终训练主体和预处理主体 |
| 导出恢复 | `.partial` 目录可复用残留文件；已有 bundle 的 resume 可跳过再审计 | 残留 staging 移至 failure 诊断目录；现有 bundle 只有重新审计通过才能恢复 |
| run 路径及写入 | `.`/`..` 是合法正则结果；多个 resume 可并发破坏 checkpoint | 拒绝逃逸名称；非阻塞 OS 文件锁包围实际执行和 resolved config 写入 |
| runner 完成状态 | 单折 all 可继续 gates；单阶段成功可覆盖其他失败；已完成阶段可能无验证跳过 | 单折不足五折时 PAUSED；保留 FAILED；关键阶段恢复重新检查证据 |
| 训练恢复 | 未绑定 epoch 目标、selector/exclusion；完成但无可用 epoch 可返回 None；游标可越界 | 保存并核对训练目标/排除主体，明确报失败，限制游标只恢复已提交 traversal |
| source/data 变更 | 只检查原文件 mtime，新增源码不被发现；训练和预测中输入变化未全面拦截 | source 文件集合/size/mtime 改变触发哈希复核；监控 canonical、split、segments 和当前 session 原始 archive |
| loss 与配置 | smooth 没有使用独立 ignore mask；Boundary sigma/范围/独立正事件门槛部分硬编码 | smooth 使用 smooth_loss_mask；Boundary 读取 resolved config，独立事件用 subject/session/event 身份 |
| 数值和输入 | NaN 分数可绕过 `< threshold` 判断；预算、CDF、calibration、时间轴缺少合法性检查 | 提前拒绝非有限值、非法概率、错位标签、非法预算/配额和缺失时间 bin；同 proposal 不能重复消耗预算 |
| 已提交事件 | 提交列表持有原候选，后续操作点搜索可修改其分数或参数 | 保存候选独立副本，保证已提交结果不被外部修改 |
| 搜索效率 | 预算与操作点搜索反复遍历整个曝光时间轴 | 按 session 预计算有效曝光累计量，使用 searchsorted 定位决策；操作点网格共用已计算总曝光 |

## 验证内容与实测

### 测试与静态检查

- 完整 pytest：484 passed、1 skipped，101.44 秒，启用 GPU 与真实 Windows worker=10 探针。
- 唯一 skip：既有 canonical 输入缺 clarified mask-count diagnostics 的旧测试。新联合模型资源测试实际执行。
- Ruff、compileall、工作区与暂存区 diff whitespace 检查通过。
- 审计新增回归覆盖：边界替换、被拒绝候选 clipping、邻居冲突 coarse 分数、NaN、重复 proposal、曝光等价、路径逃逸、run 写锁、OOF manifest/实际 checkpoint、输入变更和门禁缺证据。
- 恢复测试注入 Ctrl+C/OOM，检验部分累计梯度、RNG、session cache 和连续训练权重逐元素一致。
- 临时实际 exporter 测试验证精简 checkpoint、bundle 加载、恢复重新核验、禁止覆盖及额外文件拒绝。
- 无 XGBoost 子进程完成 bundle 加载及 native raw replay；该证据使用测试合成 bundle，不是正式部署产物。

### 真实接口的小规模 GPU 训练

使用测试生成的 NPZ archive、canonical anchors/statistics/events 与 segments index，实际经过 JointInputs、JointSessionDataset、DataLoader 和正式 train_joint_model。
执行三个 epoch，实际使用 CUDA BF16 autocast、部分 GYRO 缺失、epoch 3 truth-jitter 和事件/质量损失。
确认共享编码器参数改变、事件和质量加权梯度非零、所有模型权重有限、安全 checkpoint 可加载且完成训练可恢复。
此探针使用 worker=1；另一个实际 worker=10 测试验证进程启动及游标恢复。两者均不代表全量数据的 RAM 峰值。

### 独立 CPU/GPU smoke

| 项目 | 结果 |
|---|---|
| 长事件 | 13023 秒；17/256/整段 chunk 的事件身份一致 |
| 同输入重复推理 | 最大误差 0，满足 ≤1e-6 |
| native raw 跨 chunk 事件分数 | 最大误差 0 |
| native raw 跨 chunk 特征 | CPU 2.9802322e-6；GPU 2.3841858e-7，均小于特征舍入容限 1e-5 |
| BF16 事件/质量共享梯度范数 | 0.04441293 / 0.01058719，均有限且非零 |
| GPU allocated / reserved 峰值 | 1.1961 / 1.3887 GiB |
| GPU smoke 进程 RAM 峰值 | 1.9229 GiB |
| 硬件/运行环境 | RTX 4080 Laptop GPU；Conda bme-model；Python 3.11.16；PyTorch 2.5.1；CUDA 12.4 |

验收 JSON 记录 config/source SHA-256 与运行环境，并明确 `synthetic_validation=true`、`formal_five_fold_training=false`。
文件为 `statsfusion_v410_cpu_validation.json` 和 `statsfusion_v410_gpu_validation.json`，与本报告同目录。
修改源码后不能把这些记录作为新源码的验收证据，应重新执行验证。

## 使用与尚未验证的部分

正式启动和恢复命令见 `statsfusion_v410_joint_event.md`。worker 默认10，只有 worker/prefetch/cache 可在 resume 时调整。
模型、loss、data、source、seed 或协议变化必须新建 run；不能将修复后的源代码用于继续旧身份的训练。

没有运行正式五折、专用部署 OOF 或最终全量训练，因此不能报告新模型的比赛成绩、提升幅度或最终显存/RAM。
当前合成 smoke 的低显存不能替代正式 workload 的 `<10.5 GiB` 门禁；正式训练遇到超限立即失败，不自动缩小协议。
历史参考不满足完整配对条件时，gates 保留诊断并拒绝 final/export；S0 继续显式跳过，不能据此宣称优于 sensor-only。
内部 selector/OOF 的搜索后成绩属于开发证据；正式效果须以未参与任何搜索的五折 outer prediction 为依据。
训练恢复缓存包含本地内部 Python 状态，必须视为本 run 的受信任产物；跨 run 复制不允许，部署 checkpoint 已移除这些状态。
哈希用于身份与误修改检测，不替代外部签名认证；本次安全验收包括安全 checkpoint 加载，不作所有第三方依赖的安全保证。
