# v414-r2修复与验收

模型/schema保持v4.14/v414，协议为`statsfusion-joint-event-v7`，修复标识`v414-r2`，checkpoint状态版本3。新run为`statsfusion_v414_28_11_then_39_repair_r2_20261013a`，seed2026。r1原始修复、验收报告和旧run保留。

## 行为修复

| 问题 | r2行为 |
| --- | --- |
| Proposal空位置的bias/历史响应进入后续卷积 | 窗口输入、逐层输出、最终logit按有效mask清零 |
| 起点中心有效但邻居缺失仍确认 | 七bin均有效且相邻间隔3秒，EOF不补未来 |
| 去重后只剩负版本仍占正组额度 | 至少保留一个IoU>0.25的版本才进入事件组贡献 |
| 过滤前near-miss标签沿用到零IoU剩余版本 | 按实际剩余版本重新分类和分层 |
| 跨decoder/decision自然候选重复 | 首次可用物理代表固定，晚到来源仅按时释放诊断 |
| 预算去重依赖CDF fallback | 全路径使用模型/subject/session/start/end键 |
| 历史reference没有独立上限 | 上限128，背景优先，事件/phase轮转；当前数据不截断 |
| 旧身份与只读数组警告 | 活跃生成登记v414/v7/r2，入口模式与phase分开，复制时间戳切片 |

正事件组允许混合正负版本，分类和quality继续逐版本计算。r1端点跨session平衡、负排序梯度、原子梯度缓冲、空CDF和Boundary off保持；本次不重新调权重。

## 数据与部署

冻结28/11名单及输入只读复用：799/354 session，116/45事件。39人最终从头训练到E+4，保存五档邻近checkpoint，默认校准与部署E。Boundary固定off，最终11人校准为非独立开发证据；全部产物promotable=false。

模型容量、AdamW、BF16、256新步、历史detach、session固定权重、4-session累计、增强及64轮LR轴、16038点保持。旧协议checkpoint、optimizer和缓存不能加载或resume到r2。

## 验收与限制

本轮基线在仓库外保存HEAD、暂存/未暂存diff和逐文件副本/hash；保护旧run全部文件及v4.13源码。交付报告为`manifests/v414_r2_repair_validation.json`，记录最终源码/config身份、检查结果、失败/未执行项及保护核验。

执行完整pytest、ruff、compileall、diff、CPU/GPU前后向、恢复、长历史及无XGBoost bundle重放；随后运行5真实session/8可丢弃epoch审计及原始长短session重放。可选完整搜索对照保持同权重/候选/16038点，仅报告搜索阶段性能。

审计权重不进入正式run，不自动启动正式训练。实现验收不保证F1改善；失败或未执行项必须明确记录，不能标记全部验收通过。

## 实测验收记录

| 检查 | 结果 |
| --- | --- |
| 完整测试 | CPU 255通过、18跳过；18项随后在CUDA及资源检查环境全部通过，共273项无遗留跳过 |
| 静态检查 | ruff、compileall、git diff --check通过 |
| GPU前后向 | BF16通过，候选头batch1/128最大误差2.61e-8 |
| 模型兼容性 | 参数767197个；相同seed初始化、参数键及形状保持，三尺度全有效输入输出与r1逐位一致 |
| 真实生产循环 | 5个session、8个可丢弃epoch、10个worker；每轮2次optimizer更新，端点预期/实际计数一致 |
| 任务贡献 | 合法分类、quality、Boundary、排序及端点头均有真实提交梯度和参数更新；epoch6–8每轮820个有效对比配对 |
| 审计资源峰值 | GPU已分配1.448 GiB，进程树private RAM 17.123 GiB；低于10.5/26 GiB门槛 |
| 隔离保护 | 旧run 11988个文件的hash/mtime、v4.13的151个源码hash和冻结划分保持 |
| 真实原始数据回放 | 29秒、13022秒事件所在session按128/256新步回放，事件身份一致；长session分数最大误差3.73e-9 |
| 完整搜索对照 | 相同r2权重、909个候选、16038个操作点，逐项TP/FP/FN/F1/FP每小时无差异 |

审计epoch1–4裁剪比例0.50，epoch5–8为0；这些是小样本审计结果，不证明正式训练收敛。Ctrl+C/OOM验收采用故障注入及恢复等价比较，不主动耗尽设备显存。无XGBoost验收使用测试生成的bundle，不能视为正式39人模型已训练或导出。

四个真实session的搜索对照：参考路径82.233秒，缓存路径3.673秒，搜索阶段约22.39倍。该结果使用一致性检查用的固定校准器；完整验证的两倍加速尚未得到端到端实测证明。proposal局部CPU基准每512个bin的单尺度新增约0.11–0.23毫秒，仅适用于该局部测量。

本轮实现验收通过。完整测试、真实审计、回放和搜索证据的hash保存于交付manifest，正式新run仍未启动；必须使用新run及`--fresh`，不能续接r1。
