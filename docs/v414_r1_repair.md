# v414-r1修复交付与验收

## 身份与隔离

模型保持`statsfusion_joint_event_v414`，协议改为`statsfusion-joint-event-v6`，恢复状态版本为2。正式新run为`statsfusion_v414_28_11_then_39_repair_20261013a`，seed为2026。修复前HEAD为`1c11b6652c2019a91c6aa7f08d5687ae022698cc`。

修复基线在仓库外保存源码副本、HEAD、逐文件hash、旧run身份和文件元数据，以及v4.13源码hash。冻结划分hash为`d5c3b2e6faa67bad587e57018dfcb24ddf070b626312a3bd5a61030cfd162e0f`；成员、原始输入和canonical配方保持不变。旧协议产物不能恢复到修复后的训练。

## 修复对照

| 原问题 | 修复后的实际路径 | 验收方式 |
| --- | --- | --- |
| 无峰session额外扩大端点背景贡献 | 每epoch独立资格计数，按W/W_peak及W/W_bg平衡 | 含峰比例与权重变化、完整session/chunk计数一致 |
| 纯负组排序未进入更新 | 完整任务映射进入负贡献银行，合法辅助任务单独归一化 | 手算梯度与optimizer更新 |
| 一个重复版本导致整组丢弃 | 按版本去重，逐版本/排序对CPU梯度可重建子集 | A/B反例保留独有版本并移除失效排序对 |
| 监督几何跨池重复 | 自然/随机合池去重，正组所有权随替换更新，纯负锁定 | 重复几何和恢复测试 |
| 训练修正几何与off部署不一致 | Boundary保留合法正监督，分类/quality/排序只使用粗几何 | 标签、Boundary连接及修正调用检查 |
| 历史长事件缺少背景负例 | 每事件最多12个历史内部bin，加最多12个历史背景bin | 长事件对比任务真实配对与梯度 |
| run身份影响同分选择 | fallback、采样、预算及NMS使用稳定物理键 | 不同身份同输入的几何一致 |
| 空训练CDF导致流程异常 | 专用空候选异常记为无资格，保留既有截止规则 | 空CDF、身份错误不吞掉、验证中断计数 |
| 登记decision与真实提交时间混淆 | regular即时提交，EOF单独提交，记录批量输入可用时刻 | 3秒增量、EOF、不同chunk与未来前缀 |
| 诊断与实际贡献不一致 | 合法任务资格显式登记，峰/背景分项、保留集合目标及VJP耗时 | warmup、零梯度夹角、finite与实际更新审计 |

## 不变项与证据边界

模型容量、优化器、BF16、256新步、历史detach、session固定权重、4-session累计、增强安排、64轮LR时间轴及16038个操作点保持。开发训练选择E，最终39人训练至E+4，保存五档邻近checkpoint，默认部署E。Boundary部署固定off。

最终39人模型的11人校准数据参加过训练，继续登记为非独立开发校准，所有产物`promotable=false`。回归、GPU及8轮短审计用于验证实现，不证明F1改善。

## 验收记录

最终结果登记于`manifests/v414_r1_repair_validation.json`；CPU/GPU前后向及长事件重放分别登记于`manifests/v414_r1_validation_cpu.json`和`manifests/v414_r1_validation_gpu.json`。5个真实session、8个可丢弃epoch的生产循环审计登记于`manifests/v414_production_loop_cuda.json`。正式训练只能在验收全部通过后启动，审计权重不进入正式run。
