# 模型与推理

## 共享表示

主干宽度64，保留ACC、GYRO、方向不变运动、PPG质量门控、12项统计特征和缺失分支，以及3/15/60秒proposal路径。语义表示用于事件存在判断；短感受野定位表示复用分支block输出，经过两层因果深度可分离卷积，供onset、offset和Boundary使用。

完成的proposal窗口按有效权重池化，有效率单独表达；空窗口在时序输入、每层残差输出及最终proposal输出清零。可选mask接口保留参数形状及state_dict键，合法过去输入仍按原卷积感受野访问。

时间缓存为201维：语义64、原局部64、9个时间线/有效性标量、定位64。缓存绑定模型、数据、epoch、subject/session和处理视图；历史表示detach，只有当前chunk保留梯度。候选坐标为离散生成，属于截断联合训练。

## 候选判别

ROI为40个259维tokens，区域布局`4/16/4/8/8`：前方、内部、后方、起点局部、终点局部。每个区域独立执行共享时间编码器，再加入区域embedding。token保存mean/max、传感器有效率、位置与已观测覆盖率；无效输入在各层遮罩。

四头query pooling和五区域summary、四组内外差分通过轻量比较通路融合，输出64维candidate embedding、event logit和quality logit。事件接受使用校准事件概率；quality只用于粗区间内几何版本选择。

occupancy负责粗候选；合法onset峰提供学习起点，无峰时兼容回退。自然回放从epoch3开始，在固定session权重、eval/no_grad下生成候选；训练仅释放decision已经到达的候选，正式loss使用该session一致增强。困难负例由事件头BCE排序，随机背景先保留名额。

学习起点默认最多保留三个唯一几何：粗区间、一个兼容版本、一个学习版本。兼容版本从原无峰回退集合选择，优先结束坐标最接近粗区间，再按稳定物理hash排序；不用标签选择。学习起点与已有几何重复时合并来源、不占额外名额，再用兼容版本补足空位。无峰行为保持，no_onset消融仍保留原兼容池。固定容量下只能保护部分兼容几何，也减少了双峰时的学习版本数，不能保证所有事件覆盖不退化。

## 训练贡献

组内版本平均、事件内组平均、session内事件平均；纯负按实际family平均。累计4个session时，正负分别按有贡献session数平均，两类均存在时各占一半。Boundary/排序仅对合法事件或负family及有贡献session平均，纯负排序进入实际参数更新。受试者权重作用于实际贡献，校准保持自然候选分布。

正组共享几何时只删除重复版本并重新计算优先级，保留独有版本。CPU缓冲按版本保存分类/Boundary梯度、按排序对保存梯度，组替换或成员裁剪时精确移除旧贡献并重新归一化；最多3个loss分块batched VJP，不支持时确定顺序回退。所有实际监督几何在session内去重，纯负提交后锁定所有权。候选fallback、预算和NMS的同分顺序使用稳定物理键，独立于run身份。

正事件组是归一化单位，组内各版本仍逐一按严格IoU>0.25判定分类标签。去重后没有合法正版本的组退出正额度和贡献计数。纯负类别按过滤后的实际版本重新判断，不能把仅剩零IoU的组记为near-miss。

端点单侧Gaussian热图使用session级峰值/背景计数及epoch训练资格权重归一化，固定背景比20，避免大量无峰session额外扩大背景贡献。阈值命中与经9秒确认的真实可用起点峰分别报告。GYRO/PPG丢弃、旋转和因果低带宽增强在raw、统计、不变特征及mask中一致，同session跨chunk和恢复使用相同描述。

Boundary从epoch3保留合法正候选的定位监督；分类、quality和排序仅监督未修正几何，训练不再自动修正重评分。对比任务从epoch6启用，历史每事件最多12个内部bin，并均匀保留最多12个有效非ignore、远离边界的背景bin；历史detach且背景不作为正anchor。

历史reference总量独立限制为128，当前anchor上限仍为128。超限优先保留背景，按事件/phase轮转分配正历史；索引与标签对齐后按时间排序。冻结数据每人最多8个事件，现有历史采样最多108个bin，此上限不会截断当前合法session。

起点峰确认要求七个连续3秒bin均有效，维持9秒确认和原左右比较规则。GYRO缺失但ACC有效仍允许确认；EOF不足时不补未来。生成与诊断使用同一检测器，ignore仅用于诊断计分。

## 部署接口

```python
from bme_eating.bundle import StatsFusionJointFinalDataDetector
from bme_eating.data.stats_fusion_preprocess import RawSessionInput

detector = StatsFusionJointFinalDataDetector("path/to/run/bundle", device="cpu")
events = detector.predict_session(raw_session)
```

`raw_session`必须为包含时间戳、传感器值、有效mask及subject/session身份的`RawSessionInput`。数组字段以该dataclass定义为准，不能从label构造推理输入。分批原始输入可用`new_raw_session()`及其`push(chunk, final=...)`接口；构造、追加和结束标志以方法签名为准。

部署只执行锁定操作点，Boundary固定off。按`observe → regular提交 → EOF finish → eof提交`即时处理，`decision_phase`用于区分同一时刻的两次提交；离线预算、搜索及部署顺序一致。每个候选满足`max_input_time ≤ decision_time`；流式NMS不可撤回已提交事件，登记decision与reported end相差不超过60秒。严格`IoU > 0.25`才是匹配。批量评分和缓存必须保持逐候选decision遮罩、元数据身份及原排序。

`input_available_time_ms`与`emitted_at_ms`记录调用时实际可用数据的最晚时间。3秒增量推理验收实际返回延迟≤60秒；大批离线输入单独登记其批量可用时刻，其返回延迟可能大于60秒，不能以候选decision代替真实输入可用时间。
