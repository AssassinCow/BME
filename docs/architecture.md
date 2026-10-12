# 模型与推理

## 共享表示

主干宽度64，保留ACC、GYRO、方向不变运动、PPG质量门控、12项统计特征和缺失分支，以及3/15/60秒proposal路径。语义表示用于事件存在判断；短感受野定位表示复用分支block输出，经过两层因果深度可分离卷积，供onset、offset和Boundary使用。

时间缓存为201维：语义64、原局部64、9个时间线/有效性标量、定位64。缓存绑定模型、数据、epoch、subject/session和处理视图；历史表示detach，只有当前chunk保留梯度。候选坐标为离散生成，属于截断联合训练。

## 候选判别

ROI为40个259维tokens，区域布局`4/16/4/8/8`：前方、内部、后方、起点局部、终点局部。每个区域独立执行共享时间编码器，再加入区域embedding。token保存mean/max、传感器有效率、位置与已观测覆盖率；无效输入在各层遮罩。

四头query pooling和五区域summary、四组内外差分通过轻量比较通路融合，输出64维candidate embedding、event logit和quality logit。事件接受使用校准事件概率；quality只用于粗区间内几何版本选择。

occupancy负责粗候选；合法onset峰提供学习起点，无峰时兼容回退。自然回放从epoch3开始，在固定session权重、eval/no_grad下生成候选；训练仅释放decision已经到达的候选，正式loss使用该session一致增强。困难负例由事件头BCE排序，随机背景先保留名额。

## 训练贡献

组内版本平均、事件内组平均、session内事件平均；纯负按实际family平均。累计4个session时，正负分别按有贡献session数平均。正组可替换，对应CPU梯度同步替换；不保存跨chunk计算图。受试者贡献权重有界，校准保持自然候选分布。

端点单侧Gaussian热图使用session级峰值/背景计数归一化，固定背景比20。GYRO/PPG丢弃、旋转和因果低带宽增强在raw、统计、不变特征及mask中一致，同session跨chunk和恢复使用相同描述。

## 部署接口

```python
from bme_eating.bundle import StatsFusionJointFinalDataDetector
from bme_eating.data.stats_fusion_preprocess import RawSessionInput

detector = StatsFusionJointFinalDataDetector("path/to/run/bundle", device="cpu")
events = detector.predict_session(raw_session)
```

`raw_session`必须为包含时间戳、传感器值、有效mask及subject/session身份的`RawSessionInput`。数组字段以该dataclass定义为准，不能从label构造推理输入。分批原始输入可用`new_raw_session()`及其`push(chunk, final=...)`接口；构造、追加和结束标志以方法签名为准。

部署只执行锁定操作点，Boundary固定off。每个候选满足`max_input_time ≤ decision_time`；流式NMS不可撤回已提交事件，提交延迟不超过60秒。严格`IoU > 0.25`才是匹配。批量评分和缓存必须保持逐候选decision遮罩、元数据身份及原排序。
