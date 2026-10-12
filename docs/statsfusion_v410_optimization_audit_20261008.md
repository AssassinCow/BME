# v4.10 当前训练暴露问题与优化优先级审查

审查时间：2026-10-08，北京时间约18:11。对象：`statsfusion_v410_joint_event_20261007a`，协议 `statsfusion-joint-event-v1`，seed `[2026]`。本次分析读取当前源码、配置和已有训练产物，未执行额外模型训练或GPU性能测试。

**结论：共享编码器路线已有积极的内部验证证据，但当前瓶颈集中在起点定位、候选级监督的有效利用、选模解释性，以及重复候选评分和搜索的实现开销。现有证据不足以认定需要增加hidden width、attention heads或新增大型Transformer。**

## 1. 证据范围与当前状态

当前已完成fold0、内部partition0的第8、10、12、14轮selector评估。selector有34个有效事件，曝光266.1008小时；fit有51个事件、346个session，每轮2021个chunk、87次optimizer更新。正式outer五折结果尚未完成。

| 指标 | epoch8 | epoch10 | epoch12 | epoch14 |
| --- | ---: | ---: | ---: | ---: |
| 搜索后selector F1 | 0.5333 | 0.5231 | 0.6230 | 0.5902 |
| TP/FP/FN | 16/10/18 | 17/14/17 | 19/8/15 | 18/9/16 |
| 所选预算后的候选覆盖 | 33/34 | 32/34 | 32/34 | 28/34 |
| FP/h | 0.03758 | 0.05261 | 0.03006 | 0.03382 |
| 起点MAE，秒 | 320.9 | 275.4 | 534.2 | 469.9 |
| 起点signed error，秒 | -91.9 | -110.4 | -523.2 | -415.9 |
| 终点MAE，秒 | 167.5 | 148.4 | 79.6 | 147.4 |
| 同手召回 | 0.5385 | 0.5385 | 0.6154 | 0.4615 |
| 异手召回 | 0.4286 | 0.4762 | 0.5238 | 0.5714 |
| clipping fraction | 0.0805 | 0.1379 | 0.1149 | 0.1149 |

当前`selected.pt`保留epoch12。训练资格正常，单次epoch14回落不能直接诊断过拟合。各轮的操作点、校准器及匹配事件不同，MAE也不是同一TP集合的配对比较。selector同时参与选epoch、拟合校准及操作点搜索，不能把这些成绩作为独立正式结果。

历史参考审计当前为`qualified=false`、`comparison_performed=false`；理由包含single-holdout协议缺乏合格完整outer OOF证据。S0仍显式跳过。即使后续绝对指标达到目标，也不能据此宣称优于合格历史参考或sensor-only。

## 2. 模型架构：起点定位应优先修复

**2.1 起点头没有直接进入当前候选解码。**

联合模型输出并监督局部onset/offset。`pack_features()`保存二者，但`generate_chunk()`仅把state和3/15/60秒proposal概率以及offset送入`StreamingProposals.observe()`；onset概率没有直接用于起点生成、校正或候选判别输入。onset损失仍能更新共享局部表示，因此不能说该头完全无作用；问题是其显式起点证据尚未被解码器利用。

建议让起点候选由已观测的局部onset峰及占用概率变化共同提供，起点版本继续保留来源和父边界。起点选择与长上下文事件语义分工：长上下文负责识别事件，局部证据负责定位。任何参数均在对应训练/内部OOF中锁定，并保持因果和前缀一致性。

**2.2 固定600秒扩展、预算配额及粗边界可能共同放大提前起点。**

`StreamingProposals._ready()`的backtrack与start-expansion都生成`start-600秒、end不变`的候选。同一父候选下，二者通常几何相同、来源元数据不同。预算在事件评分之前优先满足扩展来源欠额，最终提交以事件概率×质量排序。

这一机制与epoch12/14的强烈负偏一致，但当前selector缺少所选预测逐事件来源表，尚不能证明具体多少TP来自扩展来源。必须先记录来源与残差，再做候选来源消融。

建议采用onset证据约束的多个起点版本，并在family内部学习边界质量差异；几何相同的版本可共享ROI计算，但来源元数据不同的事件头输出不能直接复用。不要把所有预测固定后移600秒，也不要未经消融删除扩展来源，因为它仍可能挽救漏覆盖。

**2.3 当前Boundary训练资格在这个fit分区内无法满足。**

Boundary要求至少60个独立匹配正事件，而当前fit总共只有51个truth。实际epoch12的边界资格只有41个独立匹配事件，`qualified=false`；epoch8/10/12/14的共享编码器Boundary加权梯度诊断全部为零。

这意味着当前selector模型的候选级Boundary分布头没有获得该训练损失，而局部onset/offset仍在训练。之后内部retrain训练集扩大，Boundary可能获得资格，但不会自动修复当前selector模型，也不保证选定训练时长足以训练新启用的头。

建议下一协议区分“允许用合法训练标签学习边界”与“允许正式启用边界修正”。训练采用可获得的独立正事件，控制每事件贡献和正则化；启用决策继续依赖足够事件的跨主体OOF及原有硬门禁。保留coarse fallback。这个调整改变训练协议，必须新run。

**2.4 长事件的16个内部ROI bins会压缩定位信息。**

内部16 bins用于完整事件语义合理，但13022秒事件每bin跨度约814秒，不能依赖它提供精细起点定位。前后4 bins仅围绕候选粗起止点；粗起点过早时，真实onset可能处在很宽的内部bin里。

建议保留全事件16 bins，同时给边界判别增加固定时间尺度的局部证据、onset/offset置信度及内外占用差异。优先优化输入信息和监督，不增加主干宽度。

## 3. 训练策略：独立事件计数不能等同于只训练一个版本

**3.1 设定的采样比例没有成为实际采样分布。**

每chunk最多8个，最大余数法配额为positive2、near-miss2、hard-FP3、random-background1。已有审计如下，数字是trainer跨chunk/epoch去重之前的候选数：

| epoch | positive | near-miss | hard-FP | random-background | 合计 | 平均每chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 61 | 39 | 1284 | 1879 | 3263 | 1.61 |
| 12 | 49 | 26 | 730 | 1882 | 2687 | 1.33 |
| 14 | 52 | 24 | 826 | 1881 | 2783 | 1.38 |

自然背景多和正事件稀少本身是数据事实。待优化处是有效正候选、near-miss及定位难例数量很少，不能把配置中的30%正样本当作实际已实现的训练比例。不能凭复制同family来补齐独立数量。

建议在保持session顺序和合法因果时间的基础上，增加事件/边界贡献的均衡统计，允许不同质量的真实候选版本参与组内监督，记录缺额与实际进入loss的数量。需要候选池时只缓存已完成事件的合法候选，并保留模型更新、detach及过期身份。

**3.2 当前跨epoch遍历内的去重限制了同事件和同family质量学习。**

`sample_candidates()`在chunk内去重family和正事件；trainer又用`event_contributions`、`family_contributions`使同事件正样本或同family负样本在整个epoch中只贡献一次。这能控制重复权重，但一个较早抽到的粗候选可能消耗该事件本轮全部正样本资格，后续边界更准确的版本被跳过。

质量头需要学习同一事件的好边界与差边界。建议保留独立事件/family计数，允许少量有差异的版本，以事件/family总权重固定的方式归一化；候选可以多，独立证据数量不膨胀。可增加组内质量排序监督，并保留严格IoU>0.25的事件目标与连续IoU质量目标分离。

**3.3 联合训练对长事件早期编码特征的梯度仍受截断限制。**

每次256个新步，历史feature detach。完整候选常在结束后才产生事件损失，因此长事件的早期表示作为历史输入参与候选判别，却不能接收该候选的编码器梯度。这是当前截断反向传播的实际边界，并非完整端到端定位。

建议增加已经发生部分的事件语义/阶段辅助监督或局部候选辅助判别，让事件中段也获得与最终判别有关的梯度。部分事件不能伪造尚未发生的终点，更不能把前缀标签直接当完整事件IoU标签。对可观测困难片段可在session完成后重新编码少量局部区域，但需重新定义训练成本与身份。

**3.4 共享梯度不均衡已有证据，但不能立即用扩大损失或PCGrad代替诊断。**

epoch14每32个chunk采样的64条诊断中，proposal/state/event/quality加权梯度范数中位数约为0.00915/0.00400/0.0000193/0.000000138。event与proposal范数均>1e-6的55条记录中，28条夹角>90度。

诊断多数落在背景chunk，无法代表正事件上的任务重要性；质量头自身参数梯度也未由这一编码器范数充分描述。应先按正事件、near-miss、背景分层统计梯度、损失、任务头梯度和实际贡献，再决定温和的权重调整。局部起止正监督很稀疏，可评估训练折内正权重或focal监督。PCGrad只作为冲突持续且影响效果时的对照。

对比学习实际并非完全未启用：epoch8/10/12/14均有63/2021个chunk产生有效对比项，每轮记录4262个有效anchor。其覆盖范围很有限；可增加同事件已观测阶段的合法配对，保持跨subject/session/model缓存隔离，并保留关闭对照。

## 4. 验证与选模：分清模型变化和操作点变化

epoch14所选`high_0.2_gap_30、budget10/h`覆盖28/34。但同轮`high_0.06_gap_30、budget20/h`可覆盖33/34，TP20、FP16、F1=0.5714、FP/h=0.06013。模型仍能提出很多真实事件，问题包含预算和最终误报约束下的筛选取舍，而不是候选能力全局崩溃。

建议保留既定完整内部搜索，同时增加固定decoder/预算的诊断曲线，并分别记录：原始proposal覆盖→预算后覆盖→事件判别接受→流式NMS提交→严格匹配TP。固定诊断点必须提前锁定，不能用outer结果倒选。

selector仅34个事件，同手仅13个。epoch12到14同手召回下降对应8个TP降为6个，尚不能认定具体手别编码分支失效。应输出按主体、手别、时长和缺失分层的事件级错误，使用同一事件集合进行配对边界比较，并依靠内部OOF及outer评估验证泛化。

8019个操作点在同一selector上搜索，再选epoch，开发成绩有选择偏差。可在下一协议中评估校准与selector选择进一步cross-fit，或选择邻近阈值稳定的区域；当前run继续以完整内部OOF和outer holdout为正式证据。

当前selector目录主要保存`search.json`和时间特征cache，缺少所选候选/拒绝候选/匹配事件的完整表。训练sample审计也在epoch去重之前计数。需增加实际loss贡献、来源、quality、边界残差和拒绝原因，使“覆盖高但TP少”的损失位置可复核。

## 5. 训练、验证与部署速度

**5.1 来源CDF拟合执行了大量不需要的候选头推理。优先级最高。**

`fit_source_mapping()`调用完整`predict_joint_model()`，对9套decoder生成的全部候选逐个评分，并生成3套Boundary回退结果。CDF只需family/source/scale/raw_score及训练subject身份，不需要event logit、quality或refinement。

应实现专用proposal/source映射路径，保留相同encoder、decoder顺序、family代表选择、CDF结果和数据审计，跳过候选判别与refinement。模型改变后来源分布也会改变，因此不能把epoch8的CDF直接复用到epoch14。

**5.2 逐候选ROI与GPU小调用使现有GPU资源难以利用。**

`score_candidate()`每个候选先扫描历史时间，再逐个构造24个bin的mean/max，执行batch1候选头，随后把两项输出转float产生同步。验证在预算前评分所有候选，9套decoder与大量变体放大成本。Boundary未qualified时仍生成多份回退对象。

建议时间轴二分定位所需区间、缓存完全相同的ROI、批量构造候选输入，并用32/64/128等验证batch做硬件测量。缓存键必须覆盖精确边界、decision time、模型/数据/feature/missing mask身份；输出缓存还必须覆盖metadata。不能仅以family作为复用键。

各预算的联合搜索可先确定其候选并集，只评分会进入至少一个预算的候选；CDF仍使用原来的完整生成分布。正式部署已经先预算后评分，不执行完整8019点搜索。

**5.3 搜索重复做校准、DataFrame构建及匹配。**

每次完整搜索为3 high×3 gap×3 budget×3 NMS×99 acceptance=8019。每个acceptance重新校准同一批候选概率、分组、提交、构造含大量诊断字段的事件DataFrame并计算匹配。

应在每组setting/budget内缓存校准概率、rank score、decision分组、ignore判定、session曝光及候选-truth IoU；匹配时使用保留原规则的最小事件数据结构。并行时恢复原搜索顺序与tie-break，控制进程复制内存。

流式提交是不可撤销的，改变threshold可能改变早期事件对后续事件的抑制关系。不能把低阈值下的最终预测简单过滤来替代其他阈值的流式NMS。不能删搜索范围来声称数值等价加速。

**5.4 断点诊断历史不断增长，完整写盘代价随epoch上升。**

每chunk写时间cache、feature cache及完整resume.pt，并分别flush/fsync；cache binding也重复原子写入。epoch14之后的resume约28MB，每轮2021个chunk，写入量可能达到数十GB量级。这个量级估算不等于实际磁盘吞吐测量。

应把不可变sample/gradient/历史诊断改为带校验的追加日志，checkpoint保留提交位置和恢复所需最小状态。仍保存RNG、模型、optimizer、累计梯度、session状态和cache游标，恢复只接受完整提交chunk。需要维护先日志/缓存持久化、后checkpoint提交的原子协议，不能随意减少安全断点。

**5.5 编码器每256个新步实际计算1137步。**

当前真实geometry为history881、新步256，总1137，即输入步数为新步数的4.44倍。历史来自实际感受野要求，直接缩短会改变模型。

验证可研究增量卷积状态及跨尺度完成块缓存；训练必须同时处理dropout、梯度范围和session内模型身份。改变chunk长度或多session训练batch会改变浮点顺序、dropout RNG和事件梯度截断，不能直接宣称当前run数值等价。

**5.6 实际验证约1小时至近2小时，当前缺乏可靠分段profiling。**

按来源cache首次创建至search保存的文件时间估算：epoch8约110.7分钟，epoch10约56.9分钟，epoch12约101.4分钟，epoch14约99.4分钟。这些包含来源拟合、selector推理和搜索，不是纯GPU验证耗时。epoch14来源拟合段约54.1分钟。

`training_history.elapsed_seconds`从一次训练函数调用起算，且resume重新起算，不是每轮耗时。应新增read/preprocess/H2D/encoder/ROI/head/CDF/search/cache/checkpoint阶段计时、候选吞吐、磁盘量及完整主/worker进程RAM。

训练persistent worker在验证期间仍可能保留，而验证新建另一套10 worker。应共享安全的数据服务或明确释放/复用，按总RAM调整预取；增加worker不等于增加CPU搜索速度。

**5.7 部署速度还需真实长session验证。**

部署只运行锁定的一套decoder、预算、校准及Boundary状态。但`JointRawStream`不断拼接raw数据，`push_features()`不断拼接历史，ROI继续扫描历史，长session存在随长度增长的额外复制和查询成本。

建议采用有身份的增量原始预处理、时间索引和分块历史存储。活跃长事件必须保留其候选所需的完整信息，不能简单丢弃固定长度窗口。前缀一致性、决策时间和wall-clock实时倍率分别验收；数据时间轴延迟≤60秒不能证明墙钟推理达标。

## 6. 推荐实施优先级与当前run兼容性

| 优先级 | 工作 | 主要目标 | 当前run处理 |
| --- | --- | --- | --- |
| P0 | 专用CDF路径、无资格Boundary跳过、ROI定位和候选批评分 | 减少重复验证推理 | 数值及搜索结果验证通过后受控resume迁移 |
| P0 | 缓存搜索输入、最小匹配数据、稳定顺序并行 | 缩短8019点搜索 | 同上，保持全部搜索范围与流式匹配规则 |
| P0 | 逐事件诊断、实际loss贡献、阶段计时和完整RAM | 定位遗漏与可靠估时 | 受控身份变更，避免运行时直接替换源码 |
| P1 | 诊断追加日志与小checkpoint事务 | 控制每chunk写盘量 | 恢复等价及崩溃测试通过后受控迁移 |
| P1 | onset参与解码、边界训练/启用门禁分离、family组内质量监督 | 改善提前起点与判别质量 | 新配置、新协议身份、新run |
| P1 | 事件/主体贡献归一化、局部事件辅助监督、梯度分层 | 提升共享训练有效性 | 新run，保留当前run作为对照 |
| P2 | 增量encoder和raw处理、多session吞吐实验 | 长session和训练加速 | 先验证数值/前缀/训练语义；无法等价时新run |
| P2 | PCGrad、额外容量或更复杂主干 | 仅用于已证实的剩余瓶颈 | 消融后决定，不作为首选 |

性能优化验收应同时核对：source CDF、所有操作点记录及最佳点、逐事件身份/边界、严格匹配、门禁、前缀一致性、同checkpoint重复误差≤1e-6，以及resume后的RNG/optimizer/累计梯度。涉及训练路径时，还需验证loss、参数梯度及后续optimizer更新。batch数值微差若改变阈值/NMS结果，不属于已证明的无影响加速。

算法修改验收应使用同一subject split、seed2026、合法内部OOF和独立outer评估；增加配对起止MAE、FP来源、正事件分层和候选漏斗。不能仅凭新的搜索后selector F1判定改动有效。

本报告不承诺F1、MAE或端到端速度提升比例。最值得优先实施的是无需改变科学协议的重复计算消除；模型效果优化则先解决已确认的监督与解码脱节，再考虑增加容量。

## 7. 可复核来源

- 训练结果与资格：`outputs/v410/runs/statsfusion_v410_joint_event_20261007a/fold_0/internal/partition_0/selection/training_history.json`、`selected.pt`、`sampling.json`、`gradients.json`、`selector/{8,10,12,14}/search.json`。
- 参考资格：同run的`reference_audit.json`；输入身份：`input_audit.json`；参数：`resolved_config.json`及`configs/statsfusion_v410_joint_event.yaml`。
- 模型：`src/bme_eating/models/statsfusion_joint_event.py:66`、`:102`、`:129`、`:154`。
- 解码与采样：`src/bme_eating/joint_stream.py:151`、`:280`、`:379`、`:435`。
- 训练、事件贡献与Boundary资格：`src/bme_eating/training/statsfusion_joint_trainer.py:393`、`:593`、`:622`、`:726`。
- 推理与来源拟合：`src/bme_eating/training/statsfusion_joint_trainer.py:814`、`:867`。
- 操作点搜索：`src/bme_eating/joint_evaluation.py:157`、`:198`。
- 部署：`src/bme_eating/joint_pipeline.py:61`、`:110`、`:150`。
- 感受野与worker：`src/bme_eating/data/joint_sessions.py:98`、`:205`、`src/bme_eating/data/stats_fusion_sequence.py:31`。

上述源码相对路径均以`C:/Users/LZX/Desktop/University/Competition/生医工/project/modeling_joint_v410`为根；结果相对路径以`C:/Users/LZX/Desktop/University/Competition/生医工`为根。原始数据及既有运行产物保持其各自审计身份。
