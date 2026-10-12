# v4.11 全局深度审计与修复报告

日期：2026-10-08。对象为本隔离目录中的现有工作树，包括未提交的恢复和性能修复。

本轮发现并修复13类实现或验证问题。当前实现通过本报告所列回归、GPU和资源检查；这不证明五折F1改善，也不等于正式晋级。未启动正式五折训练，未修改`modeling_joint_v410`目录及历史运行产物。

## 范围与身份

检查覆盖共享编码器、多尺度proposal、onset峰、ROI、事件与质量头、Boundary训练与启用、监督采样、CPU梯度缓冲、session/cache、checkpoint、resume、来源CDF、搜索与匹配、内部OOF、outer、参考、gate、final、export、bundle replay和资源控制。

| 项目 | 本轮核验值 |
|---|---|
| 模型 / 协议 | `statsfusion_joint_event_v411` / `statsfusion-joint-event-v2` |
| seed / hidden | 2026 / 64 |
| Git HEAD | `e0ec880f25412d0683261068c3dc7955b0f6d9d3`，不是完整工作树身份 |
| 当前源码聚合SHA256 | `fe488975944a18d40b41f04a555253d1fc43d63a2b2f9b1419f01f1bebcdbbf4` |
| 模型与协议配置SHA256 | `1d22250132390a806c0872535a8a2e8a505bbd0e9a14c6f83254abb29fbef286` |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU，12 GB |
| Python / PyTorch / CUDA | 3.11.16 / 2.5.1 / 12.4，Conda `bme-model` |

源码、配置和运行时身份由当前GPU验证记录核验；逐文件源码hash、输入hash和测试证据保存在`manifests/v411_deep_audit.json`。原`v411_source_snapshot.json`作为父快照证据保留。

## 已修复问题

| 编号 | 缺陷及影响 | 修复与回归证据 |
|---|---|---|
| 1 | 独立验证脚本默认v410配置，可能把旧模型测试当成v411证据 | 默认改为v411，GPU报告明确记录v2协议及source/config hash |
| 2 | Conda可编辑安装指向旧项目，pytest可能导入旧源码 | `pyproject.toml`明确设置`pythonpath=["src"]`，本目录测试无需重新安装旧环境 |
| 3 | 共享候选默认评分公式误设为v411，改变旧版概率×质量评分语义 | 恢复legacy默认；v411生成和评分路径显式设置事件概率公式，回归同时覆盖两版 |
| 4 | onset峰额外减3秒，和anchor标签时间语义不一致 | 使用峰所在anchor时间，测试27秒峰输出27秒起点 |
| 5 | EOF普通候选和finish候选分两次预算，离线和增量路径可能选不同事件 | 同一决策合并EOF池；暂存非final最后anchor决策，空final也保持相同预算与NMS顺序 |
| 6 | 真实终点未出现时同时关闭起点监督；Gaussian监督可能落在实际支持之外 | 起止分别检查已观测时间及有效支持范围，合法起点独立参与学习 |
| 7 | 邻居冲突恢复coarse后，诊断接受/质量分数仍可能保留refined值 | 回退同步刷新`acceptance_score`和`quality_score` |
| 8 | 提交时只浅拷贝嵌套metadata，输入或返回值修改可能改写已提交事件 | 输入、诊断、内部提交和返回事件使用独立深拷贝，测试修改source flags及坐标 |
| 9 | 同父proposal的版本匹配不同真实事件，却全部计入代表事件，扭曲独立事件数与归一化 | 按真实正事件拆分监督组，按subject/session/ancestor分组，独立事件分别应用监督上限 |
| 10 | YAML采样比例与实际硬编码2/2/3/1不一致，改配置可能不生效 | 用`group_quotas`显式注册并读取；拒绝旧比例键及未注册组数/版本数 |
| 11 | 参考审计核验了声明的lineage，但未逐一核验内部checkpoint实际source/config/data/epoch | 对内部retrain与selector权重核验实际身份、epoch、selector资格标志，增加有效hash下仍拒绝错误身份的8个用例 |
| 12 | export resume仅凭相同model identity可能接受旧bundle权重或旧部署锁 | 比较当前final manifest部署锁及checkpoint实际权重；即便更新合法文件hash仍拒绝旧bundle |
| 13 | opt-in资源测试只加载v410，无法验证v411新增训练路径 | 对v410/v411分别运行真实10-worker游标恢复、完整感受野GPU smoke和原生archive训练/checkpoint测试 |

本轮没有扩大模型容量，没有放宽训练clipping或正式晋级门禁，也没有复用旧权重或校准器。

## 验证结果

| 检查 | 结果与证据 |
|---|---|
| 全套CPU pytest | 654项：634通过、20因硬件/opt-in条件等跳过，0失败；`v411_deep_audit_pytest.xml`，406.551秒 |
| 补跑CPU阶段跳过的GPU用例 | 16通过，3项opt-in资源测试与1项旧canonical诊断测试仍跳过；`v411_deep_audit_gpu_pytest.xml` |
| 显式启用资源测试并扩展v411配置 | 6项全部通过；`v411_deep_audit_resources.xml` |
| CUDA中断恢复 | 注入Ctrl+C和OOM，恢复累计梯度、正事件CPU缓冲及日志尾部，连续/恢复训练权重逐项完全一致；`v411_deep_audit_cuda_resume.xml` |
| ruff / compileall / diff | `ruff check src tests scripts --no-cache`、`compileall -q src scripts`、标准`git diff --check`全部通过 |
| 当前源码GPU smoke | 真实forward/backward、bfloat16、原生采样接口及长事件特征replay通过；`v411_deep_audit_gpu_validation.json` |
| 无XGBoost replay | 通过bundle fixture及实际export/replay回归，子进程禁止导入XGBoost |
| 当前共享编码器梯度 | event/quality各自非零；Boundary独立梯度范数0.4370738268，确认三类损失均可更新共享表示 |
| GPU batch1 / batch128 | 128个不同几何和metadata候选最大误差`1.210719347e-8`，低于`1e-6`；`v411_deep_audit_gpu_heads.json` |
| 重复推理 / 原生输入分块 | 重复误差0；原生raw特征分块误差`2.384185791e-7`，事件身份和分数一致 |
| 长事件分块 | 合成长事件13023秒，在17、256及整段chunk下产生相同事件身份；另有历史detach与缓存回归 |
| 合成GPU smoke资源 | 峰值allocated约1.199 GB、reserved约1.391 GB、进程树private RAM约3.954 GB；本项没有10个worker |
| 实际canonical输入只读审计 | 39名受试者、1153个session、2125个segment文件；输入及subject split hash核验通过；`v411_deep_audit_inputs.json` |

按本轮测试集合合并，已执行656个不同的有效测试用例并通过，剩余1项旧canonical mask-count诊断用例因既有产物早于该诊断字段而跳过。资源测试中的“native archive”使用临时生成的原生采样fixture，不能称为正式数据集训练。

首轮全套测试期间继续修改源码，训练恢复测试正确触发了`Runtime source changed before checkpoint`保护。固定源码后完整重跑通过；本报告引用固定源码后的结果。

## 门禁与剩余限制

1. 当前参考`statsfusion_v410_joint_event_20261007a`缺少`fold_0/completion.json`，审计为`qualified=false`、`comparison_performed=false`。可以继续新模型诊断；当前参考不能支持正式配对晋级和bundle导出。见`v411_deep_audit_reference.json`。
2. 未发现本轮检查范围内outer标签用于模型训练、来源CDF、内部校准或选模的证据；有关排除、OOF、纯FP受试者bootstrap、缺失门禁与导出阻断有回归保护。这是当前审查结论，不能保证不存在其他尚未触发的问题。
3. 没有已训练v411 checkpoint和完整五折产物。因此未证明F1/MAE提升、真实完整验证至少2倍加速或正式训练全程资源峰值。上述GPU数值属于合成测试；不能作为整个真实数据集在10 workers下的峰值保证。
4. 当前端点坐标和候选选择仍离散，历史特征detach；长事件GPU计算图不会包含几小时前的全部历史。通过梯度测试不等于完全可微端到端定位。
5. CPU/GPU fixture bundle replay已验证导出机制；正式训练后的部署bundle、真实长短session资源及墙钟耗时仍须在相应产物存在后验收。
6. 本轮模型和协议配置、源码身份已变化。若已有旧身份v411 run，须新建run；不能强行resume。检查时尚无`outputs/v411/runs`目录，注册的`20261008a`名称仍可用于首次fresh运行。

## 复核命令

在本隔离目录中：

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
conda run --no-capture-output -n bme-model python -m pytest -q
Remove-Item Env:CUDA_VISIBLE_DEVICES
$env:BME_V410_RESOURCE_TESTS = "1"
conda run --no-capture-output -n bme-model python -m pytest -q tests/test_statsfusion_joint_resources.py
conda run --no-capture-output -n bme-model python scripts/validate_statsfusion_joint.py `
  --config configs/statsfusion_v411_learned_boundary.yaml --device cuda
conda run -n bme-model python -m ruff check src tests scripts --no-cache
conda run -n bme-model python -m compileall -q src scripts
git diff --check
```

正式运行命令仍按主文档使用全新run和`--fresh`。本轮验收状态为“实现与回归验证完成”；“完整五折诊断”和“正式晋级”均未执行。
