# v4.12 一体化修复与优化

## 身份与证据范围

本目录是当前 v4.11 HEAD 加工作目录修改的独立源码副本，包含此前恢复、进度条和性能修复。`v412_source_snapshot.json` 保存复制时逐文件 hash；`manifests/v412_source_provenance.json` 保存原目录路径、HEAD、staged/unstaged diff 和原源码未变核验。后者明确原目录与隔离目录的对应关系。正式 run 保存两份文件并绑定其 hash。

注册身份为 v4.12 / v412 / `statsfusion_joint_event_v412` / `statsfusion-joint-event-v3`。模型 seed 固定 2026，配置为 `configs/statsfusion_v412_integrated_repair.yaml`。原始输入、canonical features 和 subject split 只读使用；不加载旧权重、optimizer、候选、CDF、校准器或操作点。

默认只接受 `fold0 + single`；完整模式只接受 `fivefold + oof`。执行模式在 run 身份中冻结。fold0 已参与模型开发，因此所有单折产物均声明 `promotable=false`、`subject_excluded_development_only`。单折诊断 bundle 的资格依赖训练、操作点和已启用 Boundary 的资格；不能作为正式五折晋级证据。

## 同一次交付的训练修复

### 统一增强

增强描述由 seed、epoch、subject、session 决定，同一 session 的 chunk 与重叠 warmup 完全一致。GYRO dropout、PPG dropout、合法旋转和低带宽概率均为 0.20；低带宽目标等概率取 25、26.316、50 Hz。

先在物理空间变换，再产生 raw、方向不变分支、12 项因果统计、missing mask、PPG 质量及模态有效率。低带宽使用因果 FIR、session 对齐降采样和样本保持；运动输出为 100 Hz，PPG 保持原 50 Hz 网格；目标率高于 PPG 原生率时不增加观测。运动 jerk 按 100 Hz 网格计算。缺失区间保留缺失。

统计使用原 15 秒因果窗口，normalization/scaler 只拟合对应训练 subject 的自然数据。验证、CDF、校准与测试使用自然视图。每 worker 的派生统计 LRU 上限为 256 MiB，身份含数据、特征代码、增强、normalization 与 scaler。

### 稀疏起止监督

真实端点后的首个 3 秒网格位置为峰值 1，延迟小于 3 秒。单侧 Gaussian sigma 为 9 秒，onset 支持 30 秒、offset 支持 60 秒。ignore、无传感器输入和左截断端点不参与监督。

Focal alpha=2、beta=4，正峰与非峰背景分别按 session 的有效数量归一化，chunk 累加。起止总权重仍为 0.1，不再额外乘 chunk/session 长度比例。onset 触发 0.30，offset 0.50；occupancy 保留事件存在判断，无合法学习起点时保留兼容几何。

验证保存端点命中、最大概率、背景峰/h、粗父候选的起止 signed error 与 MAE。

### 候选贡献

每 chunk 最多预评分 32 个不同负 family，每 family 至多三个几何，额外候选头调用至多 96 个候选。优先为 near-miss 保留八个 family，零 IoU 背景补位；采用来源、时长、wear_hand、hand_relation、GYRO missing 分层的确定性轮换。`unknown` 独立处理。

挖掘暂时关闭 dropout，在 `no_grad` 下按实际 event BCE 排序；训练模式随后恢复，正式 loss 重新带梯度计算。监督配额为 2 positive / 2 near-miss / 3 hard-FP / 1 random background。负 family 在同 session 只参与一次。

每独立正事件最多保留三组，每组最多三个不同几何。优先最高 IoU，再补足三个质量区间与 decision time 差异；已覆盖事件可被更优新组替换。旧 CPU 梯度贡献同步移除，session 完成时按实际事件、组、版本归一化。保存实际 loss proposal、采样原因、替换集合、配额缺口和挖掘耗时。refined 几何重新计算 event、quality 和组内排序目标。

受试者权重为 `clip(c/sqrt(session_count), .5, 2)`，使 session 加权均值为 1，一致作用于时间线、正事件及负样本。自然校准不使用此训练权重。

### 模型与调度

主干保持 64 维、现有传感器分支和 3/15/60 秒 proposal，不增加 attention heads。40 个 ROI tokens 分为 4/16/4/8/8，分别通过共享参数 TCN，再加入区域 embedding 与四头 query pooling。每个区域内、每层前后重施 mask；卷积不跨区域接缝。

保留 256 新步、真实 receptive-field warmup、历史 detach、session 固定权重和累计四个 session 更新。属于截断联合训练，离散候选坐标不具备可微定位梯度。AdamW encoder LR=1e-4、head LR=3e-4，weight decay=.01，BF16，clip norm=5。

前两轮 LR 从 10% 升至 100%，之后沿固定 64 轮轴余弦衰减到 10%。重训按相同调度前缀执行，不能将周期缩短到选定 epoch。最大 64 轮；epoch 8 起每两轮验证，三次没有 .003 以上 F1 改善早停。原 clipping、finite、lineage 和身份资格保持。

## 划分、搜索与 Boundary

单 partition 保留原 subject 划分：13 fit + 7 selector 选 epoch；从头重训 20 人并预测排除的 11 人；锁定内部校准/操作点后从头训练 31 outer-train 人，预测八名 fold0 subject。最终直接保存该 31 人 checkpoint，不用 outer 标签重训或调参。

完整五折使用每折三个内部校准分区及对应从头重训，外层只应用锁定协议。搜索空间仍为 9 decoder × 3 budget × 3 NMS × 99 acceptance = 8019 个 coarse 操作点。接受分数登记为 `calibrated_event_probability_v412`；quality 用于同父粗区间的几何选择。每阈值独立流式 NMS，提交不可撤销。

Boundary 从第三轮学习；正式修正需要至少 60 个独立配对正事件、覆盖 ≥.70 和既有 MAE/F1/TP→FP/clipping 门禁。修正搜索前检查最大可用独立事件数；不足则锁定 `off`、记录原因并跳过六组无资格修正，不停止局部端点及 Boundary 头学习。当前单内部留出只有 44 个 truth 时必然关闭。

参考可审计 v410、v411、v412；必须核对 resolved config、完成文件、模型排除、校准、cohort、truth/ignore 和曝光。同名身份冲突报错；参考未完成或不合格记录比较未执行，不妨碍新模型诊断。S0 显式跳过，不能宣称优于 sensor-only。

## 速度、恢复与资源

CDF 只生成父 proposal，跳过候选评分与 Boundary。相同实际模型权重和自然预处理可复用 137 维时间特征；完整缓存检查身份、文件 hash、行数和时间遍历，读取时只暴露当前 causal prefix。未完整提交的预测缓存重建。

验证先求每 decoder 三种预算可能进入的候选并集，再以 batch128 评分。精确 ROI、校准概率、decision 分组和 IoU 搜索缓存保留。最多四个 CPU 搜索进程，按注册顺序合并；不减少搜索范围。outer 和 bundle 只执行锁定操作点。

默认 worker10、prefetch4、reader cache4。训练与验证 loader 交替释放。记录编码/CDF/ROI/评分/搜索耗时、吞吐、head 次数、cache 命中、写盘量和进程树资源；GPU 峰值限 10.5 GB、private RAM 限 26 GB，超限失败。

每 chunk 持久化缓存和校验追加日志，再原子保存 checkpoint。保存 RNG、LR/optimizer、累计梯度、可替换正组 CPU 梯度、增强、流式候选状态、cache 与日志游标；恢复截断未提交尾部。Ctrl+C 为 PAUSED，OOM/异常为 FAILED，未完成 epoch 不标完成。resume 仅允许 worker/prefetch/cache 参数变化，模型、loss、采样、数据、源码、seed、协议或执行模式变化需要新 run。

PowerShell TTY 使用白色单行条，训练、CDF、验证推理、select 搜索均显示进度。使用 `conda run --no-capture-output`；验证后打印 F1、TP/FP/FN、FP/h、覆盖、边界、操作点、clipping、best epoch 和早停。

## 运行命令

默认单折诊断：

```powershell
Set-Location -LiteralPath "C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v412"

conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v412_integrated_repair.yaml `
  --run-name statsfusion_v412_integrated_repair_20261010b `
  --stage all --evaluation-mode fold0 --internal-mode single --skip-s0 `
  --reference-run statsfusion_v411_fold0_continue64_20261009a --fresh
```

中断恢复使用同一命令，将 `--fresh` 改为 `--resume`。新 run 不能加载旧 v411 checkpoint。

Windows 验证缓存路径可能超过传统的 260 字符限制，表现为 `FileNotFoundError: [WinError 3]`，即使父目录存在且系统已启用长路径。项目 `.env` 的 `BME_OUTPUT_ROOT` 使用 `\\?\` 长路径前缀。当前终端已有同名环境变量时，它优先于 `.env`；先执行以下命令，再运行原来的 `--resume` 命令：

```powershell
. .\scripts\enable_statsfusion_long_paths.ps1
```

该辅助脚本只规范当前进程的输出路径，支持本地盘和 UNC 路径，不修改模型、配置 hash、训练源码或恢复 checkpoint。它指向原输出目录，第 8 轮训练完成后在来源 CDF/验证缓存初始化处失败的 run 可以保留训练进度，恢复后继续该轮验证。无需新建 run，也不需要绕过源码身份校验。

完整模式必须使用新的 run name：

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v412_integrated_repair.yaml `
  --run-name statsfusion_v412_integrated_repair_fivefold_20261010b `
  --stage all --evaluation-mode fivefold --internal-mode oof --skip-s0 `
  --reference-run statsfusion_v411_fold0_continue64_20261009a --fresh
```

单折参考不能通过完整五折的配对参考门禁；可替换为已完成且核验合格的完整参考 run。

## 验收和产物

`tests/test_statsfusion_v412.py` 覆盖统一增强、因果带宽、端点梯度、正组替换、事件头挖掘、区域 mask、严格 IoU、缓存、Boundary 资格和新协议。恢复/Windows 多 worker/bundle/no-XGBoost 测试复用已有测试并扩展 v412。

```powershell
python -m pytest -q
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
git diff --check
python scripts/validate_statsfusion_joint.py `
  --config configs/statsfusion_v412_integrated_repair.yaml --device cuda `
  --output manifests/v412_gpu_validation.json
python scripts/validate_statsfusion_v412_real_inputs.py --device cuda
```

GPU smoke 使用随机初始化新模型，包含 forward/backward、batch1/128、13022 秒长事件、原生/增量/前缀 replay 和资源检查；不代表实际训练 F1。真实输入 smoke 只读审核 canonical，再对两个训练域 session 做有限 forward/backward；其自然预处理仅属于 smoke，不能导入正式训练。

性能比较入口 `scripts/benchmark_statsfusion_v412.py` 对同一合格 v412 checkpoint 和排除 subject 比较完整 coarse 搜索与参考执行路径，检查操作点、匹配和门禁一致。需要新训练的合格 checkpoint；交付时没有该 checkpoint，尚不能宣称完整验证 ≥2 倍或真实 F1/MAE 改善。

单折完成保存 `fold_0/model/latest.pt`、内部预测/搜索、`fold0_gate_manifest.json`、`fold0_final/final_manifest.json` 和合格时的 `fold0_bundle/SHA256SUMS.json`，全部 `promotable=false`。训练或操作点失败保留独立失败报告和诊断权重；不会生成可误认为正式部署的 bundle。

完整五折只有所有正式门禁和合格参考通过后才保存正式 `joint_gate_manifest.json`、`final_manifest.json`、`SHA256SUMS.json` 与部署 bundle。严格区分实现验收、五折诊断和正式晋级，不承诺性能增幅。

### 本次实施验收（2026-10-10）

- 完整 CPU 回归执行 712 项；唯一失败是旧配置白名单遗漏 single-partition 和 v412 配置，已修复。随后对相关模块及新增资格检查执行 111 项复测，零失败。合并去重结果为 708 通过、14 跳过、无未修复失败，详见 `manifests/v412_implementation_audit.json`。
- CUDA 中断、OOM、累计梯度和正组 CPU 缓冲恢复测试 5 项通过；ruff、compileall、diff 检查通过。
- 最终源码 GPU smoke 通过：重复误差 0、batch1/128 最大误差 1.49e-8、长事件与原生增量 replay 一致，峰值显存约 1.20 GiB。
- 最终源码真实输入 smoke 通过：完成 canonical 审核与默认 10-worker 的两个增强 chunk forward/backward，峰值显存约 1.19 GiB，进程树 private RAM 约 11.44 GiB。这是有限 smoke，不能替代完整训练的最坏资源检查。
- 再次核验原 v411 快照中的 481 个源码、配置、测试及文档文件，全部未变。
- 没有启动正式 v412 训练，没有实际 F1/MAE 提升或完整验证两倍加速的实测结论；五折诊断与正式晋级均未完成。

### 流水线审查后的修复（2026-10-10）

本次修复五处已复现的问题：

- 来源 CDF 按 model、subject、session、ancestor、父来源和尺度去重。同一 family 中不同父 proposal 均进入经验分布；完整候选与 source-only 路径不再依赖版本标记来决定去重。父候选的几何版本仍只贡献一次。
- Boundary 的 TP→FP 按原 coarse 匹配 prediction ID，追踪其是否成为 refined 输出中真正的 FP。其他候选补回同一 truth 不会掩盖原候选的转换。被阈值拒绝、未输出及 ignore 的候选分别统计；truth 保留率和成对 MAE 继续独立记录。原有正式门禁不放宽。
- coarse 多进程搜索、参考执行路径及 Boundary 搜索均检查主进程及所有子进程的 private RAM，矩阵分配前检查预计内存，超过注册的 26 GB 限制报错。多进程按注册顺序归并结果，保留原有完整搜索和 tie-break。
- 训练时间特征缓存每个 chunk 记录滚动 SHA256；checkpoint 绑定已提交行数和 hash。恢复先核验已提交内容，再截断未提交尾部，拒绝同长度损坏、缺失校验证据及落在 chunk 中间的游标。正常运行同一 session 复用缓存对象，避免每 chunk 重扫全部历史。累计梯度和 RNG 的恢复语义保持。
- 预测特征缓存身份纳入整个 `src/bme_eating` 的逐文件 hash，覆盖 sequence、baseline 派生统计等传递依赖；完整 manifest 同时绑定缓存 hash 链及文件校验值。

新增回归文件为 `tests/test_statsfusion_joint_pipeline_repairs.py`，并扩展已有恢复测试，验证 Ctrl+C/OOM 后参数与连续执行逐位一致。

CDF 修复改变候选排序，源码与缓存身份也发生变化，因此使用新的 `20261010b` run 从头训练。已有 run 不得用普通 `--resume` 混接此次修复；新 run 中断后继续按同名 `--resume` 恢复。模型容量、损失权重、数据划分及 8019 个 coarse 操作点不变。此前审查报告及初次实施验收文件保留为历史证据，当前验收另存 `manifests/v412_pipeline_repairs_audit.json`。
