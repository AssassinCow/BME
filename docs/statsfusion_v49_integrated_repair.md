# v4.9 一体化修复训练协议

## 当前状态与范围

本次正式实现对应配置 `configs/hierarchical_v4_v49_integrated_optimized.yaml`，协议为 `integrated_repair_v2`。代码实现和训练结果分开验收；尚无合格的 v4.9 五折结果或部署 bundle，不能将目标 F1 写成实测结果。

本地已核实冻结 Deep 参考 `hierarchical_v4_r32_deep_frontier_20260930a`，其可引用 F1 为 0.5375。用户已确认不存在合格的 sensor-only（S0）五折基线，当前采用显式 `--skip-s0` 直接重训。S0 对照记录为 `not_available`、`comparison_performed=false`，不视为通过，也不能据此宣称统计分支的独立增益。所有训练资格、候选、冻结 Deep 对照、独立五折验证、Boundary 与导出门禁仍必须通过。

v4.9 的 StatsFusion 序列读取在 Windows 多进程 DataLoader 下会让每个 worker 独立持有 segment cache，并叠加预取内存。集成配置按用户要求使用 `training.num_workers=10` 和 `training.inference_num_workers=10`，同时将 `loader_prefetch_factor=1`、`session_reader_cache_size=1` 固定为低峰值设置。worker 数量和这两个缓存参数属于可恢复的数据加载运行参数，resume 时可以调整，不改变模型、样本划分或评估协议。

`hierarchical_v4_v49_integrated_repair_20261006c` 以及旧命名 run 仅保留为诊断/恢复证据，不能作为正式 v4.9 输入。正式修复使用全新 `hierarchical_v4_v49_integrated_optimized_*` run，不复用旧候选池、校准器、阈值或 Deep 权重。State 保持单 seed `[2026]`，Deep Verifier 固定三 seed `[2026, 2027, 2028]`；raw IMU 使用训练折 robust normalization，并记录 shape、mask、因果窗口和训练主体。

旧 run 的状态、失败和 fold 结果仍可审计，但不参与新 run 的训练或晋级。

所有工作在一个连续的修复训练阶段完成：准备和审计输入 → 五折 outer State/候选/选择/评估 → 自动协议锁 → fully excluded nested State OOF/Deep/Boundary/部署重训 → 导出与回放。fold 1 后无需人工暂停或执行旧两段式 freeze。

## 修复内容

| 模块 | v4.9 行为与证据 |
|---|---|
| 候选 | 保留 17 种 jitter、600 秒 backtrack/起点扩展、120 秒 merge、proposal-head，transition 关闭。session 内按来源 percentile 排序，proposal-head/短候选/扩展配额为 20%/15%/15%；第一轮保证 family 多样性并为未满足配额留名额。配额缺口显式记录；候选 ID、family、ancestor、父边界和 hash 完整保存。 |
| Verifier | 保持 hidden width、attention heads 和 learned-query/latent bridge 容量，开启现有 raw IMU 分支和三个 seed。三段 30 秒 snippet 使用 10 Hz、六路传感器及六路 mask，最大未来输入 15 秒；robust normalization 仅由模型训练主体拟合，参数随 checkpoint 保存。 |
| State 资格 | selector 最低 epoch 3，仅选择完成训练且 finite、候选召回、校准和 clipping 合格的 epoch。内部 selector event F1 优先，其次 recall、ECE、fragment、soft BCE，相近时取较早 epoch；warmup 后 50% abort 和 20% 晋级 clipping 门禁保持。 |
| checkpoint | 原子保存前再次检查权重有限值、配置哈希、工作区/运行时源码、输入文件身份和受试者 lineage；恢复训练与晋级权重分别保留资格证据。 |
| hard negatives | 按受试者、佩戴手/手关系、IoU、GYRO 缺失、时长、来源及 family 分层，缺失字段保留 unknown。记录候选数和独立 family 数；禁止 family 跨主体、跨训练/预测折。 |
| bootstrap | truth、prediction、ignore、完整 window 时间轴的主体并集；保留纯 FP、无候选和只有观测时间的主体。正式比较使用 seed 2026、2000 次 paired subject bootstrap。 |
| pooled Deep | 每个 outer prediction fold 的上游 State OOF 在整个训练、预处理和校准路径中排除该 fold；缺失 nested State OOF 即停止。Deep 校准使用训练折内的主体 cross-fit。 |
| 操作参数 | threshold/NMS 在对应训练主体的内部 OOF 上选择并锁定，再应用于 outer fold。当前 gap merge=120 秒及候选预算沿用预注册的单一设置；不在 pooled holdout 上重新搜索。部署 threshold/NMS 使用五个训练折锁定参数的中位数。 |
| EndpointRefiner | 覆盖充分且每折至少 60 个独立正事件后进行 cross-fit；entropy `[0.55,0.65,0.75]` 仅在内部 selector 主体上选择，每折锁定后用于该 outer fold，部署取中位数。记录成对 TP、实际端点改变、MAE、F1、TP→FP、有效 bin、clipping 和冲突，失败保留 coarse。 |
| 导出 | 验证完整五折协议锁、数据/源码/配置身份、参考证据和正式 Deep gate；已声明启用的 Boundary 若证据失败则禁止导出，不混接旧模型或阈值。 |

人工指定的固定参数与通过训练数据选择的参数均写入搜索及 lineage 记录。这里的前置五折协议锁只验证训练资格和候选覆盖；Deep F1/FP/h、paired bootstrap、matching、stress 与 Boundary 检查在最终 OOF 产物生成后执行。两种门禁由同一执行器自动串联，前置门禁不以 State F1 冒充 Deep 晋级结果。

## 一条命令执行

在 modeling 根目录执行。没有合格 S0 时显式跳过，并从全新 run 开始：

```powershell
$envName = "bme-model"
$config = "configs/hierarchical_v4_v49_integrated_optimized.yaml"
$run = "hierarchical_v4_v49_integrated_optimized_20261007a"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

conda run --no-capture-output -n $envName python scripts/run_hierarchical_v49_integrated.py `
  --config $config --run-name $run --skip-s0 --fresh
```

同一正式 run 中断后只使用相同配置和源码身份的 `--resume`；模型、数据、协议、checkpoint 或源码改变必须新建 run。worker、prefetch、cache 可在 resume 时调整，训练资格与协议参数不可变。

五折协议锁仍核对候选 run 与冻结 Deep 参考的输入哈希、truth、ignore、观测时间轴及主体划分，避免跨数据版本比较。原方案逐条执行的命令仍支持，五折后须执行：

```powershell
conda run --no-capture-output -n $envName python scripts/check_hierarchical_v4_gates.py `
  --config $config --mode full --run-name $run --skip-s0

conda run --no-capture-output -n $envName python scripts/train_hierarchical_v4_final.py `
  --config $config --run-name $run --fresh

conda run --no-capture-output -n $envName python scripts/export_hierarchical_v4_bundle.py `
  --config $config --run-name $run --fresh
```

任一命令非零退出时停止后续步骤。gate 会写失败清单，final 和 export 不接受缺失、失败或哈希变化的门禁。

如之后补齐真实 S0 五折基线，可在另一个新 run 中用 `--s0-run <真实名称>` 替代 `--skip-s0`。两者必须且只能选一个。提供 S0 时仍执行完整身份、五折完成、数据一致性及逐折候选覆盖比较；省略两个参数或在中途改变策略会停止。

## 正式门禁

| 检查 | 条件 |
|---|---|
| 候选覆盖 | 至少 125/161；proposal-head 目标 126/161；每个 outer fold recall ≥0.70；短事件、手关系、GYRO 缺失和来源分层单独报告，同手/异手覆盖不得低于冻结参考。 |
| Deep | F1 ≥0.5475，FP/h ≤0.0609237。 |
| 手关系召回 | 同手、异手分别相对冻结参考下降 ≤0.02。 |
| paired bootstrap | 超过冻结参考的概率 ≥0.80；参考预测必须重新复现冻结 F1。 |
| matching | 最大匹配与 greedy 不得发生排名反转。 |
| stress | folds 2/3/4 分别相对开发 folds 0/1 的合并 F1 下降 ≤0.03；单独报告 fold 4 FP、阈值、起点 signed error。 |
| Boundary | 起止 MAE 分别改善 ≥10%，F1 下降 ≤0.005，TP→FP ≤1%，残差 clipping ≤5%，有效 bin 和事件身份通过，冲突自动回退。 |

候选覆盖目标 126 为目标记录，硬下限为 125。Boundary 未获得可选启用资格时写 `boundary_failure_report.json` 并保留 coarse；如果最终选择已宣称启用 Boundary，之后正式证据失败则 fail-closed。Deep 正式门禁失败不会降级拼接旧模型。

## 产物与失败处理

- `outputs/v4/experiments/<run>/fold_<fold>/`：resolved config、run manifest、State 训练/选择、候选、评估、数据身份及主体 lineage。
- `outputs/v4/experiments/<run>/v49_gate_manifest_v2.json`：五折证据及 S0/冻结参考哈希、配置/输入/源码/运行环境身份、协议锁。
- `outputs/v4/experiments/<run>/logs/`、`execution_status.json`、`execution_identity.json` 和 `execution_protocol.json`：逐命令 UTF-8 日志、`CREATED/RUNNING/PAUSED/FAILED/COMPLETE` 状态、源码/环境锁及 S0 策略。每条命令记录 worker/prefetch/cache 和采样资源峰值；GPU 资源日志为整卡总量，正式显存验收使用 PyTorch allocator 峰值。
- `outputs/v4/final/<run>/`：nested State、Deep crossfit scores/校准/搜索、hard-negative 分层统计、`v49_deep_gate.json`、Boundary 诊断、部署权重和 `final_manifest.json`。
- `model_bundle/SHA256SUMS.json`：合格部署 bundle 的文件校验；bundle 只包含推理所需源码和模型。
- `failure_report.json`、`failure_evidence/<stage>_<timestamp>/`：失败原因、全证据哈希和诊断/lineage 备份；已有 bundle 隔离到失败证据目录，不能留在部署位置。原 checkpoint 保留。

## 代码验收与正式回放

```powershell
conda run --no-capture-output -n $envName python -m pytest -q
conda run --no-capture-output -n $envName python -m ruff check src tests scripts --no-cache
conda run --no-capture-output -n $envName python -m compileall -q src scripts
git diff --check
conda run --no-capture-output -n $envName python scripts/smoke_test_hierarchical_v4.py --config $config
conda run --no-capture-output -n $envName python scripts/replay_hierarchical_v4_raw_session.py --config $config
```

上述 smoke 和不带 `--bundle` 的 raw replay 验证架构、预处理一致性与随机初始化模型的重复推理，不代表正式 v4.9 checkpoint 或性能验收。回归测试另在隔离子进程中只复制 bundle 推理源码，禁止导入训练包和 XGBoost，执行实际 v4.9 候选 → Verifier 特征 → forward；它仍是合成数据代码测试。

合格 bundle 导出后，执行器自动运行下面的真实 checkpoint 回放：

```powershell
$bundle = "..\..\outputs\v4\final\$run\model_bundle"
conda run --no-capture-output -n $envName python scripts/replay_hierarchical_v4_raw_session.py `
  --config $config --bundle $bundle --forbid-xgboost
```

回放加载实际部署权重，连续运行完整 raw session 两次，比较 State 中间预测、Deep 候选分数和最终事件；误差须 ≤1e-6，GPU 峰值须 <10.5 GB。即使最终没有事件，也检查中间预测。`--forbid-xgboost` 通过阻止导入模拟该包不可用，不修改 Conda 环境。回放结果写入 `bundle_replay.json`。

## 本次实现验证（2026-10-07）

以下结果对应新的 optimized 配置和本次代码实现，仅验证代码、架构与预处理，不作为正式五折性能证据。

| 检查 | 实测结果与范围 |
|---|---|
| pytest | 415 passed、1 skipped；唯一跳过项为 `test_current_r32_canonical_manifest_mask_count_matches_anchors_when_available`，因为现有旧 canonical 输入早于补充后的 mask-count 诊断字段。单独运行 `-rs` 已确认原因。 |
| 静态检查 | ruff、compileall、`git diff --check` 均通过；optimized 配置协议校验及 runner CLI 检查通过。 |
| GPU State smoke | batch 8、1049 steps，forward/backward 峰值约 8.093 GiB，重复推理最大误差 0。 |
| GPU Deep smoke | raw IMU 分支启用，forward/backward 有限值检查通过；峰值约 0.025 GiB，重复推理最大误差 0。 |
| raw-session replay | 一个含 9 个 fragment 的真实 session；全部预处理输入最大误差 0，重复 State logit 最大误差 0。模型为随机初始化。 |
| 隔离推理测试 | 合成候选、raw IMU 特征和 checkpoint 重载通过；禁止 XGBoost/训练模块导入，重复推理误差满足 ≤1e-6。 |
| 状态与恢复 | worker=10、模拟 OOM/Ctrl+C 的终态落盘、运行参数调整及源码身份不一致拒绝恢复的回归测试通过。Windows 进程树 working-set 采样实测返回有效值。 |

真实 session 回放记录为 `outputs/v4/diagnostics/raw_session_replay/a8d98ccd7768b2f2.json`。smoke 的显存数值仅覆盖相应测试 batch；不代表完整训练的系统内存峰值或正式 bundle 回放显存。

正式五折训练尚未启动，新的 F1、FP/h、候选覆盖、Boundary 增益以及真实训练后 bundle replay 均待验证。旧诊断报告不能替代 `integrated_repair_v2` 新 run 的门禁证据。

## 预期边界

125–130/161 覆盖、F1 0.55–0.62、FP/h 0.055–0.0609 和起点 MAE 150–220 秒均是待验证的条件性预期。只有新 run 的完整独立验证通过后才能更新性能结论。
