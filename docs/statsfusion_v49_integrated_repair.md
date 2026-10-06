# v4.9 一体化修复训练协议

## 当前状态与范围

本次实现对应配置 `configs/hierarchical_v4_v49_integrated_repair.yaml`。代码实现和训练结果分开验收；尚无合格的 v4.9 五折结果或部署 bundle，不能将目标 F1 写成实测结果。

本地已核实冻结 Deep 参考 `hierarchical_v4_r32_deep_frontier_20260930a`，其可引用 F1 为 0.5375。尚未找到同时具备 sensor-only 身份、五折 EVALUATED 清单和一致输入数据的 S0 基线。一体化执行器在耗时训练前检查此项，缺失或不符即停止。冻结 Deep 参考启用了统计分支，不能替代 S0。

中断的 `hierarchical_v4_v49_integrated_repair_20261006a` 使用过修复期间的代码，不得继续训练或作为晋级证据。源码或配置变化后使用全新 run name；只有同一代码、配置、数据身份下的中断可以 `--resume`。旧 v4.7.1/v4.8/v5.2 模型及校准器不作为新 run 的训练输入。

所有工作在一个连续的修复训练阶段完成：准备和审计输入 → 五折 outer State/候选/选择/评估 → 自动协议锁 → fully excluded nested State OOF/Deep/Boundary/部署重训 → 导出与回放。fold 1 后无需人工暂停或执行旧两段式 freeze。

## 修复内容

| 模块 | v4.9 行为与证据 |
|---|---|
| 候选 | 17 种对称 jitter、600 秒 backtrack、600 秒起点扩展、120 秒 gap merge、proposal-head；transition 关闭。保留候选 ID、family、ancestor、父边界、父来源、父候选 SHA-256 和完整生成配置，检查来源、边界及身份一致性。 |
| Verifier | 保持已有 hidden width、attention heads 和 learned-query/latent bridge 容量；增加来源、相对父边界偏移、时长及局部质量输入。扩展边界作为可拒绝的候选，保留 coarse parent 供模型辨别。 |
| State 资格 | selector 最低 epoch 3，只从完成训练且有限值、候选召回、校准和 clipping 均合格的 epoch 中选择；不合格时停止。warmup 后 clipping 超过 50% 即失败，晋级 checkpoint 要求 clipping 不超过 20%。 |
| checkpoint | 原子保存前再次检查权重有限值、配置哈希、工作区/运行时源码、输入文件身份和受试者 lineage；恢复训练与晋级权重分别保留资格证据。 |
| hard negatives | 按受试者、佩戴手/手关系、IoU、GYRO 缺失、时长、来源及 family 分层，缺失字段保留 unknown。记录候选数和独立 family 数；禁止 family 跨主体、跨训练/预测折。 |
| bootstrap | truth、prediction、ignore、完整 window 时间轴的主体并集；保留纯 FP、无候选和只有观测时间的主体。正式比较使用 seed 2026、2000 次 paired subject bootstrap。 |
| pooled Deep | 每个 outer prediction fold 的上游 State OOF 在整个训练、预处理和校准路径中排除该 fold；缺失 nested State OOF 即停止。Deep 校准使用训练折内的主体 cross-fit。 |
| 操作参数 | threshold/NMS 在对应训练主体的内部 OOF 上选择并锁定，再应用于 outer fold。当前 gap merge=120 秒及候选预算沿用预注册的单一设置；不在 pooled holdout 上重新搜索。部署 threshold/NMS 使用五个训练折锁定参数的中位数。 |
| EndpointRefiner | 覆盖门禁通过且每折有至少 60 个独立训练正事件后才进行跨主体 cross-fit；分别检查起终点 MAE、F1、TP→FP、有效 bin、残差 clipping 及冲突。未获得启用资格时保持 coarse boundary。 |
| 导出 | 验证完整五折协议锁、数据/源码/配置身份、参考证据和正式 Deep gate；已声明启用的 Boundary 若证据失败则禁止导出，不混接旧模型或阈值。 |

人工指定的固定参数与通过训练数据选择的参数均写入搜索及 lineage 记录。这里的前置五折协议锁只验证训练资格和候选覆盖；Deep F1/FP/h、paired bootstrap、matching、stress 与 Boundary 检查在最终 OOF 产物生成后执行。两种门禁由同一执行器自动串联，前置门禁不以 State F1 冒充 Deep 晋级结果。

## 一条命令执行

在 modeling 根目录执行。先把 `$s0Run` 替换为真实、已核验的 S0 run；没有该基线时不要填写冻结 Deep 名称或自行伪造证据。

```powershell
$envName = "bme-model"
$config = "configs/hierarchical_v4_v49_integrated_repair.yaml"
$run = "hierarchical_v4_v49_integrated_repair_20261006b"
$s0Run = "<真实且已核验的 sensor-only 五折 run>"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

conda run --no-capture-output -n $envName python scripts/run_hierarchical_v49_integrated.py `
  --config $config --run-name $run --s0-run $s0Run --fresh
```

同一阶段中断后将最后的 `--fresh` 改成 `--resume`，使用同一个 run name。执行器读取实际 fold stage 和 final manifest 恢复，不需要另建自动续跑监视器。未开始的新 run 使用 `--fresh`；已有目录使用 `--fresh` 会被拒绝。

执行器先完整核验 S0 的定义和产物，再准备输入；准备后再次核对 S0 与当前输入哈希。五折协议锁还比较 S0、候选 run、冻结 Deep 参考的 truth、ignore、观测时间轴与主体划分，避免跨数据版本比较。原方案逐条执行的命令仍支持，五折后须执行：

```powershell
conda run --no-capture-output -n $envName python scripts/check_hierarchical_v4_gates.py `
  --config $config --mode full --run-name $run --s0-run $s0Run

conda run --no-capture-output -n $envName python scripts/train_hierarchical_v4_final.py `
  --config $config --run-name $run --fresh

conda run --no-capture-output -n $envName python scripts/export_hierarchical_v4_bundle.py `
  --config $config --run-name $run --fresh
```

任一命令非零退出时停止后续步骤。gate 会写失败清单，final 和 export 不接受缺失、失败或哈希变化的门禁。

## 正式门禁

| 检查 | 条件 |
|---|---|
| 候选覆盖 | 至少 125/161；proposal-head 目标 126/161；同手/异手覆盖不得低于冻结参考，逐折不低于 S0 对应覆盖。 |
| Deep | F1 ≥0.5475，FP/h ≤0.0609237。 |
| 手关系召回 | 同手、异手分别相对冻结参考下降 ≤0.02。 |
| paired bootstrap | 超过冻结参考的概率 ≥0.80；参考预测必须重新复现冻结 F1。 |
| matching | 最大匹配与 greedy 不得发生排名反转。 |
| stress | folds 2/3/4 分别相对开发 folds 0/1 的合并 F1 下降 ≤0.03；单独报告 fold 4 FP、阈值、起点 signed error。 |
| Boundary | 起止 MAE 分别改善 ≥10%，F1 下降 ≤0.005，TP→FP ≤1%，残差 clipping ≤5%，有效 bin 和事件身份通过，冲突自动回退。 |

候选覆盖目标 126 为目标记录，硬下限为 125。Boundary 未获得可选启用资格时写 `boundary_failure_report.json` 并保留 coarse；如果最终选择已宣称启用 Boundary，之后正式证据失败则 fail-closed。Deep 正式门禁失败不会降级拼接旧模型。

## 产物与失败处理

- `outputs/v4/experiments/<run>/fold_<fold>/`：resolved config、run manifest、State 训练/选择、候选、评估、数据身份及主体 lineage。
- `outputs/v4/experiments/<run>/v49_gate_manifest.json`：五折证据及 S0/冻结参考哈希、配置/输入/源码身份、协议锁。
- `outputs/v4/experiments/<run>/logs/` 和 `execution_status.json`：逐命令 UTF-8 日志与当前执行状态。
- `outputs/v4/final/<run>/`：nested State、Deep crossfit scores/校准/搜索、hard-negative 分层统计、`v49_deep_gate.json`、Boundary 诊断、部署权重和 `final_manifest.json`。
- `model_bundle/SHA256SUMS.json`：合格部署 bundle 的文件校验；bundle 只包含推理所需源码和模型。
- `failure_report.json`、`failure_evidence/<stage>_<timestamp>/`：失败原因、不可部署标记、全证据哈希和诊断/lineage 备份。原权重保留，不生成晋级 bundle。

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

## 预期边界

125–130/161 覆盖、F1 0.55–0.62、FP/h 0.055–0.065 和起点 MAE 150–220 秒均是待验证的条件性预期。FP/h 预期范围上部可能不满足正式上限；达成预期范围不等于晋级。只有新 run 的完整独立验证通过后才能更新性能结论。
