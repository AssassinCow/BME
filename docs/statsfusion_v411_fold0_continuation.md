# v4.11 单 fold0 与 64 轮接续

## 运行范围

`scripts/run_statsfusion_joint.py` 支持 `--evaluation-mode fold0|fivefold`。未指定时读取 `execution.mode`，旧配置默认 `fivefold`。运行范围属于冻结配置：同一个 run 的 resume 不能从单折切换为五折。`--fold 0` 仅限制当前 outer 执行，不替代单折模式，也不会跳过完整五折正式门禁。

单折模式保留原来的 subject 划分、三个内部 OOF 分区、35% selector、选模后从头 retrain、自然分布校准、完整操作点搜索和 outer fold0 测试。只跳过其余四个 outer fold、局部消融及全数据部署重训。不会把 selector 的搜索后 F1 当作独立测试成绩。

单折 `all` 依次执行 audit、validate、outer fold0、单折资格检查、保存 fold0 模型、单折导出和 replay。训练资格、内部 OOF、校准、subject 排除和启用的 Boundary 仍须通过。F1、FP/h、候选覆盖和 stress 与目标的差距单独记录；单折产物始终标为 `single_fold_diagnostic`、`promotable=false`，不替代正式五折晋级证据。

## 本次接续

父 run：`statsfusion_v411_learned_boundary_20261008a`。

新 run：`statsfusion_v411_fold0_continue64_20261009a`。

新配置：`configs/statsfusion_v411_fold0_continue64.yaml`。相对于原配置仅将 `training.max_epochs` 从 32 增至 64，新增 `execution.mode=fold0`。

现有 32 轮完成的是 `fold_0/internal/partition_0/selection`，并非全部 outer fold0 已完成。复制该 selector 的模型、AdamW 状态、梯度、Python/NumPy/Torch/CUDA RNG、最佳 epoch、早停计数、追加日志及日志游标。新 run 从 epoch33 开始，上限 64；每两轮验证、min delta=0.003、三次无显著改善的早停规则保持不变，因此不保证必定跑到 64 轮。

模型结构、损失、采样、优化器、学习率、梯度门禁、受试者划分、校准和搜索逻辑不变。迁移前逐文件限制源码变更，并对 trainer、验证推理和搜索剥离日志/进度包装后的 AST 进行等价检查。父 run 的源码快照、配置、原 resume 和原身份复制到 `continuation_parent`；`continuation_manifest.json` 保存复制前后 hash、身份重绑定和来源证据。

不复制旧 epoch cache、尚未完成的 retrain、最终校准器、outer 预测或 deployment 产物。旧 selector 搜索文件仅作为历史开发证据保留；新的验证、内部 OOF、最终校准和测试结果在新 run 中生成。新 run 从临时目录完整提交后才出现在 `runs`，中断复制不会留下可误 resume 为正式接续的半成品。

`--max-epochs` 可为全新 run 设置轮数上限；直接给旧 run 加 `--max-epochs 64 --resume` 不会绕过身份校验。轮数接续应使用专用复制工具，不能手改 checkpoint 身份或放宽普通 resume。

## 显示与保存

训练 chunk、来源 CDF、selector 推理、校准预算准备及完整操作点搜索均显示进度条。搜索进度以 27 个 decoder/budget 组表示，仍执行原来的 8019 个粗操作点；多进程结果按原注册顺序归并。验证完成打印 F1、TP/FP/FN、precision/recall、候选覆盖、FP/h、起止 MAE、起点偏差、ECE、操作点、资格和早停状态。outer 测试完成单独打印锁定操作点后的测试指标。

每个 run 的 `console.log` 保留终端输出。checkpoint 仍每个 chunk 提交，异常仍记录 PAUSED/FAILED；未提交日志尾部仍按恢复游标截断。

单折结果位于该 run：

- `fold_0/evaluation.json`：独立 outer fold0 测试与边界诊断。
- `fold0_gate_manifest.json`：单折资格与目标差距。
- `fold0_final/checkpoint.pt`：测试时使用的 outer fold0 模型，测试 subject 未参与训练。
- `fold0_final/final_manifest.json`：模型、校准、OOF、测试排除和指标身份。
- `fold0_bundle/`：模型、校准、来源映射、源码与 `SHA256SUMS.json`。

单折 bundle 使用 `bme_eating.joint_single_fold.StatsFusionJointFoldDetector`，继承原推理接口。正式五折 loader 仍拒绝单折产物。失败时保留最终 checkpoint 和诊断，并将无效或不完整 bundle 移到 `failures`，不能误认为可正式晋级。

## 命令

2026-10-10：用户选择仅执行一个内部 partition，当前接续 run 已完成保留状态的身份迁移。该 run 后续恢复改用 [单 partition 接续说明](statsfusion_v411_single_partition.md) 中的新配置与命令；以下旧三分区配置仅用于说明迁移前流程，不能再用于已迁移 run 的 resume。

在 `modeling_joint_v411` 中执行。首次接续复制只执行一次；本次实现已准备新 run 后，无需重复复制。

```powershell
conda run --no-capture-output -n bme-model python scripts/fork_statsfusion_joint_run.py `
  --config configs/statsfusion_v411_fold0_continue64.yaml `
  --parent-run statsfusion_v411_learned_boundary_20261008a `
  --run-name statsfusion_v411_fold0_continue64_20261009a
```

继续训练：

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v411_fold0_continue64.yaml `
  --run-name statsfusion_v411_fold0_continue64_20261009a `
  --stage all --evaluation-mode fold0 --skip-s0 `
  --reference-run statsfusion_v410_joint_event_20261007a --resume
```

完整五折模式需使用另一个全新 run。以下示例从头训练，不能加到单折接续 run 上：

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v411_learned_boundary.yaml `
  --run-name statsfusion_v411_fivefold64_20261009a `
  --stage all --evaluation-mode fivefold --max-epochs 64 --skip-s0 `
  --reference-run statsfusion_v410_joint_event_20261007a --fresh
```

该完整五折 run 恢复时保留所有参数，只将 `--fresh` 改为 `--resume`。
