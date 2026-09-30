# StatsFusion v4.7.1 运行手册

## 唯一正式路线

- 协议：`statsfusion-r3.2`
- 配置：`configs/hierarchical_v4_r32_deep_frontier.yaml`
- state、Deep verifier、Boundary seed：`2026`
- state：每 fold 一个 subject-disjoint holdout，最多 32 epochs
- early stopping：至少 3 epochs，连续 3 次无稳健改进停止
- checkpoint：从 epoch 1 起参与选择；稳健指标仍使用三轮滚动窗口，前两轮只在已有历史窗口可计算时参与
- 后端：五折 outer OOF 汇总后依次比较 state-only、pooled Logistic、pooled Deep verifier
- Boundary：只在 verifier 锁定后训练；门禁失败自动保留 coarse boundary

旧 r0/r1/r2/r3/r3.1、v4.5.1 和旧 checkpoint 选择配置均为 blocked predecessor，不能
resume。`canonical_input_r3_2` 只有在完整身份验证通过时才允许只读复用。

## 完整命令

```powershell
$config = "configs/hierarchical_v4_r32_deep_frontier.yaml"
$run = "hierarchical_v4_r32_deep_frontier_20260930a"

python scripts/prepare_statsfusion_v4_inputs.py --config $config --workers 8 --resume
python scripts/audit_statsfusion_feature_provenance.py --config $config --resume

foreach ($fold in 0..1) {
  python scripts/train_hierarchical_v4_state.py --config $config --run-name $run --fold $fold --fresh
  python scripts/build_event_candidates_v4.py --config $config --run-name $run --fold $fold --resume
  python scripts/select_hierarchical_v4_pipeline.py --config $config --run-name $run --fold $fold --resume
  python scripts/evaluate_hierarchical_v4.py --config $config --run-name $run --fold $fold --resume
}

# pooled Deep/Boundary final training requires hash-locked promotion evidence.
# Fill these with an existing, valid sensor-only/baseline run; do not use $run itself.
$s0Run = "<existing-s0-run-name>"
# After fold 0/1 evidence exists, run the registered ablation comparisons:
# python scripts/check_hierarchical_v4_gates.py --config $config --mode ablation --run-name $run --s0-run $s0Run --compare <TYPE>:<BASELINE>:<CANDIDATE>
# Then run development/stress gates with the real baseline:
# python scripts/check_hierarchical_v4_gates.py --config $config --mode development --run-name $run --s0-run $s0Run
# Freeze after the gates pass (required before final pooled heads):
# python scripts/freeze_hierarchical_v4_protocol.py --config $config --run-name $run --fold 1 --fresh

foreach ($fold in 2..4) {
  python scripts/train_hierarchical_v4_state.py --config $config --run-name $run --fold $fold --fresh
  python scripts/build_event_candidates_v4.py --config $config --run-name $run --fold $fold --resume
  python scripts/select_hierarchical_v4_pipeline.py --config $config --run-name $run --fold $fold --resume
  python scripts/evaluate_hierarchical_v4.py --config $config --run-name $run --fold $fold --resume
}

# Run after folds 2-4 have been evaluated:
# python scripts/check_hierarchical_v4_gates.py --config $config --mode stress --run-name $run --s0-run $s0Run

python scripts/train_hierarchical_v4_final.py --config $config --run-name $run --fresh
python scripts/export_hierarchical_v4_bundle.py --config $config --run-name $run --fresh
```

如果没有可核验的 `$s0Run` 及其五折评估证据，前面的五折 state/candidate/evaluation 仍可运行，
但不要伪造 gate；`train_hierarchical_v4_final.py` 会按设计拒绝缺失或失败的 promotion/stress/freeze
证据。中断后只对同一阶段改用 `--resume`，不要重新使用 `--fresh`。

同一阶段中断时，把该命令的 `--fresh` 改为 `--resume`。Git commit、dirty worktree 和运行时代码
SHA 变化只写入 `source_identity_history`，不会要求提交 Git、重建 scaler 或重启 run。结果相关
配置、数据/canonical identity、subject folds、truth/ignore、训练/预测受试者集合及模型/OOF 父
artifact 哈希仍严格锁定；模型张量结构不兼容时 checkpoint 加载会立即失败。经过代码变化续跑的
结果应标记为混合源码证据，manifest 保留每次身份迁移供复核。

## 单折诊断头

时间不足时，可在某一 fold 完成 state 后训练 Logistic、Deep verifier 和 Boundary 诊断头：

```powershell
python scripts/train_hierarchical_v4_single_fold_heads.py `
  --config $config --run-name $run --fold 0 --fresh
```

该结果只覆盖一个 outer fold，不是五折 pooled 证据，不得作为最终 bundle 的晋级依据。

## 数据与标签契约

- 训练与 raw-session 推理均使用 session-wide 3 秒右端点网格。
- 3 秒 anchor `t` 表示 `(t-3s,t]`；15 秒 grid `g` 表示 `(g-15s,g]`。
- truth/ignore 按 `subject_key + session_id` 划分。
- ignore 及最长 60 秒因果提示区间不产生 state、onset、offset 或 smooth 梯度。
- state target 是软 occupancy；校准使用 Soft-Platt、soft Brier 和 soft ECE。
- 候选匹配仍使用严格 `IoU > 0.25`；官方一对一匹配细节仍为 `UNKNOWN`。
- 最大未来数据为 60 秒；当前正式 decoder 禁用 Semi-Markov 和 transition candidates。

## 训练参数

- `batch_size=8`
- `gradient_accumulation=4`
- 有效 batch 为 32
- 每位训练受试者每 epoch 采样 1000 clips
- `learning_rate=0.00015`
- `warmup_fraction=0.10`
- `gradient_clip_norm=5.0`
- `validation_every_epochs=1`
- promotion recall 跑通门槛 `0.70`，开发目标 `0.75`

selector 在训练受试者内部再按 subject 隔离；固定 seed、完整 selector session 时间轴和三轮滚动
稳健指标用于选择 epoch。重训使用所选 epoch，但保持同一 32-epoch cosine schedule horizon，避免
因缩短 horizon 改变前期学习率轨迹。

## Pooled heads

五折 state 全部完成后，final 阶段才训练 pooled heads。每个预测 fold 的 Logistic、Deep verifier、
校准器和 Boundary 只使用其余四折候选；训练、selector、calibration 与 prediction subjects
存在交集时立即失败。模型门禁失败时按 `Deep -> Logistic -> state-only` 自动回退，Boundary
失败时保留 coarse endpoint。

pooled head 的训练候选不能直接复用其他 outer fold 的 state OOF：那些 state 模型可能训练过
当前预测 fold。final 会对每个预测 fold 排除其全部受试者，并在余下四折重新做三分区、单 seed、
按对应 outer fold 所选 epoch（上限 5）训练 state OOF；中断后可校验哈希并恢复。总计最多增加 15 次短 state 训练，
旧 pooled final 缺少 `fully_excluded_nested_state_oof_v1` 证据时拒绝导出或加载。

完整 pooled OOF 允许联合选择 decoder、模型类型和部署阈值，这是时间受限开发取舍，报告必须
标记为 development/stress evidence。verifier 无法找回 state 阶段未生成的事件，候选召回必须
单独报告。

## 推理与 bundle

公共入口保持：

```python
HierarchicalEatingDetectorV4.predict_session(session: RawSessionInput) -> list[Event]
```

`RawSessionInput` 必须是 `statsfusion-raw-v2`、毫秒时间戳、原生未标准化采样，并严格使用
`[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]` 通道顺序。bundle 不包含 XGBoost 工件、训练
标签、受试者名单、个人绝对路径或凭据，并必须通过 `SHA256SUMS.json` 完整性校验。

## 验收

```powershell
python -m pytest -q
python -m ruff check src tests scripts --no-cache
python -m compileall -q src scripts
git diff --check
python scripts/smoke_test_hierarchical_v4.py --config $config
python scripts/replay_hierarchical_v4_raw_session.py --config $config
```

GPU smoke 要求峰值显存低于 10.5 GB，同 checkpoint 重复推理概率误差不超过 `1e-6`。最终还需
在禁止 `import xgboost` 的最小环境中执行 bundle raw-session replay。官方文件适配器仍为
`UNKNOWN`，收到官方接口后只增加薄 adapter，不改变模型预处理语义。
