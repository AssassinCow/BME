# 事件级救援 v5：分步骤运行

v5 保留每一个冻结 XGBoost 事件，只追加通过分数筛选且远离基线事件的 DTP 事件。
`q=0.90` 是**训练受试者等权的事件分数第 90 百分位**，不是概率阈值 0.90。
三个 meta split 分别在其余两个分区拟合数值阈值；用于 outer 的阈值由完整
outer-train OOF 重拟合。q、120 秒排除间隔、解码器和门禁没有搜索入口。

## 证据与限制

批准方案时的开发 OOF 目标如下；正式 fit 会断言 TP/FP/FN、异侧召回、strict F1
和起止 MAE 复现。这里的 F1 不是 outer 测试 F1，也不是独立验证结果。

| 开发 OOF | XGBoost F1 | v5 F1 | 异侧召回变化 | FP/hour 比例 |
|---|---:|---:|---:|---:|
| fold 0 | 0.530769 | 0.558304 | 0.402778 → 0.500000 | 1.209677 |
| fold 1 | 0.482517 | 0.512821 | 0.397260 → 0.520548 | 1.170455 |

fold 0 的 FP/hour 门禁失败，不能四舍五入成通过。当前用户批准的是登记后的
folds 2–4 **统一压力测试**，不是把开发失败改写为通过。三个 stress fold 全部
完成后汇总决定晋级，不能观察一折后换阈值、跳过差的一折或切换工作点。
这些受试者已间接参与之前的 OOF 开发，不能称为完全独立未见验证。

v5 复用原 legacy dual EMA 的全部语义：fast/slow 半衰期 12/36 秒，启动阈值
0.55/0.30，退出比例 0.25，关闭持续 60 秒，最短事件 30 秒，DTP 内部合并
间隔 60 秒，边界回看 60 秒。旧解码器使用已有冻结 start/end 概率；v5 没有
新增、训练或替换边界头。分数为合并前 `mean(max(fast, slow))`，legacy 合并
使用最高子事件分数；这不是校准后的事件正确概率。

同 subject/session 内，与基线事件重叠或最近间隔 **≤120 秒**的 DTP 事件拒绝。
其余按原边界和原分数追加，不重新合并整个输出。基线逐字段保留，并断言
每个 meta 分区 TP/recall 不下降。保持原 session 标识和旧解码器的时间缺口语义，
不在 v5 内另加 6 秒切段规则；缺口统计写入 selection。改变 session 切分属于
另一个协议，不能在本次压力测试中更换。

XGBoost 的后处理由原 outer-train OOF 选定，并非针对本次 meta heldout
重新嵌套拟合；现有 baseline 的补录来源清单也不等价于重新训练的完整来源证明。
保留这些限制与原输入哈希，不能因为 v5 清单来自干净 Git 就提升旧来源的证据等级。

## 1. 审阅代码并提交，拟合开发折

先运行测试、审阅并提交当前改动。正式 fit、登记、训练、评价都要求干净 Git；
不需要为了测试而提交，`validate_fusion_event_rescue.py` 是明确非正式的开发检查。
登记后不要改代码或配置；发生变化必须开启新协议实验，不能续跑旧 run。

```powershell
conda activate bme-model
python -m pytest tests/test_fusion_event_rescue.py tests/test_fusion_v4_cli.py -q
```

审阅、提交并同步后，在项目根目录分别运行：

```powershell
$sourceRun = "baseline_dtp_fusion_clean_20260920a"
$rescueRun = "fusion_event_rescue_v5_20260923a"

python scripts/fit_fusion_event_rescue.py `
  --config configs/dtp_fusion_event_rescue.yaml `
  --source-run $sourceRun `
  --run-name $rescueRun `
  --fold 0 `
  --fresh
if ($LASTEXITCODE -ne 0) { throw "v5 fold 0 fit failed" }
```

```powershell
python scripts/fit_fusion_event_rescue.py `
  --config configs/dtp_fusion_event_rescue.yaml `
  --source-run $sourceRun `
  --run-name $rescueRun `
  --fold 1 `
  --fresh
if ($LASTEXITCODE -ne 0) { throw "v5 fold 1 fit failed" }
```

`--fresh` 创建该 fold 的全新目录，不覆盖已有结果。fit 中断后用原命令将
`--fresh` 换为 `--resume`；签名覆盖预测、标签/忽略区间、划分、代码、Git 和
配置。v5 没有耗时网格，部分 fit 恢复时重做固定解码和证据计算；已完成的 fit
只校验清单后返回，不重新选参数。代码或输入变化会拒绝恢复。

## 2. 登记统一协议

```powershell
python scripts/register_fusion_event_rescue.py `
  --config configs/dtp_fusion_event_rescue.yaml `
  --run-name $rescueRun
if ($LASTEXITCODE -ne 0) { throw "v5 registration failed" }
```

写入 run 根目录的 `protocol_registration.json`。登记会核验开发折重放、冻结
代码/config/source run，并明确记录 fold 0 的已知失败。登记本身不是晋级。
同一 run 的登记不能改写；重复同一登记只校验一致性。

## 3. 只生成 folds 2–4 DTP 冻结预测

保持原模型、训练配置与三个 crossfit 分区，不重训 folds 0/1。以下只生成预测，
不执行 v4 融合调参或 outer 评价；v4 原门禁及历史失败记录保留。

```powershell
2..4 | ForEach-Object {
  python scripts/train_fusion.py `
    --config configs/dtp_fusion_event_rescue.yaml `
    --run-name $sourceRun `
    --event-rescue-run $rescueRun `
    --fold $_
  if ($LASTEXITCODE -ne 0) { throw "DTP source fold $_ failed" }
}
```

训练中断时仍用 `train_fusion.py --resume <该 fold 的 crossfit_N/last.pt>`，并保留
相同 config、run-name、event-rescue-run、fold。已完成分区按原训练器校验后跳过。
如果 folds 2–4 已用其他配置训练过，先核对来源，不覆盖已有目录。

## 4. 先拟合所有 stress fold，再统一评价

```powershell
2..4 | ForEach-Object {
  python scripts/fit_fusion_event_rescue.py `
    --config configs/dtp_fusion_event_rescue.yaml `
    --source-run $sourceRun `
    --run-name $rescueRun `
    --fold $_ `
    --fresh
  if ($LASTEXITCODE -ne 0) { throw "v5 OOF fold $_ failed" }
}
```

```powershell
2..4 | ForEach-Object {
  python scripts/evaluate_fusion_event_rescue.py `
    --config configs/dtp_fusion_event_rescue.yaml `
    --run-name $rescueRun `
    --fold $_
  if ($LASTEXITCODE -ne 0) { throw "v5 outer fold $_ evaluation failed" }
}
```

评价不开放 source-run、q、缓冲、工作点或覆盖开关。每折产物在 `fold_N/outer`
一次性提交；并发运行由进程锁阻止。崩溃前已开始的评价保留 `outer.pending`
及 attempt 记录，并拒绝直接重跑，防止在看过结果后静默换参数。保留该目录
以诊断中断，不能通过删除目录当成新实验。

## 5. 汇总并按原门禁决策

```powershell
python scripts/summarize_fusion_event_rescue.py `
  --config configs/dtp_fusion_event_rescue.yaml `
  --run-name $rescueRun
if ($LASTEXITCODE -ne 0) { throw "v5 summary failed" }
```

汇总用三个互不重叠的 outer 受试者的事件、标签和观测时长重新计算 pooled
指标，不平均各折 F1。`stress_summary.json` 保存所有硬门禁及最终
`submission_candidate`：未通过就是 `xgboost`。指标未达标不属于文件执行错误，
因此单折评价/汇总退出码 0 代表记录成功，不代表模型晋级。

硬门禁：pooled F1 +0.02、异侧召回 +0.03、FP/hour ≤1.2×、起止 MAE 各 ≤1.1×、
strict F1 不降、基线事件保留、TP/recall 不降；三折至少两折 F1 不降且任一折
下降不超过 0.03。所有 stress meta 分区 F1/strict F1 降幅不得超过 0.01。
受试者配对 bootstrap 固定 1000 次、seed 2026，只报告区间，不能挽救失败门禁。

## 产物、推理与性能

每个 fit 保存 `selected_event_rescue.json`、完整 meta 分区范围/数值阈值、
meta-OOF 事件、接受/拒绝候选及原因、逐受试者/手别/失败案例、bootstrap 和
`run_manifest.json`。outer 保存同口径事件与指标；登记和汇总另外记录其输入
清单哈希。原 XGBoost、DTP source、v4 和历史消融目录只读。

部署 API 为 `apply_frozen_event_rescue(baseline_predictions, dtp_predictions, selection)`。
它只做两次固定解码、稳定排序和事件追加；不读标签、不拟合阈值、不搜索、不
bootstrap。`evaluate_...` 是离线评价工具，会读取标签并计算诊断，不能当成
官方单样本推理入口。

开发验证和性能检查（允许未提交代码，产物**不能用于正式登记**）：

```powershell
python scripts/validate_fusion_event_rescue.py `
  --config configs/dtp_fusion_event_rescue.yaml `
  --source-run $sourceRun `
  --run-name fusion_event_rescue_v5_devcheck_unique
```

产物在 `v2/diagnostics/<run-name>`。性能使用 OOF 中固定的 84,913 / 169,826 /
339,652 窗口前缀，仅计后处理，输入加载、神经网络推理、标签和 bootstrap 不计入。
RSS 每 10ms 采样，报告绝对峰值及相对已加载输入的增量，可能漏掉更短暂峰值。
事件扫描为 O(B+D)，排序为 O(N log N)；实测缩放用于检查异常开销，不能把单次
计时当成严格的复杂度证明。官方运行时限尚未知，不预设虚构的绝对合格线。
