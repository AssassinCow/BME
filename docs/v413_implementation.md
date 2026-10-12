# v4.13 实施与验收

2026-10-11深入审计补充：生产BF16挖掘导致任务头梯度丢失、无效warmup导致TCN梯度放大，两项已修复。此前手工backward验收不足以覆盖正式采样顺序；当前验收增加真实三轮生产训练与头部更新断言。详情见docs/v413_deep_audit_20261011.md及manifests/v413_deep_audit.json，后者登记当前源码与新证据。

## 身份和隔离

- 架构：statsfusion_joint_event_v413；协议：statsfusion-joint-event-v4；schema：v413；seed：2026。
- 默认 fold0/single，最多64轮。13人fit/7人selector选轮次，20人重新训练并预测11人内部留出，31人重新训练并预测8人fold0。
- 新项目拥有独立Git对象库，复制来源HEAD及工作树修复。来源HEAD、staged/unstaged diff及253个来源文件hash见 manifests/v413_source_provenance.json 与同目录patch。
- 原v4.12源码和运行产物不修改。新模型不加载任何旧权重、optimizer、CDF或校准器。

## 已实现路径

1. 分数有效率加权平均使用真实权重和；无有效输入返回零。3秒proposal保持输入共享特征。
2. 共享raw分支一次编码，公开可选block表示；定位最大感受野45秒，包含15秒PPG输入支持、最多12秒已完成块保持及两层因果卷积的18秒历史。时间缓存201维，ROI40×259。
3. 五个ROI区域各自编码；五个区域summary、四个差分和query经640→64残差比较；Boundary仍132维。
4. 稀疏端点loss采用固定1:20峰值/背景均值，两个端点取平均；bias初始化概率0.01。
5. epoch3开始，固定session权重自然eval/no_grad回放18个decoder，保存权重/数据/session/源码绑定与hash。释放游标保证未来候选不可用于当前chunk；回放恢复全部RNG和模块训练状态。
6. 先保留随机背景family，事件头BCE挖掘32个family、最多96个版本；随机时长15–3600秒。truth-jitter的模型metadata使用已观测自然父信息。
7. 可替换正组CPU梯度；事件内组平均、session内事件平均，纯负family按实际session数量平均；四session更新正/负分别按有贡献session平均。Boundary和排序仅平均合法贡献，最后不足四session也使用实际计数。
8. 时间线监督使用session有效计数；有界subject权重；epoch1–8固定增强强度调度；零梯度夹角输出null。
9. 完整18×3×3×99=16038点搜索。相同几何首次可用decision消费一次预算；CDF使用训练父来源/尺度并支持父来源及全局回退。
10. 校准优化成功与区分能力分别登记；常数或近零斜率不能通过资格。合格检查才消耗F1早停耐心，epoch24无资格或连续三次失去资格记录诊断失败。
11. 验证按session收集、按自身decision遮罩ROI后batch128评分，恢复原decision顺序；部署即时评分。缓存布局与旧版隔离。
12. 独立v413能力注册贯穿数据、模型、训练、搜索、推理、参考、门禁及bundle。固定decoder曲线只用于诊断，不能参与选模。

## 验证和限制

CPU测试、真实数据10-worker短审计、合成13023秒事件和无XGBoost bundle回放用于实现验证。短审计checkpoint明确为disposable，不进入正式run。

GPU训练或测试必须先通过require_idle_gpu。原v4.12进程仍存在时不会抢占资源。GPU峰值10.5GB和进程树private RAM26GB门禁保留。

性能脚本使用同一v413模型、候选和完整16038搜索点对照参考执行；two_times_target_met仅针对本次测量范围登记。少量session的CPU对照不能证明完整selector集合或GPU吞吐达到两倍。真实长短session、GPU FP32/BF16及bundle回放已通过；完整Selector（含CDF）耗时与全训练自然回放成本尚未测量。

默认产物均promotable=false，fold0已用于开发。实现测试、单折/五折诊断、正式门禁三个状态分别保存；未训练的新版本不能声称F1改善。

## GPU验收复核命令

本次已在RTX 4080 Laptop GPU上完成以下检查。后续复核仍需等待GPU空闲，不与已有训练并行。

```powershell
Set-Location -LiteralPath "C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v413"
conda run --no-capture-output -n bme-model python scripts/validate_statsfusion_joint.py --config configs/statsfusion_v413_integrated_repair.yaml --device cuda --output .modeling/v413_gpu_smoke.json
conda run --no-capture-output -n bme-model python scripts/audit_statsfusion_v413_training.py --device cuda --workers 10 --output .modeling/v413_gpu_training_audit.json
conda run --no-capture-output -n bme-model python scripts/audit_statsfusion_v413_production.py --device cuda
conda run --no-capture-output -n bme-model python scripts/audit_statsfusion_v413_replay.py --device cuda --audit .modeling/v413_gpu_training_audit.json --output .modeling/v413_gpu_session_replay.json
```

## 正式执行

```powershell
Set-Location -LiteralPath "C:\Users\LZX\Desktop\University\Competition\生医工\project\modeling_joint_v413"
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v413_integrated_repair.yaml `
  --run-name statsfusion_v413_integrated_repair_20261011a `
  --stage all --evaluation-mode fold0 --internal-mode single --skip-s0 `
  --reference-run statsfusion_v411_fold0_continue64_20261009a --fresh
```

中断恢复仅将--fresh改成--resume。仅worker、prefetch、reader/derived cache等运行参数可以变更；源码、loss、seed、模型、采样或协议改变必须新run。

完整五折采用新run，改用--evaluation-mode fivefold --internal-mode oof。默认all不启动额外消融训练。参考未完成时比较记为未执行，正式晋级保持关闭。

## 性能复核

```powershell
conda run --no-capture-output -n bme-model python scripts/audit_statsfusion_v413_training.py --device cpu --workers 10
conda run --no-capture-output -n bme-model python scripts/benchmark_statsfusion_v413.py --real-audit --output .modeling/v413_real_benchmark.json
```

--real-audit使用可丢弃短审计的自然回放模型和两个真实session，不使用outer标签决定模型；模型权重SHA必须与来源CDF绑定的回放权重一致。CPU/GPU短审计保存到独立临时目录，并核验checkpoint SHA，防止后一轮审计覆盖前一轮replay输入。完整候选搜索可通过--evidence输入candidate_sets/windows/truth/ignore/mappings的本地可信Torch证据执行。Torch训练恢复和审计payload只能读取本项目可信文件；部署checkpoint保持weights_only加载并核验hash。

## 本次CPU/GPU验收证据

- 全套CPU pytest为758通过、40跳过，无失败；CUDA专项为72通过、19跳过，无失败。GPU专项包含新增device转发、CUDA bundle回放与连续/中断恢复检查。ruff、compileall和diff检查通过。

- 真实数据短审计使用10个worker；两chunk的event、quality、Boundary共享raw编码器梯度均非零，optimizer更新量0.0003003627。该权重明确为可丢弃审计权重。
- 真实短session为30秒，长session为35571秒并包含13022秒事件。256步和128步编码，以及17步增量提交的事件身份、边界、decision一致。长session的事件评分最大误差1.87e-9，时间特征最大误差4.53e-6；时间特征容差1e-5与事件评分容差1e-6分别登记。
- 长session审计的进程树private RAM峰值2.664GiB，历史表示不保留计算图。该结果不能替代GPU训练峰值验收。
- 候选头batch1与batch128最大误差CPU为5.97e-8、GPU为2.43e-8；同checkpoint重复推理误差0。GPU FP32/BF16 backward通过，event/quality/Boundary均能更新共享表示。
- GPU短训练审计使用10个worker，显存reserved峰值约1.428GiB，进程树private RAM峰值约12.43GiB；GPU长session回放reserved约0.244GiB。均低于10.5GB显存与26GB进程树RAM门禁。这里只证明这些验收负载，正式训练仍持续记录资源并执行超限门禁。
- CUDA连续执行与中断恢复checkpoint逐参数一致；Ctrl+C/OOM、日志尾部及部分梯度累计检查通过。OOM使用异常注入，未真实耗尽显存。真实长短session及无XGBoost bundle在CPU/CUDA均可回放。
- 完整16038点搜索与参考路径指标、选点一致。两真实短session且两条路径分别预热时间特征的CPU推理加搜索对照：6.704秒对56.640秒，约8.45倍；搜索约8.86倍。该样本只有36个进入候选头的几何，参考路径为432次调用。测量不含CDF生成和首次编码，不能推断252个session的完整Selector性能、正式训练耗时或GPU加速比。单个短审计session的自然回放CPU约1.69秒、GPU约1.22秒，全训练额外成本仍待测。
- 详细状态、源码hash和证据文件由 manifests/v413_implementation_acceptance.json 登记。实现验收完成；正式训练、单折/五折诊断和正式晋级尚未执行，promotable=false，不能据此声称F1提升。
