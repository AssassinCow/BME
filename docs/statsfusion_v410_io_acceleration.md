# 当前 run 的 I/O 加速与恢复

适用 run：`statsfusion_v410_joint_event_20261007a`。用户已暂停在第八轮、已提交 1481 个 chunk，并授权启用源码级加速。模型、损失、学习率、AMP、chunk 长度、四个 session 的梯度累计、样本顺序、随机增强和验证协议保持一致。

| 参数 | 原值 | 当前配置 |
| --- | ---: | ---: |
| `training.num_workers` | 10 | 10 |
| `training.loader_prefetch_factor` | 1 | 4 |
| `training.session_reader_cache_size` | 1 | 4 |

预取队列上限由 10 个 batch 增至 40 个，配合 pinned memory 和非阻塞 CUDA 输入拷贝。每个 worker 缓存四个原始 archive，减少跨 archive 窗口的重复读取。DataLoader 仍按 sampler 顺序交付，增强仍由 seed、epoch、subject 和 session 决定，缓存只保存原始数组。
同一训练分区复用读取 worker，避免 Windows 每轮重新启动十个 Python 进程；epoch 仍重新生成原有 subject/session/chunk 顺序，更换分区必须新建 dataset/loader。模型历史特征缓存仍按 epoch、subject/session 和 optimizer update 隔离。

本机为 i9-14900HX、32 GB RAM、RTX 4080 Laptop 12 GB VRAM。检查时训练正常推进，GPU 利用率单次快照约 7%、已用显存约 2.5 GB、可用物理内存约 8.7 GB。快照只能说明当时资源状态，不能作为完整性能剖析。2125 个原始 archive 的解压后中位大小约 7.76 MiB、P95 约 18.68 MiB、最大约 63.90 MiB；十个 worker 各增加三个缓存的数组容量上限估计约 1.87 GiB，另有预取 batch、pinned memory 和 Python 管理开销。

训练配置身份仍为 `ed8ca128a5cd7d301a936bba3bbefcd6e23933504c71af04712f4fa04debc235`。源码变化通过 `configs/statsfusion_v410_performance_resume_migration.json` 精确登记，包含监督目标批量生成、finite 检查合并、读取 worker 复用和 checkpoint 传输/序列化优化；模型文件不变。原始 identity、所有前序迁移 receipt、旧 checkpoint 和累计梯度保持原样。

## 数值与性能

3/15/60 秒监督目标取消逐端点 CUDA 标量同步。二值 target/mask 窗口采用精确整数计数和原浮点除法；非二值窗口继续使用原来的逐窗口浮点规约，因此 soft label 或有效 mask 不改变舍入顺序。缺失窗口、时间对齐与有效 bin 语义保持一致。

finite 检查仍覆盖全部权重与累计梯度，但按 device 合并判定，减少多次 CPU/GPU 同步。checkpoint 的模型、optimizer 和梯度按 dtype/device 汇集拷至 CPU；CPU 历史状态先使用原生 pickle 封装，减少 Torch serializer 对大量历史诊断字段的逐对象处理。保存仍逐 chunk 同步、fsync 并原子 replace；没有降低保存频率或异步跳过提交。旧格式与新格式的 `torch.load(..., weights_only=False)` 都返回原来的 state 字典。

实测结果：

| 实测范围 | 原实现 | 新实现 | 局部提速 |
| --- | ---: | ---: | ---: |
| 实际输入、1137 步共享编码器/state/proposal 前后向 | 0.4423 s | 0.0870 s | 5.08 倍 |
| 实际 checkpoint 的 GPU 拷贝与序列化 | 1.0617 s | 0.2714 s | 3.91 倍 |

第一项逐参数梯度与 loss 完全一致，第二项恢复后的 state、model、optimizer、累计梯度和 RNG 完全一致。第一项峰值 GPU allocated/reserved 为约 1.29 GiB；完整 forward/backward 与原始 session 训练另执行资源测试，门槛为 10.5 GiB。上述局部测试不等于整轮训练或完整五折提速，checkpoint 测试不含磁盘 fsync；候选处理、calibration/search 和验证仍有耗时。

## 生效与回退

普通 `--resume` 自动验证并提交登记的性能迁移，无需额外 flag；先重新运行实现验证，再从第八轮第 1482 个 chunk 继续。已有完整 chunk、累计梯度、RNG 和 session/cache 游标继续保留；未提交 chunk 重新执行。备份及迁移证据写入 `resume_migrations/numerically_equivalent_performance_v1/` 和 `performance_resume_migration.json`。

```powershell
python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v410_joint_event.yaml `
  --run-name statsfusion_v410_joint_event_20261007a `
  --stage all --skip-s0 `
  --historical-reference hierarchical_v4_r32_deep_frontier_20260930a `
  --resume
```

若实际内存压力或稳定性变差，将 prefetch 和 reader-cache 降为 2，必要时改回 1 后恢复；worker 保持 10。这两个字段允许恢复，不影响模型配置身份。源码优化需要保留登记身份，不自行回退源码后继续该 run。

本方案不承诺整轮固定提速倍率。没有启用 TF32、改变 dropout 随机序列的 compile/CUDA Graph、扩大监督 chunk、改变梯度累计或降低 checkpoint 频率；这些无法满足当前 run 的数值与恢复不变性要求。

## 验收

- 原始 archive 读取结果逐元素相同；跨两个 archive 的重复读取由八次解压减少至两次。
- 真实 JointSessionDataset 的旋转、模态 dropout、missing mask、标签和时间轴在 prefetch/cache 更改后逐 tensor 相同；从中途 chunk 恢复也保持同一 sampler identity 和 batch 内容。
- CPU/CUDA 的二值/soft/missing 监督目标、累计梯度和三轮训练后权重逐 tensor 相同。
- CPU/CUDA Ctrl+C/OOM 恢复测试保留部分累计梯度，最终权重与不中断的参考训练逐 tensor 相同。
- persistent loader 的 worker PID 保持不变，epoch RNG/顺序与新建 loader 一致，并拒绝跨 dataset/训练分区复用。
- 实际 checkpoint 的紧凑状态封装恢复后重新 pickle 的 bytes 与原状态完全相同，optimizer、梯度和 RNG 逐 tensor 相同。
- 当前 checkpoint SHA-256 固定为 `62565f6361310230db53893f75c6833f418166ed1b000f45ad2ae001aac5cdd9`，性能迁移只接受该已暂停断点及登记源码；正式验收包括全套 pytest、ruff、compileall 和 GPU/十 worker 资源测试。

实际验收：启用 `BME_V410_RESOURCE_TESTS=1` 后，全套测试为 599 passed、1 skipped，耗时 210.22 秒；ruff、compileall 和 diff 检查通过。普通 `--stage audit --resume` 已提交性能迁移，`--stage validate --resume` 已重新验证当前实现；当前 run 保持 `PAUSED`，outer fold 0 未继续执行。

另通过真实 `train_joint_model` 恢复入口，在第一个 chunk 之前停止核验：全部模型权重、两个 optimizer 参数组、353 个非空累计梯度、Python/NumPy/CPU/CUDA RNG、session/cache/history 和恢复游标与原断点一致。原 checkpoint、原 identity 和两层前序迁移 receipt 的 SHA-256 均未改变。测试与性能证据保存在 run 的 `performance_validation.json`，实际恢复入口证据保存在 `real_resume_validation.json`；其范围不包含新一轮完整模型训练或晋级成绩。
