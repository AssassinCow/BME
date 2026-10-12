# v4.14 保存完整性与失败状态修复验收

日期：2026-10-12。当前源码 SHA256：`04528d318b80e07f78523872f00915440be37ae85c9a258ca74895e65ffaab81`。

## 实施范围

本次仅修改 v414 项目内的工作流校验、runner 失败状态处理、bundle 导出前检查及测试/说明。模型、训练器、loss、采样、学习率、seed、配置和冻结划分未修改。默认训练方案继续为 28/11 开发训练、39 人从头训练到 E+4；默认校准和部署仍使用 E。没有启动正式 v414 run。

v413 来源快照的 247 个文件重新核验全部一致，仍运行的 v413 训练未被中断或迁移。没有修改或覆盖既有验收报告。

## 完整性修复

- `final39_training_manifest.json` 的 `neighbor_checkpoints` 逐一登记 E−4/E−2/E/E+2/E+4 的路径、epoch、SHA256、完成状态、资格和 clipping。五档文件全部纳入 `evidence_sha256`，并随现有证据复制到 bundle 存档。
- 统一核验扩展轮数 E+4、完整 history、无 selector、39 人身份，以及五档权重的 finite、源/data/config 身份、预处理和准确 epoch。
- 默认 E 权重必须与保留的 E 档一致并通过原部署资格；邻近档允许作为诊断权重，不自动重选或降低资格门槛。
- 完成阶段 resume、直接导出及 bundle 加载均执行上述校验。缺失、损坏、错误 epoch/模型身份或不完整权重阻止放行。

## 失败状态修复

- 已绑定 run 的前置审计、完成阶段校验、执行及收尾错误纳入统一异常处理。已有合法 v414 run 在输入审计或初始化身份核验失败时，也记录 audit 失败；尚未建立的新 run 直接报告审计错误。
- Ctrl+C/GPU 忙为 PAUSED；OOM、文件/身份校验失败为 FAILED。记录准确阶段和原始异常，不将已完成状态留作失败结果。
- 报告保存发生次生错误时保留原始异常；可以写入的状态、失败报告及资源信息仍独立尝试保存。
- `final39_training_progress` 区分 E 权重是否已完整保存及合格、history 的最后记录轮数、目标 E+4，以及扩展训练是否经 manifest 验证。history 记录不替代恢复 checkpoint 的提交游标。
- E 已保存而扩展失败时保留 checkpoint，整体阶段仍未完成，按原 resume 协议继续。没有新增跳过扩展、绕过门禁或自动导出行为。

## 验证结果

| 检查 | 结果 |
|---|---|
| v414/v413、恢复与流水线 CPU 测试 | 81 passed，7 GPU/CUDA 条目排除 |
| v411/v412、旧 bundle 与 audit CPU 兼容回归 | 120 passed，13 skipped，11 GPU/CUDA 条目排除 |
| ruff，范围 src/scripts/tests | 通过 |
| compileall | 通过 |
| git diff --check | 通过；仅既有 LF/CRLF 提示 |
| CPU bundle replay，独立进程禁止 XGBoost | 通过 |
| 13023 秒合成长事件跨 chunk replay | 一个事件身份一致，最大延迟 15 秒 |
| 重复原始流式 replay | 最大分数误差 0，完整/分块/native stream 身份一致 |
| 模型/训练器/配置/划分的五文件基线 | hash 全部未变 |
| v413 来源快照 | 247 文件 hash 全部一致 |

新增测试覆盖文件删除/损坏、错误 epoch/模型/受试者、非 finite、未完成权重、扩展 metadata 失配、诊断邻近权重、bundle 契约、前置审计失败、完成阶段失败、Ctrl+C/OOM 和报告写入失败。

实际 trainer 循环的恢复测试使用合成两 session 数据及 39 人身份 fixture，从 E=8 训练到 12，分别在第 9 轮 Ctrl+C、第 11 轮 OOM 中断。恢复后的 E 权重、实际第 12 轮权重及 optimizer 状态与连续执行逐项零误差；history 完整；五档为 4/6/8/10/12；默认 latest.pt 仍为 8。此项验证恢复机制，不代表真实 39 人训练的效果或资源资格。

测试期间曾整理 runner 导入，触发一次源码身份校验。固定源码后重新完整执行，最终上述两组回归均无失败；没有放宽源码一致性检查。

机器可读记录：`manifests/v414_integrity_repairs_validation.json`。测试 XML：`manifests/v414_integrity_repairs_cpu.xml`、`manifests/v414_integrity_repairs_compatibility.xml`。当前源码 bundle replay：`manifests/v414_integrity_repairs_replay.json`。

## 验证边界

GPU 被 v413 训练占用，本次未并行执行 GPU 测试。正式 run 尚未创建；源码变化需从当前源码 fresh 建立身份，不能迁移旧 run 的恢复身份。现有 GPU 空闲检查与开发训练前预检继续生效。

这些修复不消除最终模型在已见 11 人上校准的乐观偏差，不扩大缺失域/短事件支持，也不保证 F1 提升。所有结果仍沿用 development_on_final_model、independent_calibration=false、promotable=false 的证据边界。
