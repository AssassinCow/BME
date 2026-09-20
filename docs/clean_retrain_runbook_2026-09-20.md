# 全量 clean-room 重训手册

日期：2026-09-20。该流程取代“复用冻结 baseline”的快捷路线，用当前干净 Git 提交从原始
下载数据开始重新生成全部派生产物。

## 隔离边界

- 新 `BME_OUTPUT_ROOT` 在启动时必须不存在；脚本不会删除、覆盖或读取旧审计、checkpoint、
  indices、NPZ、特征、模型、预测或评估文件。
- 原始下载数据仍只读使用现有 `BME_DATA_ROOT`，不重新下载，也不复制到输出目录。
- 唯一复用的旧文件是私有 `subject_salt.hex`。这样匿名受试者键和五折划分仍可与历史结果
  公平比较；salt 内容不会打印、写入 Git 或进入 manifest。
- 若改用新 salt，得到的是一套新的匿名键和可能不同的五折划分，不能与旧 baseline 做严格
  配对比较。

## 单一启动命令

先提交本次代码并在 4080 上拉取，确认 `git status --short` 没有输出，然后在 `bme-model`
环境和仓库根目录执行：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
powershell -ExecutionPolicy Bypass -File scripts/run_clean_retrain.ps1 `
  -OutputRoot "D:\BME2026\outputs_clean_$stamp" `
  -SubjectSaltFile "D:\BME2026\outputs\private\subject_salt.hex"
```

脚本依次执行：环境检查、完整测试、Ruff、CUDA smoke、1112 个附件全量 schema 审计、重复
表头审计、全量预处理、质量报告、质量快照冻结、baseline 特征、五折 baseline、baseline
汇总、五折 DTP 残差融合和最终晋级比较。

重复表头审计因 34 个应隔离附件返回退出码 1 是当前已知状态；脚本仅在完整审计 JSON 已写出
时接受该退出码，随后由预处理和质量门禁验证精确分类与数量。

## 人工质量门禁

脚本打印匿名质量报告后暂停。确认附件数、恢复/隔离数、受试者数、segment/session、事件、
coverage、PPG 槽位和折分摘要合理后，输入大写 `FREEZE`。其他输入都会停止，baseline 不会
开始训练。

## baseline 与 fusion 证据链

- baseline 五折均使用 `--no-resume`，且输出根目录此前不存在。
- 每折 manifest 必须记录相同的当前 Git commit、clean working tree、配置哈希和数据/折分
  指纹。
- fusion fold 0 启动前会一次性验证五折 baseline 完整且满足上述条件；不会接受旧回填
  manifest 或声明为 `3ca55bb` 的历史 baseline。
- fusion 任一门禁失败即停止，clean baseline 仍是完整保底结果。
- fold 0 外层门禁仍造成条件筛选偏差，五折 fusion 汇总只能称为经过门禁的本地结果。

只重跑数据和 baseline、暂不训练 fusion 时加 `-SkipFusion`。若流程中断，不要删除目录；保留
现场诊断。需要断点续训时按对应模块的显式 resume 命令处理，而不是再次以同一目录启动
clean-room 脚本。

官方 partial 计分、最终一对一匹配、提交字段和测试接口仍为 `UNKNOWN`，所有输出均不是
官方分数。
