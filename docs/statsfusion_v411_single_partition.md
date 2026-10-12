# v4.11 fold0 单 partition 接续

时间受限时使用 `configs/statsfusion_v411_fold0_single_partition.yaml`，或在新 run 中设置 `--evaluation-mode fold0 --internal-mode single`。完整三分区模式仍为 `--internal-mode oof`，默认配置不变。单分区只允许 v4.11 fold0 诊断，不能进入五折正式晋级。

## 保留内容与执行流程

分区仍使用原三等分 assignment（seed=2026），仅执行 partition0，并非把 partitions 改成1、取消留出或把同一批受试者同时用于训练和校准。fold0共有31名outer-train和8名outer-test受试者。partition0保留20人训练、11人独立预测/校准；20人内部原13名fit、7名selector不变。

当前 run `statsfusion_v411_fold0_continue64_20261009a` 的selector已在第44轮早停，选定第38轮。partition0的20人重训按原38轮执行，从第21轮已提交chunk继续，随后只预测未进入该模型的11人，拟合自然候选校准并锁定候选、NMS、接受阈值和Boundary模式。partition1和partition2不再训练。

最后仍从头训练fold0全部31人，训练轮数由partition0确定为38轮，使用此前11人独立预测锁定的操作点，测试8名outer-test受试者并保存 `fold0_final` / `fold0_bundle`。模型架构、loss、学习率、batch/chunk、梯度累计、clipping、早停与单个模型的训练数学逻辑不变。不能将20人checkpoint直接接到31人训练，以免改变从头重训协议。

单组校准预测只覆盖11人，搜索和Boundary评估仅读取这11人的truth/ignore，不能把未生成预测的另20人计为FN。其声明为 `calibration_kind=single_partition_holdout`、`full_internal_oof=false`；不能宣称完整内部OOF，也不能用fold0测试标签补充校准或边界资格。60个独立配对正事件的Boundary门禁不放宽，样本不足时保持关闭。

## 已有 run 身份迁移

原run的配置/source身份被冻结，因此不能仅改配置后直接resume。专用迁移入口仅授权上述执行范围变化，校验训练函数AST不变、data/runtime及其余配置不变，限制修改文件清单，并要求run处于PAUSED且尚未完成校准或outer训练。

原身份、配置、源码快照与所有partition0 checkpoint保存于 `resume_migrations/v411_single_partition_holdout_v1`。`single_partition_migration.json`记录变更、备份hash及checkpoint状态hash。模型参数、optimizer、累计梯度、RNG、session/cache/journal游标以及原model identity保持一致，只重绑定config/source身份。原run identity hash作为模型祖先身份保留。迁移的部分写入可用同一脚本恢复，普通resume拒绝未提交完整的迁移。

迁移命令（同一run执行一次，重复执行只验证已完成迁移）：

```powershell
conda run --no-capture-output -n bme-model python scripts/migrate_statsfusion_single_partition.py `
  --config configs/statsfusion_v411_fold0_single_partition.yaml `
  --run-name statsfusion_v411_fold0_continue64_20261009a
```

继续训练：

```powershell
conda run --no-capture-output -n bme-model python scripts/run_statsfusion_joint.py `
  --config configs/statsfusion_v411_fold0_single_partition.yaml `
  --run-name statsfusion_v411_fold0_continue64_20261009a `
  --stage all --evaluation-mode fold0 --internal-mode single --skip-s0 `
  --reference-run statsfusion_v410_joint_event_20261007a --resume
```

单组会减少校准证据和稳定性，省去另外两个partition的选模和重训，但仍保留当前20人重训以及后续31人outer训练的必要计算。产物只代表单折、单校准留出诊断；F1提升不作保证。

## 终端进度条

训练和验证使用纯白Unicode条形，同一进度条以回车在原行刷新，宽度随终端调整，并预留一个字符避免终端自动换行。日志包装保留原终端的编码及文件描述符，使宽度检测有效。每个训练轮次或验证步骤结束才另起一行，指标仍单独打印。

`statsfusion_v411_fold0_continue64_20261009a` 的显示修复通过 `.modeling/migrate_white_progress.py` 做专项身份迁移，保留原checkpoint、optimizer、累计梯度、RNG和恢复游标，记录 `white_progress_migration.json` 及独立备份。该迁移只允许 `LogMirror` 与 `joint_progress.py` 显示代码变化，普通resume仍拒绝未授权源码变化。

在PowerShell中直接执行上面的python命令，或使用 `conda run --no-capture-output`；避免将输出经过 `Tee-Object`、`Out-File` 或其它逐行捕获器再显示，这类输出管道可能无法呈现终端原行刷新。
