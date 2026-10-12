# 原始数据到训练输入

仓库提供`原始下载材料 → v2索引和session缓存 → v4 canonical特征`的完整入口。仅使用现有下载材料，不下载数据。唯一YAML仍为`configs/statsfusion_v414_full_data.yaml`；入口从其中读取环境变量名称，另外构造冻结的v2预处理配方，不把v4.14训练配置当作旧预处理配置使用。

## 必需材料

`BME_DATA_ROOT`包含：

```text
manifests/download_state.json
derived/valid_metadata/sensor.csv
derived/valid_metadata/meal.csv
raw/sensor_attachments/.../*.zip
```

下载manifest中的`relative_path`、`sha256`和`status=downloaded`必须对应本地ZIP。使用已核验的metadata；数据申请、下载和生成valid_metadata不在此入口范围内。

复现冻结受试者身份必须沿用原来的私有salt：设置原`BME_SUBJECT_SALT`，或保留`BME_OUTPUT_ROOT/private/subject_salt.hex`。将输出移到新目录时，应私下复制该文件到新目录的同一位置。环境变量形式按UTF-8字符串使用，文件形式按hex解码使用，两者不能直接互换。salt不能提交Git；缺少原salt时入口停止，不自动生成新身份。

## 生成命令

在仓库根目录运行，`BME_OUTPUT_ROOT`指向一个新的输出目录，或尚未创建v2/v4输入的目录：

```powershell
conda run --no-capture-output -n bme-model python scripts/prepare_statsfusion_inputs.py `
  --config configs/statsfusion_v414_full_data.yaml --stage all --workers 8 --fresh
```

- `--stage v2`：索引、全量格式审计、多section审计、预处理、事件覆盖、原五折划分及质量门禁。
- `--stage canonical`：只读使用已有合格v2，生成v4 canonical输入。
- `--stage all`：按上述顺序完成两步。
- 中断后使用相同stage和输出目录，将`--fresh`替换为`--resume`。允许调整worker；原始数据、salt或预处理配方/源码变化会拒绝v2恢复。

格式审计使用流式字节扫描检查整个TXT成员的标准表头；即使第二个表头出现在第10万行之后，也会进入多section审计。`maximum_rows=100_000`仅限制PPG槽统计抽样，不限制表头检测。审计报告分别记录`header_scan_scope=full_member`和`ppg_sample_maximum_rows`；旧版截断扫描checkpoint不再复用。

`--fresh`不会覆盖已有输入；旧入口生成的v2没有新的恢复manifest，不能用`--stage v2 --resume`接管。已有合格v2缺少canonical时，使用`--stage canonical --fresh`。

若准备任务在旧版截断扫描上失败，更新源码后会因源码身份变化拒绝`--resume`。请保留旧目录，在新的`BME_OUTPUT_ROOT`中沿用原私有salt并执行上述`--fresh`命令，重新生成v2及canonical；不要删除、覆盖或迁移正在被训练使用的输入。修复不修改冻结的质量计数：完整审计应识别79个重复表头附件，由多section审计区分45个完全重复、17个冲突和17个非单调附件。

v2配方与整理前一致：ACC/GYRO 100 Hz、PPG 50 Hz、20个有效PPG槽、3秒网格和session连接间隔、5折及seed 2026。排除三名官方作废受试者；45个已审计的完全重复多section附件去重，34个冲突或非单调附件隔离。隔离数量和分类必须通过原质量规则，不能把全部异常视为可恢复数据。

v2的`preparation_manifest.json`记录原始材料hash、配方、源码、私有salt摘要、状态和产物hash；质量报告通过预注册计数和结构门禁后冻结`quality_expectations.json`。canonical继续使用原`statsfusion-r3.2`配方及原有身份/恢复校验。

## 与训练的衔接

生成结果位于`BME_OUTPUT_ROOT/v2`及`BME_OUTPUT_ROOT/v4/canonical_input_r3_2`，随后执行README中的训练命令。训练仍验证`splits/v414_dev28_11.json`及其数据身份；准备入口不会修改该划分。

保持salt能复现受试者键，但路径、原始材料或依赖版本不同仍可能改变数据hash。若冻结划分的数据身份校验失败，应检查差异并明确冻结新数据/划分、使用新run，不能跳过训练审计或自动替换旧身份。
