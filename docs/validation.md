# 验收与证据

## CPU检查

```powershell
python -m pytest -k "not cuda and not gpu"
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
git diff --check
python scripts/validate_statsfusion_joint.py --device cpu
```

覆盖模型梯度、fractional pooling、因果定位、区域隔离、一致增强、贡献归一化、自然回放RNG与释放、严格IoU、缓存身份、操作点、Ctrl+C/OOM恢复、邻近checkpoint、最终校准身份及无XGBoost bundle replay。

CPU单元测试不替代GPU、真实session资源和正式训练验收。pytest中的CUDA及真实数据资源检查单独启用，GPU任务须等待当前训练释放。

## 真实与GPU检查

```powershell
python scripts/validate_statsfusion_joint.py --device cuda --output validation_gpu.json
python scripts/audit_statsfusion_v414_training.py --device cuda --long-session
$env:BME_RESOURCE_TESTS = "1"
python -m pytest tests/test_resources.py
```

短训练审计使用独立临时目录，其权重不进入正式run；检查实际采样贡献、自然回放、head梯度、参数更新、clipping和资源。bundle生成后使用`validate_statsfusion_joint.py --bundle <run>/bundle --device cpu`执行锁定重放。

长事件验收支持13022秒历史；GPU图不随历史长度增长。batch1/128误差要求≤1e-6，不同chunk/未来前缀不得改变已提交事件。验证性能比较保持同权重、候选和16038操作点，不能缩减搜索宣称加速。

## 本次仓库整理

历史来源见`manifests/source_provenance.json`。重构前后使用独立CPU进程、同seed和同合成输入，核对state_dict、前向、ROI、Boundary、loss、梯度与AdamW更新。验收记录见`manifests/repository_cleanup_validation.json`；旧版报告仍可从其登记的Git提交中读取。

本次不启动正式训练，也不占用正在运行的v4.13 GPU任务。重构后source identity变化，须fresh新run。冻结划分和canonical特征recipe保持原始hash。

## 结果可信度

28/11开发结果参与epoch、校准和操作点选择；39人最终模型的11人校准集参与过训练。两者均属于开发证据。bootstrap和搜索稳定性报告支持数及不确定性，不能消除模型选择偏差。正式晋级需要独立外部证据或完整评估协议，当前bundle一直标记`promotable=false`。
