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
python scripts/audit_statsfusion_v414_training.py --device cuda --long-session --epochs 8
python scripts/audit_statsfusion_v414_replay.py --device cuda --benchmark-search
$env:BME_RESOURCE_TESTS = "1"
python -m pytest tests/test_resources.py
```

短训练审计使用5个真实session、8个可丢弃epoch，覆盖正例、背景、自然GYRO缺失和长事件。检查每epoch计数表与实际端点mask一致、自然回放、完整增强调度、原子梯度保留、历史背景对比、head梯度、参数更新、clipping和资源。权重不进入正式run，短审计F1不作为效果证据。bundle生成后使用`validate_statsfusion_joint.py --bundle <run>/bundle --device cpu`执行锁定重放。

v414-r1回归额外覆盖含峰session比例与受试者权重变化、无监督项关闭、A/B重叠组的独有0.286版本、纯负排序梯度、几何锁定、稳定物理排序、恒定0.4的不可用起点峰、EOF先后顺序及空CDF验证中断计数。全部检查通过后方可启动修复后的全新run，历史run及v4.13源目录保持原样。

v414-r2进一步覆盖三尺度空窗口、缺失邻居/断档峰确认、混合组逐版本标签、失去唯一正版本的组退出、过滤后类别重算、CDF fallback两种模式的预算去重、晚到重复释放与恢复、历史reference上限和背景保留。验收记录采用`manifests/v414_r2_*`命名，保留r1证据。

v414-r3新增实际流式候选覆盖反例、双峰配额、重复起点补位和前缀/身份一致性；对无资格截止及合格早停分别在终止提交后注入Ctrl+C/OOM，核对一次/两轮验证频率下不追加epoch，模型及optimizer逐项一致，失败报告恢复一致。r3证据使用`manifests/v414_r3_*`，历史r1/r2报告保留，其GPU和真实训练审计不自动视为当前源码验收。

真实重放审计复用本轮可丢弃权重，比较长短session在128/256新步下的事件及分数。可选完整16038点对照仅衡量真实审计子集上的搜索阶段，不代表完整验证或正式数据上的加速倍数；固定校准器只用于路径等价检查。

长事件验收支持13022秒历史；GPU图不随历史长度增长。batch1/128误差要求≤1e-6，不同chunk/未来前缀不得改变已提交事件。验证性能比较保持同权重、候选和16038操作点，不能缩减搜索宣称加速。

## 本次仓库整理

历史来源见`manifests/source_provenance.json`。重构前后使用独立CPU进程、同seed和同合成输入，核对state_dict、前向、ROI、Boundary、loss、梯度与AdamW更新。验收记录见`manifests/repository_cleanup_validation.json`；旧版报告仍可从其登记的Git提交中读取。

本次不启动正式训练，也不占用正在运行的v4.13 GPU任务。重构后source identity变化，须fresh新run。冻结划分和canonical特征recipe保持原始hash。

## 结果可信度

28/11开发结果参与epoch、校准和操作点选择；39人最终模型的11人校准集参与过训练。两者均属于开发证据。bootstrap和搜索稳定性报告支持数及不确定性，不能消除模型选择偏差。正式晋级需要独立外部证据或完整评估协议，当前bundle一直标记`promotable=false`。
