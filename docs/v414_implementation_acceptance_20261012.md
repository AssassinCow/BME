# v4.14 实施验收记录

日期：2026-10-12。状态：实现完成，CPU检查通过；GPU验收、正式开发训练与39人重训尚未完成。没有启动 `statsfusion_v414_28_11_then_39_20261012a`，没有生成正式模型或部署bundle。

## 源码与数据身份

- 来源HEAD：`e0ec880f25412d0683261068c3dc7955b0f6d9d3`。
- v4.13来源文件：247项；聚合SHA256：`b9aef5f11a600b07e97c7a71e49477e5d9df42ccd4f1affa3d257500ad86612a`。交付前后核验一致；本次未写入v4.13目录或训练产物。正在训练的产物可能由原训练进程正常更新，不声称这些活动文件的hash静止不变。
- v4.14实际运行源码SHA256：`ec2c327a70eed7afa4c1e80f903e74cc35b60d2c0d658e97de8274dc8b61ddea`。
- 展开配置SHA256：`f02046a3638c103e5aa3a9b210a430b308a26dc8a02614f3b52f3bfb133adb4b`。
- 数据身份SHA256：`59942597374a142ae2e8c6d9acc63c4e920c5252e7831f73fb6220cd041810fe`。
- 冻结划分SHA256：`d5c3b2e6faa67bad587e57018dfcb24ddf070b626312a3bd5a61030cfd162e0f`。
- Git对象库独立，没有alternates；已提交、staged、unstaged与相关untracked修复均有来源记录。新runner拒绝旧版配置，避免进入旧输出命名空间。

| 集合 | 人数 | Session | 有效事件 | 同手/异手 | GYRO全缺失事件 |
|---|---:|---:|---:|---:|---:|
| 开发训练 | 28 | 799 | 116 | 52/64 | 6 |
| 开发验证 | 11 | 354 | 45 | 18/27 | 4 |
| 最终训练 | 39 | 1153 | 161 | 70/91 | 10 |

session数相对约801/352目标偏差为−2/+2；约束和成员已经冻结。原fold0八人进入训练，两名自然GYRO全缺失受试者各进入一组。

## 已完成检查

| 检查 | 结果及范围 |
|---|---|
| 现有CPU回归 | 773通过、19跳过、33取消选择；在最后的CLI、lineage与runtime防护补齐之前执行 |
| 最终版本专项回归 | 40通过，0失败、0跳过；`manifests/v414_final_targeted_cpu.xml` |
| ruff / compileall / diff检查 | 通过；Git仅提示既有CRLF转换信息 |
| 模型兼容数学 | v4.14与v4.13的初始化参数及主要输出完全一致 |
| CPU前后向 | 事件与质量梯度连接共享表示；重复推理误差0 |
| batch1/128 | 候选头最大误差`4.0978193283081055e-08`，低于`1e-6` |
| 长事件 | 13023秒，17/256/4362步chunk下事件身份一致，最大延迟15秒 |
| 原始流式输入 | 增量时间戳、事件身份与公共字段一致；特征最大误差`2.6226043701171875e-06`，事件评分误差0 |
| 隔离bundle replay | 合成fixture通过锁定操作点推理、重复/增量一致性及独立无XGBoost加载和replay；不是正式训练bundle |
| 最新CLI输入审计 | `v414_implementation_audit_20261012b`的audit完成，整体PAUSED；未进入训练 |
| 训练身份与恢复 | 测试覆盖28/11排除、39人固定epoch、旧checkpoint拒绝、完成阶段hash核验、final39别名不改变冻结配置，以及继承的恢复路径 |

对应报告：`manifests/v414_cpu_validation.json`、`manifests/v414_bundle_replay_validation.json`、`manifests/v414_delivery_verification.json`。

## 真实生产循环审计及限制

另用冻结28人训练池中的5个真实session、4名受试者，在CPU上从头训练3轮。审计包含自然GYRO缺失及最长39375秒session，每轮58个chunk，权重不进入正式run。

第三轮实际正组4、负组296，最大困难负例预评分调用11，低于96上限。事件、质量和Boundary头有实际梯度和参数更新，包含最后不足4个session的更新；进程树private RAM峰值约5.56 GiB。

**该审计三轮的clipping fraction均为0.50，每轮仅2个optimizer step。**这只确认早期小样本生产路径可运行，不证明正式checkpoint达到`≤0.20`资格，也不证明训练稳定性或效果提升。epoch≥6的失败规则和epoch≥8的选择资格保持原协议，未放宽。正式训练必须继续观察这些门禁。

审计报告是`manifests/v414_production_loop_cpu.json`，绑定当时源码`e607fa62df0f5cf1604f1fffd59facd4aaefa844ee3e48b712127e6bde45d30e`；随后补齐防护后的最终源码身份另由40项专项测试和CPU模型验证记录。没有把早期审计hash改写成最新hash。

## 尚未完成

v4.13训练进程仍占用GPU，因此本次未运行并行GPU测试。GPU资源门禁、CUDA/BF16前后向及GPU恢复需要等待该进程释放资源；`dev`入口在启动正式训练前会执行同身份模型预检，GPU忙时保存PAUSED。

28人开发训练、39人最终重训、真实训练bundle replay、官方测试及效果比较均未运行。没有F1提升承诺，也没有宣称正式晋级通过。

Boundary部署强制off，完整16038操作点保留。39人模型在原11人上的校准登记为`development_on_final_model`，标签已经参与训练；bootstrap是锁定操作点的受试者重采样，不能校正选模或搜索乐观偏差。最终产物仍为`official_test_candidate=true`、`promotable=false`。

正式运行与分阶段继续命令见[运行说明](statsfusion_v414_full_data.md)。中断后仅使用同配置、同run的`--resume`；不得通过resume改变成员、模型、loss、seed、数据、源码或协议。
