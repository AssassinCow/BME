# StatsFusion 统计特征溯源

当前 `statsfusion-r3.2` 固定使用 12 项 trailing 15 秒统计特征，顺序如下：

1. `local_acc_y_mean`
2. `local_acc_z_zero_crossing_rate`
3. `local_gyro_z_mad`
4. `local_acc_mag_std`
5. `local_gyro_z_zero_crossing_rate`
6. `local_acc_mag_iqr`
7. `local_acc_mag_jerk_rms`
8. `local_gyro_y_mad`
9. `local_acc_y_median`
10. `local_ppg_valid_fraction`
11. `local_gyro_mag_jerk_p95`
12. `local_acc_x_median`

这些特征来自早期 baseline 五折 gain 诊断。历史诊断显示 ACC 约占归一化 gain 的
`48%–52%`、GYRO 约 `36%–38%`、PPG 约 `10%–14%`；该结果只用于固定候选特征集合，
不作为当前模型的独立外层验证证据。

原始五折 metadata 的 SHA-256 已冻结如下，清理后的正式训练不再把这些历史输出作为运行时输入：

- fold 0: `d4b2398127bf42a348f037ea8aa8d9c77e6fbe955c6bf6c41fb281ab5be0c791`
- fold 1: `5d77d632d1b6a2c247690a8057af175220f9d83b6875a99ae6ade94d94e65cb9`
- fold 2: `a5d3853c1f3b8d60d90c96bd1148f2881167084c38fc0940136e3baf1db8be95`
- fold 3: `0a7c1bcbc7c9398028994f24ad32b1082ca88ea68cada5c1ed9c25f50610f7df`
- fold 4: `23d54589ec1d040217dd17108c2e43d8fbed9e56c9afb18882bcef69232fd9d3`

当前 `feature_provenance.source_paths` 为空，审计哈希锁定本文件，从而保留特征名称、顺序、
历史证据哈希和证据边界，又不要求部署或重训环境携带旧 baseline 产物。由于选择过程查看过
全部开发折，相关结果统一标记为 `development_stress_only`；独立证据只能来自官方隐藏测试或
新增受试者。
