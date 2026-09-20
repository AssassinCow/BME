# 限时核心优化实施状态（2026-09-20）

## 已实现

- 冻结 `baseline`，新增 `baseline_fastslow`、`baseline_boundary`、
  `baseline_dyadic_v2` 配置和独立实验目录。
- `hysteresis_v1` 保持默认；新增快慢双 EMA、2/3 快通道启动、慢通道启动、
  双通道持续退出和 session 强制重置。
- 双阶段 OOF 后处理搜索、前 20 组筛选、三折均值/最差折/异侧召回/边界 MAE
  选择以及所有搜索参数触边门禁。
- start/end XGBoost 回归头、删失 mask、20 倍正边界权重、20 倍随机负样本和
  10 倍困难负样本；缺少学习边界时回退概率差分。
- 异侧权重 `[1.0, 1.5, 2.0]` 的三折 OOF 选择和正类总权重归一；
  `hand_relation` 保持为非特征元数据。
- `calibration_fold` 私有 OOF 字段、无敏感路径的 `run_manifest.json`、失败峰值诊断、
  有符号边界误差、完整晋级门禁和 2000 次受试者级配对 bootstrap。
- 指数桶 fold-0 外层训练 OOF 筛选，以及不访问外层测试结果的 `--oof-only` 入口。
- 旧版全量指数桶未通过 OOF/外层门禁后，新增一次性的
  `baseline_dyadic_lite`：只保留 3-96 秒运动桶和非冗余统计，取消 192 秒运动桶与
  PPG 历史桶；未通过预注册 OOF 门禁即永久停止该路线。

## 运行边界

- 未新增 TCN，未修改或运行 DTP-SQF。
- 本机不做正式训练；全量训练、CUDA XGBoost、五折和消融仅在 RTX 4080 的
  `bme-model` 环境验收。
- 所有候选保持因果，未来上下文关闭。
- 官方 partial 计分、匹配细节、提交字段和测试接口仍为 `UNKNOWN`；本地产物不得
  称为官方分数。
