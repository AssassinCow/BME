from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReportThresholds:
    good_score: float = 0.60
    warning_score: float = 0.30
    good_mae_seconds: float = 60.0
    warning_mae_seconds: float = 180.0
    good_fp_per_hour: float = 0.10
    warning_fp_per_hour: float = 0.50


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _optional_int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if math.isfinite(number) else None


def _relation_metrics(payload: dict[str, Any], relation: str) -> tuple[int, int, float]:
    relation_payload = payload.get("hand_relation", {}).get(relation, {})
    truth = int(relation_payload.get("truth_events") or 0)
    matched = int(relation_payload.get("matched_events") or 0)
    sensitivity = matched / truth if truth else float("nan")
    return truth, matched, sensitivity


def _fold_status(
    metrics: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    postprocess: dict[str, Any] | None,
) -> str:
    if metrics is not None:
        return "complete"
    if postprocess and bool(postprocess.get("threshold_at_search_boundary")):
        return "postprocess_boundary_blocked"
    if postprocess is not None:
        return "postprocess_complete_evaluation_missing"
    if metadata is not None:
        return "model_complete_postprocess_missing"
    return "missing"


def collect_experiment_results(
    experiment_name: str,
    experiment_root: Path,
    expected_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for fold in range(expected_folds):
        fold_dir = experiment_root / f"fold_{fold}"
        metrics_path = fold_dir / "test_metrics.json"
        metadata_path = fold_dir / "metadata.json"
        postprocess_path = fold_dir / "selected_postprocess.json"
        metrics = _read_json(metrics_path)
        metadata = _read_json(metadata_path)
        postprocess = _read_json(postprocess_path)
        status = _fold_status(metrics, metadata, postprocess)
        row: dict[str, Any] = {
            "experiment": experiment_name,
            "fold": fold,
            "status": status,
            "validation_auprc": _number((metadata or {}).get("validation_auprc")),
            "final_estimators": _optional_int((metadata or {}).get("final_estimators")),
            "train_rows": _optional_int((metadata or {}).get("train_rows")),
            "validation_rows": _optional_int((metadata or {}).get("validation_rows")),
            "test_rows": _optional_int((metadata or {}).get("test_rows")),
            "high_threshold": _number((postprocess or {}).get("high_threshold")),
            "low_threshold": _number((postprocess or {}).get("low_threshold")),
            "threshold_at_search_boundary": bool(
                (postprocess or {}).get("threshold_at_search_boundary", False)
            ),
            "metrics_path": str(metrics_path),
        }
        if metrics is not None:
            method = str(metrics.get("primary_method", "max_cardinality_iou"))
            primary = metrics.get(method, {})
            counts = metrics.get("evaluation_counts", {})
            same_truth, same_hits, same_sensitivity = _relation_metrics(metrics, "same")
            different_truth, different_hits, different_sensitivity = _relation_metrics(
                metrics, "different"
            )
            row.update(
                {
                    "primary_method": method,
                    "true_positive": _number(primary.get("true_positive")),
                    "false_positive": _number(primary.get("false_positive")),
                    "false_negative": _number(primary.get("false_negative")),
                    "ignored_predictions": _number(primary.get("ignored_predictions")),
                    "precision": _number(primary.get("precision")),
                    "sensitivity": _number(primary.get("sensitivity")),
                    "f1": _number(primary.get("f1")),
                    "start_mae_seconds": _number(primary.get("start_mae_seconds")),
                    "end_mae_seconds": _number(primary.get("end_mae_seconds")),
                    "false_positives_per_observed_hour": _number(
                        primary.get("false_positives_per_observed_hour")
                    ),
                    "evaluable_truth": _number(counts.get("evaluable_truth")),
                    "ignored_truth": _number(counts.get("ignored_truth")),
                    "predicted_events": _number(counts.get("predicted_events")),
                    "same_truth_events": same_truth,
                    "same_matched_events": same_hits,
                    "same_sensitivity": same_sensitivity,
                    "different_truth_events": different_truth,
                    "different_matched_events": different_hits,
                    "different_sensitivity": different_sensitivity,
                }
            )
            for relation in ("same", "different", "unknown"):
                relation_payload = metrics.get("hand_relation", {}).get(relation, {})
                relation_rows.append(
                    {
                        "experiment": experiment_name,
                        "fold": fold,
                        "hand_relation": relation,
                        "truth_events": relation_payload.get("truth_events", 0),
                        "matched_events": relation_payload.get("matched_events", 0),
                        "sensitivity": relation_payload.get("sensitivity"),
                        "start_mae_seconds": relation_payload.get("start_mae_seconds"),
                        "end_mae_seconds": relation_payload.get("end_mae_seconds"),
                    }
                )
            for subject_key, subject_metrics in metrics.get("by_subject", {}).items():
                subject_rows.append(
                    {
                        "experiment": experiment_name,
                        "fold": fold,
                        "subject_key": subject_key,
                        **subject_metrics,
                    }
                )
            sources.append(
                {
                    "experiment": experiment_name,
                    "fold": str(fold),
                    "artifact": str(metrics_path),
                }
            )
        fold_rows.append(row)
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(subject_rows),
        pd.DataFrame(relation_rows),
        sources,
    )


def _weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    eligible = frame[value].notna() & frame[weight].notna() & (frame[weight] > 0)
    if not eligible.any():
        return float("nan")
    return float(np.average(frame.loc[eligible, value], weights=frame.loc[eligible, weight]))


def aggregate_experiment(folds: pd.DataFrame, expected_folds: int = 5) -> dict[str, Any]:
    complete = folds[folds["status"] == "complete"].copy()
    result: dict[str, Any] = {
        "experiment": str(folds.iloc[0]["experiment"]),
        "folds_complete": int(len(complete)),
        "folds_expected": expected_folds,
        "is_final": len(complete) == expected_folds,
        "blocked_folds": ",".join(
            str(value)
            for value in folds.loc[
                folds["status"] == "postprocess_boundary_blocked", "fold"
            ].tolist()
        ),
    }
    if complete.empty:
        return result
    true_positive = float(complete["true_positive"].sum())
    false_positive = float(complete["false_positive"].sum())
    false_negative = float(complete["false_negative"].sum())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    same_truth = float(complete["same_truth_events"].sum())
    same_hits = float(complete["same_matched_events"].sum())
    different_truth = float(complete["different_truth_events"].sum())
    different_hits = float(complete["different_matched_events"].sum())
    result.update(
        {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "micro_precision": precision,
            "micro_sensitivity": sensitivity,
            "micro_f1": f1,
            "macro_f1_mean": float(complete["f1"].mean()),
            "macro_f1_std": float(complete["f1"].std(ddof=0)),
            "macro_f1_min": float(complete["f1"].min()),
            "macro_f1_max": float(complete["f1"].max()),
            "weighted_start_mae_seconds": _weighted_mean(
                complete, "start_mae_seconds", "true_positive"
            ),
            "weighted_end_mae_seconds": _weighted_mean(
                complete, "end_mae_seconds", "true_positive"
            ),
            "mean_false_positives_per_observed_hour": float(
                complete["false_positives_per_observed_hour"].mean()
            ),
            "same_sensitivity": same_hits / same_truth if same_truth else float("nan"),
            "different_sensitivity": (
                different_hits / different_truth if different_truth else float("nan")
            ),
            "zero_recall_folds": int((complete["sensitivity"] <= 0).sum()),
            "boundary_flag_folds": int(
                folds["threshold_at_search_boundary"].fillna(False).sum()
            ),
        }
    )
    return result


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    return value


def _rating(value: float, metric: str, thresholds: ReportThresholds) -> str:
    if not math.isfinite(value):
        return "na"
    if metric in {"f1", "precision", "sensitivity", "same_sensitivity", "different_sensitivity"}:
        if value >= thresholds.good_score:
            return "good"
        if value >= thresholds.warning_score:
            return "warning"
        return "bad"
    if metric in {"start_mae_seconds", "end_mae_seconds"}:
        if value <= thresholds.good_mae_seconds:
            return "good"
        if value <= thresholds.warning_mae_seconds:
            return "warning"
        return "bad"
    if metric == "false_positives_per_observed_hour":
        if value <= thresholds.good_fp_per_hour:
            return "good"
        if value <= thresholds.warning_fp_per_hour:
            return "warning"
        return "bad"
    return "na"


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _metric_cell(value: Any, metric: str, thresholds: ReportThresholds) -> str:
    number = _number(value)
    rating = _rating(number, metric, thresholds)
    return f'<td class="metric {rating}">{_fmt(number)}</td>'


def _status_label(status: str) -> str:
    return {
        "complete": "已完成",
        "postprocess_boundary_blocked": "阈值边界阻塞",
        "postprocess_complete_evaluation_missing": "待测试评估",
        "model_complete_postprocess_missing": "待后处理",
        "missing": "无产物",
    }.get(status, status)


def _build_warnings(aggregate: dict[str, Any], folds: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    experiment = aggregate["experiment"]
    if not aggregate["is_final"]:
        warnings.append(
            f"{experiment}: 仅完成 {aggregate['folds_complete']}/{aggregate['folds_expected']} 折，"
            "当前汇总是临时结果，不能作为最终五折结论。"
        )
    blocked = folds.loc[folds["status"] == "postprocess_boundary_blocked", "fold"].tolist()
    if blocked:
        warnings.append(f"{experiment}: fold {blocked} 的后处理最优阈值位于搜索边界。")
    if int(aggregate.get("zero_recall_folds", 0)):
        warnings.append(f"{experiment}: 存在召回率为 0 的测试折。")
    f1_std = _number(aggregate.get("macro_f1_std"))
    if math.isfinite(f1_std) and f1_std > 0.10:
        warnings.append(f"{experiment}: 折间 F1 标准差为 {f1_std:.3f}，泛化稳定性较差。")
    sensitivity = _number(aggregate.get("micro_sensitivity"))
    different = _number(aggregate.get("different_sensitivity"))
    if math.isfinite(sensitivity) and math.isfinite(different) and sensitivity - different > 0.15:
        warnings.append(
            f"{experiment}: 异侧召回率比总体召回率低 {sensitivity - different:.3f}。"
        )
    return warnings


def _render_html(
    aggregates: list[dict[str, Any]],
    folds: pd.DataFrame,
    warnings: list[str],
    thresholds: ReportThresholds,
) -> str:
    cards: list[str] = []
    tables: list[str] = []
    for aggregate in aggregates:
        experiment = str(aggregate["experiment"])
        final_label = "完整五折" if aggregate["is_final"] else "临时结果"
        cards.append(
            f"""
            <section class="card">
              <div class="eyebrow">{html.escape(experiment)} · {final_label}</div>
              <div class="big {_rating(_number(aggregate.get('micro_f1')), 'f1', thresholds)}">
                {_fmt(aggregate.get('micro_f1'))}
              </div>
              <div class="label">总体事件级 F1</div>
              <div class="sub">完成 {aggregate['folds_complete']}/{aggregate['folds_expected']} 折</div>
            </section>
            """
        )
        experiment_folds = folds[folds["experiment"] == experiment]
        rows: list[str] = []
        for row in experiment_folds.itertuples(index=False):
            f1_value = _number(getattr(row, "f1", float("nan")))
            width = max(0.0, min(100.0, f1_value * 100)) if math.isfinite(f1_value) else 0.0
            rows.append(
                "<tr>"
                f"<td>{row.fold}</td><td><span class='status'>{html.escape(_status_label(row.status))}</span></td>"
                f"{_metric_cell(getattr(row, 'validation_auprc', float('nan')), 'f1', thresholds)}"
                f"{_metric_cell(getattr(row, 'f1', float('nan')), 'f1', thresholds)}"
                f"{_metric_cell(getattr(row, 'precision', float('nan')), 'precision', thresholds)}"
                f"{_metric_cell(getattr(row, 'sensitivity', float('nan')), 'sensitivity', thresholds)}"
                f"{_metric_cell(getattr(row, 'different_sensitivity', float('nan')), 'different_sensitivity', thresholds)}"
                f"{_metric_cell(getattr(row, 'start_mae_seconds', float('nan')), 'start_mae_seconds', thresholds)}"
                f"{_metric_cell(getattr(row, 'end_mae_seconds', float('nan')), 'end_mae_seconds', thresholds)}"
                f"{_metric_cell(getattr(row, 'false_positives_per_observed_hour', float('nan')), 'false_positives_per_observed_hour', thresholds)}"
                f"<td><div class='bar'><span style='width:{width:.1f}%'></span></div></td>"
                "</tr>"
            )
        tables.append(
            f"""
            <section class="panel">
              <h2>{html.escape(experiment)}：逐折表现</h2>
              <div class="table-wrap"><table>
                <thead><tr><th>Fold</th><th>状态</th><th>验证 AUPRC ↑</th><th>F1 ↑</th>
                <th>精确率 ↑</th><th>召回率 ↑</th><th>异侧召回 ↑</th>
                <th>起点 MAE 秒 ↓</th><th>终点 MAE 秒 ↓</th><th>FP/小时 ↓</th><th>F1</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table></div>
            </section>
            """
        )
    aggregate_rows = []
    for item in aggregates:
        aggregate_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['experiment']))}</td>"
            f"<td>{item['folds_complete']}/{item['folds_expected']}</td>"
            f"{_metric_cell(item.get('micro_f1'), 'f1', thresholds)}"
            f"{_metric_cell(item.get('micro_precision'), 'precision', thresholds)}"
            f"{_metric_cell(item.get('micro_sensitivity'), 'sensitivity', thresholds)}"
            f"{_metric_cell(item.get('same_sensitivity'), 'same_sensitivity', thresholds)}"
            f"{_metric_cell(item.get('different_sensitivity'), 'different_sensitivity', thresholds)}"
            f"{_metric_cell(item.get('weighted_start_mae_seconds'), 'start_mae_seconds', thresholds)}"
            f"{_metric_cell(item.get('weighted_end_mae_seconds'), 'end_mae_seconds', thresholds)}"
            f"{_metric_cell(item.get('mean_false_positives_per_observed_hour'), 'false_positives_per_observed_hour', thresholds)}"
            f"<td>{_fmt(item.get('macro_f1_std'))}</td>"
            "</tr>"
        )
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    if not warning_html:
        warning_html = "<li class='ok'>未发现预设的完整性、边界或稳定性警告。</li>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>生医工进食检测实验汇总</title>
<style>
:root{{--bg:#f3f6fa;--panel:#fff;--text:#172033;--muted:#64748b;--line:#dbe3ee;--good:#16865b;--warn:#b7791f;--bad:#c2414b;--accent:#275dad}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 Inter,"Microsoft YaHei",sans-serif}}
main{{max-width:1500px;margin:auto;padding:32px}} h1{{font-size:30px;margin:0 0 6px}} h2{{font-size:19px;margin:0 0 16px}}
.lead{{color:var(--muted);margin-bottom:24px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:20px}}
.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 6px 20px #1e293b0d}}
.card{{padding:20px}} .panel{{padding:20px;margin:16px 0}} .eyebrow,.sub{{color:var(--muted)}} .big{{font-size:42px;font-weight:750;line-height:1.1;margin:12px 0 2px}}
.big.good{{color:var(--good)}} .big.warning{{color:var(--warn)}} .big.bad{{color:var(--bad)}} .label{{font-weight:650}}
.table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse;white-space:nowrap}} th,td{{padding:10px 11px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} th{{color:#475569;background:#f8fafc;position:sticky;top:0}}
.metric.good{{color:var(--good);font-weight:700;background:#ecfdf5}} .metric.warning{{color:var(--warn);font-weight:700;background:#fffbeb}} .metric.bad{{color:var(--bad);font-weight:700;background:#fff1f2}}
.status{{font-size:12px;color:#475569}} .bar{{width:90px;height:8px;background:#e2e8f0;border-radius:9px;overflow:hidden}} .bar span{{display:block;height:100%;background:var(--accent)}}
.warnings{{border-left:5px solid var(--warn)}} .warnings li{{margin:6px 0}} .warnings .ok{{color:var(--good)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted)}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}
code{{background:#eef2f7;padding:2px 5px;border-radius:4px}} @media(max-width:700px){{main{{padding:18px}}}}
</style>
</head>
<body><main>
<h1>进食检测实验结果仪表板</h1>
<p class="lead">所有数值直接读取各折 <code>test_metrics.json</code>。颜色是工程辅助判读，不是官方获奖或合格线；正式结论必须基于完整五折。</p>
<div class="cards">{''.join(cards)}</div>
<section class="panel warnings"><h2>优先关注</h2><ul>{warning_html}</ul></section>
<section class="panel"><h2>总体汇总</h2><div class="table-wrap"><table>
<thead><tr><th>实验</th><th>完成折</th><th>Micro F1 ↑</th><th>精确率 ↑</th><th>召回率 ↑</th><th>同侧召回 ↑</th><th>异侧召回 ↑</th><th>起点 MAE 秒 ↓</th><th>终点 MAE 秒 ↓</th><th>平均 FP/小时 ↓</th><th>折间 F1 σ ↓</th></tr></thead>
<tbody>{''.join(aggregate_rows)}</tbody></table></div></section>
{''.join(tables)}
<section class="panel"><h2>颜色说明</h2><div class="legend">
<span><i class="dot" style="background:var(--good)"></i>较好：分数 ≥ {thresholds.good_score:.2f}；MAE ≤ {thresholds.good_mae_seconds:.0f}s；FP/h ≤ {thresholds.good_fp_per_hour:.2f}</span>
<span><i class="dot" style="background:var(--warn)"></i>关注：分数 ≥ {thresholds.warning_score:.2f}；MAE ≤ {thresholds.warning_mae_seconds:.0f}s；FP/h ≤ {thresholds.warning_fp_per_hour:.2f}</span>
<span><i class="dot" style="background:var(--bad)"></i>较差：低于或高于上述关注界限</span>
</div><p class="lead">这些颜色只用于快速定位问题，不是赛事官方门槛，也不能替代逐折、逐受试者和失败案例分析。</p></section>
<section class="panel"><h2>指标怎么读</h2>
<p><strong>F1</strong>：精确率与召回率的综合分数，越高越好；本页总体 F1 按所有已完成折的 TP、FP、FN 重新计算。</p>
<p><strong>精确率</strong>：模型报出的进食事件中有多少是真的；低表示误报多。</p>
<p><strong>召回率</strong>：真实进食事件中有多少被检出；低表示漏报多。</p>
<p><strong>起点/终点 MAE</strong>：匹配事件边界的平均绝对时间误差，单位为秒，越低越好。</p>
<p><strong>FP/小时</strong>：每小时观察数据产生的误报数，越低越好。异侧召回用于检查佩戴手与进食手不同时的性能退化。</p>
</section>
</main></body></html>"""


def generate_experiment_report(
    experiment_roots: dict[str, Path],
    output_dir: Path,
    expected_folds: int = 5,
    thresholds: ReportThresholds | None = None,
) -> dict[str, Path]:
    thresholds = thresholds or ReportThresholds()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_folds: list[pd.DataFrame] = []
    all_subjects: list[pd.DataFrame] = []
    all_relations: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    aggregates: list[dict[str, Any]] = []
    warnings: list[str] = []
    for experiment_name, experiment_root in experiment_roots.items():
        folds, subjects, relations, experiment_sources = collect_experiment_results(
            experiment_name, experiment_root, expected_folds
        )
        aggregate = aggregate_experiment(folds, expected_folds)
        all_folds.append(folds)
        if not subjects.empty:
            all_subjects.append(subjects)
        if not relations.empty:
            all_relations.append(relations)
        sources.extend(experiment_sources)
        aggregates.append(aggregate)
        warnings.extend(_build_warnings(aggregate, folds))
    fold_frame = pd.concat(all_folds, ignore_index=True)
    subject_frame = pd.concat(all_subjects, ignore_index=True) if all_subjects else pd.DataFrame()
    relation_frame = (
        pd.concat(all_relations, ignore_index=True) if all_relations else pd.DataFrame()
    )
    aggregate_frame = pd.DataFrame(aggregates)
    paths = {
        "html": output_dir / "experiment_dashboard.html",
        "summary_json": output_dir / "experiment_summary.json",
        "aggregate_csv": output_dir / "aggregate_metrics.csv",
        "fold_csv": output_dir / "fold_metrics.csv",
        "subject_csv": output_dir / "subject_metrics.csv",
        "relation_csv": output_dir / "hand_relation_metrics.csv",
    }
    aggregate_frame.to_csv(paths["aggregate_csv"], index=False)
    fold_frame.to_csv(paths["fold_csv"], index=False)
    subject_frame.to_csv(paths["subject_csv"], index=False)
    relation_frame.to_csv(paths["relation_csv"], index=False)
    summary = {
        "version": 1,
        "provisional": any(not item["is_final"] for item in aggregates),
        "aggregates": [
            {key: _finite_or_none(value) for key, value in item.items()}
            for item in aggregates
        ],
        "warnings": warnings,
        "sources": sources,
        "rating_thresholds": thresholds.__dict__,
    }
    paths["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths["html"].write_text(
        _render_html(aggregates, fold_frame, warnings, thresholds),
        encoding="utf-8",
    )
    return paths
