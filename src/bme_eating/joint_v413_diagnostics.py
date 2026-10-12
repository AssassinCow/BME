from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classifier_diagnostics(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    report = {"candidates": len(labels), "positives": int(labels.sum()),
              "negatives": int((labels == 0).sum()), "roc_auc": None, "average_precision": None}
    if len(np.unique(labels)) == 2:
        report.update({"roc_auc": float(roc_auc_score(labels, logits)),
                       "average_precision": float(average_precision_score(labels, logits))})
    for name, target in (("positive", 1), ("negative", 0)):
        values = logits[labels == target]
        report[name + "_logits"] = {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "quantiles": np.quantile(values, [0, .1, .5, .9, 1]).tolist() if len(values) else []}
    return report


def source_coverage(candidates, truth, windows):
    from bme_eating.joint_evaluation import candidate_coverage
    groups = {}
    for item in candidates:
        parent = item.generation_parameters.get("parent_source", item.source)
        groups.setdefault(f"{parent}:{item.scale_seconds}", []).append(item)
    return {name: candidate_coverage(pool, truth, windows) for name, pool in sorted(groups.items())}
