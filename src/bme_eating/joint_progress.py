from __future__ import annotations

import shutil

from tqdm.auto import tqdm


class TerminalProgress(tqdm):
    @property
    def format_dict(self):
        fields = super().format_dict
        columns = fields.get("ncols") or shutil.get_terminal_size(fallback=(100, 24)).columns
        fields["ncols"] = max(1, columns - 1)
        return fields


def progress(iterable, *, total, description, initial=0, unit="chunk"):
    return TerminalProgress(iterable, total=total, initial=initial, desc=description,
                            unit=unit, dynamic_ncols=True, mininterval=0.5,
                            ascii=False, colour="#FFFFFF")


def print_metrics(label, metrics):
    def number(key, decimals=4):
        value = metrics.get(key)
        return "N/A" if value is None else f"{value:.{decimals}f}"

    counts = "/".join(number(key, 0) for key in
                      ("true_positive", "false_positive", "false_negative"))
    tqdm.write(f"[{label}] F1={number('f1')} TP/FP/FN={counts} "
               f"precision={number('precision')} recall={number('sensitivity')} "
               f"candidate_recall={number('candidate_recall')} FP/h={number('fp_per_hour', 5)} "
               f"start_MAE={number('start_mae_seconds', 1)}s "
               f"end_MAE={number('end_mae_seconds', 1)}s "
               f"start_bias={number('start_signed_error_seconds', 1)}s "
               f"ECE={number('calibration_ece')} eligible={metrics.get('promotion_eligible', 'N/A')} "
               f"setting={metrics.get('setting', 'N/A')} budget={metrics.get('budget', 'N/A')}/h "
               f"NMS={metrics.get('nms', 'N/A')} threshold={number('acceptance')}")

