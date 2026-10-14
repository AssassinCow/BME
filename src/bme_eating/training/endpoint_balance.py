from __future__ import annotations

from bme_eating.data.augmentation import augmentation_description
from bme_eating.protocol import digest


def epoch_balance(dataset, epoch, weights, config):
    sessions = {}
    for key in sorted(dataset.session_groups):
        dataset.description = augmentation_description(config["project"]["seed"], epoch, *key,
            {**config["training"], "_stabilized": True})
        sessions["|".join(key)] = {name: list(dataset.endpoint_counts(key, name))
                                  for name in ("onset", "offset")}
    total = sum(weights[key[0]] for key in dataset.session_groups)
    scales = {}
    for name in ("onset", "offset"):
        peak = sum(weights[key[0]] for key in dataset.session_groups
                   if sessions["|".join(key)][name][0] > 0)
        background = sum(weights[key[0]] for key in dataset.session_groups
                         if sessions["|".join(key)][name][1] > 0)
        scales[name] = {"peak_scale": total / peak if peak else 0.0,
            "background_scale": total / background if background else 0.0,
            "peak_weight": peak, "background_weight": background}
    payload = {"epoch": epoch, "total_weight": total, "sessions": sessions, "scales": scales,
               "background_ratio": 20, "training_subjects": sorted(weights)}
    return {**payload, "sha256": digest(payload)}
