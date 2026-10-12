from __future__ import annotations

import copy

from bme_eating.hierarchical_artifacts import sha256_file, write_json_atomic
from bme_eating.joint_protocol import digest
from bme_eating.training.hierarchical_v4_trainer import _selector_split, stacking_partitions
from bme_eating.training.statsfusion_joint_trainer import train_joint_model


def execute_ablations(inputs, config, run, resume):
    outer = {subject for subject, fold in inputs.subject_folds.items() if fold == 0}
    subjects = set(inputs.subject_folds) - outer
    partitions = stacking_partitions(subjects, 3, 2026)
    prediction = {subject for subject, partition in partitions.items() if partition == 0}
    training = subjects - prediction
    anchors, _, events, _ = inputs.frames(training, labels=True)
    fit, selector, report = _selector_split(training, .35, 2026, events=events, anchors=anchors)
    results = []
    for name in ("no_onset", "no_boundary", "single_candidate"):
        ablation = copy.deepcopy(config)
        ablation["ablation"] = {"name": name, "diagnostic_only": True, "fold": 0, "partition": 0}
        ablation["decoder"]["ablation"] = name
        directory = run / "ablations" / name
        identity = digest({"run": sha256_file(run / "identity.json"), "ablation": name,
                           "fit": sorted(fit), "selector": sorted(selector), "excluded": sorted(outer | prediction)})
        write_json_atomic(directory / "resolved_config.json", ablation)
        write_json_atomic(directory / "selector_split.json", report)
        epoch = train_joint_model(inputs, fit, ablation, directory / "selection", identity,
            epochs=ablation["training"]["max_epochs"], resume=resume, selector=selector,
            outer_excluded=outer | prediction)
        results.append({"name": name, "epoch": epoch, "identity": identity,
                        "diagnostic_only": True, "automatic_main_selection": False,
                        "fit_subjects": sorted(fit), "selector_subjects": sorted(selector)})
    write_json_atomic(run / "ablation_manifest.json", results)
    return results
