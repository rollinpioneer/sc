"""Select one validation-only proposer and operating threshold under frozen rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binary, threshold_metrics


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def choose_threshold(pair):
    labels, scores = pair.is_effective_intervention.astype(bool).to_numpy(), pair.fused_pair_score.to_numpy(float)
    options = []
    for threshold in np.unique(scores):
        item = threshold_metrics(labels, scores, threshold)
        if item["no_effect_far"] is not None and item["no_effect_far"] <= 0.20:
            localization = float(((scores >= threshold) & labels & (np.abs(pair.fused_predicted_t.astype(int).to_numpy() - pair.intervention_t.fillna(-999).astype(int).to_numpy()) <= 1)).sum() / max(labels.sum(), 1))
            options.append((item["effective_recall"], localization, float(threshold), item))
    if not options: return None
    return sorted(options, key=lambda value: (value[0], value[1], value[2]), reverse=True)[0][-1]


def source_metrics(pair, transition, source, transfer):
    labels, scores = pair.is_effective_intervention.astype(int), pair.fused_pair_score.astype(float)
    base = binary(labels, scores)
    threshold = choose_threshold(pair)
    effective = pair[pair.is_effective_intervention.astype(bool)]
    top1 = float((np.abs(effective.fused_predicted_t.astype(int) - effective.intervention_t.astype(int)) <= 1).mean()) if len(effective) else None
    iou = float(((effective.fused_predicted_t.astype(int) >= effective.responsible_start.astype(int)) & (effective.fused_predicted_t.astype(int) <= effective.responsible_end.astype(int))).mean() / 3.0) if len(effective) else None
    delay = float(np.abs(effective.fused_predicted_t.astype(int) - effective.intervention_t.astype(int)).mean()) if len(effective) else None
    by_task = {task: binary(part.is_effective_intervention.astype(int), part.fused_pair_score) for task, part in pair.groupby("task")}
    recovery_far = None
    if threshold is not None and "is_recovery" in transition:
        recovery = transition.is_recovery.astype(bool)
        if recovery.any():
            recovery_far = float((transition.loc[recovery, "fused_transition_score"].astype(float) >= threshold["threshold"]).mean())
    result = {**base, "proposal_region_recall": transfer["validation"][source]["overall"]["responsibility_region_recall"], "mean_candidates_per_pair": float(pair.mean_candidates_per_pair.mean()), "threshold": threshold, "top1_within_1": top1, "region_iou": iou, "mean_abs_localization_delay": delay, "recovery_false_attribution": recovery_far, "by_task": by_task}
    if threshold:
        result.update({"no_effect_far": threshold["no_effect_far"], "effective_recall": threshold["effective_recall"]})
    else: result.update({"no_effect_far": None, "effective_recall": None})
    return result


def choose_source(sources, metrics):
    """Apply the frozen tolerance-aware proposer tie breaks in order."""
    if not sources:
        return None
    best_ap = max(metrics[source]["auprc"] for source in sources)
    tied = [source for source in sources if metrics[source]["auprc"] >= best_ap - 0.01]
    best_far = min(metrics[source]["no_effect_far"] for source in tied)
    tied = [source for source in tied if metrics[source]["no_effect_far"] <= best_far + 0.02]
    best_recall = max(metrics[source]["effective_recall"] for source in tied)
    tied = [source for source in tied if metrics[source]["effective_recall"] == best_recall]
    return min(tied, key=lambda source: metrics[source]["mean_candidates_per_pair"])


def artifact_hashes() -> dict:
    root = Path(os.environ["STAGE3_METHOD_ROOT"])
    library = Path(os.environ["ACTION_LIBRARY_V02"])
    checkpoints = {f"full_seed_{seed}": root / "runs" / f"full_seed_{seed}" / "best.pt" for seed in (0, 1, 2)}
    return {
        "verifier_checkpoints": {name: {"path": str(path), "sha256": sha(path)} for name, path in checkpoints.items()},
        "proposal_calibration_sha256": sha(root / "proposals" / "proposal_score_calibration_v02.json"),
        "action_library": {"index_sha256": sha(library / "action_library_index.parquet"), "support_thresholds_sha256": sha(library / "support_thresholds.json")},
        "feature_normalizer_sha256": sha(root / "features" / "verifier_normalizer.npz"),
        "oracle_score_spec_sha256": sha(Path(os.environ["SCORE_SPEC_V02R"])),
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--pair-scores", type=Path, required=True); p.add_argument("--transition-scores", type=Path, required=True); p.add_argument("--replacement-scores", type=Path, required=True); p.add_argument("--verifier-learning", type=Path, required=True); p.add_argument("--proposer-transfer", type=Path, required=True); p.add_argument("--config", type=Path, required=True); p.add_argument("--output-metrics", type=Path, required=True); p.add_argument("--output-protocol", type=Path, required=True); p.add_argument("--output-csv", type=Path, required=True); a = p.parse_args()
    config, learning, transfer = json.loads(a.config.read_text()), json.loads(a.verifier_learning.read_text()), json.loads(a.proposer_transfer.read_text())
    pairs, transitions = pd.read_parquet(a.pair_scores), pd.read_parquet(a.transition_scores)
    all_metrics = {}
    valid_sources = []
    for source in ("full_top5", "action_top5", "union_top5"):
        pair, transition = pairs[pairs.proposer_source.eq(source)].copy(), transitions[transitions.proposer_source.eq(source)].copy()
        item = source_metrics(pair, transition, source, transfer); all_metrics[source] = item
        task_ok = all(item["by_task"].get(task, {}).get("auroc") is not None and item["by_task"][task]["auroc"] >= config["validation"]["min_task_pair_auroc"] for task in ("can", "square"))
        if item["proposal_region_recall"] is not None and item["proposal_region_recall"] >= config["validation"]["min_proposal_region_recall"] and item["auroc"] is not None and item["auroc"] >= config["validation"]["min_pipeline_pair_auroc"] and item["threshold"] and item["effective_recall"] >= config["validation"]["min_pipeline_effective_recall"] and task_ok:
            valid_sources.append(source)
    shortcut = learning["full_vs_action_only"]
    action_better_ap = shortcut["replacement_auprc_difference_full_minus_action"] is not None and shortcut["replacement_auprc_difference_full_minus_action"] < -0.02
    full_far, action_far = shortcut["full_matched_recall_no_effect_far"], shortcut["action_only_matched_recall_no_effect_far"]
    shortcut_failure = bool(action_better_ap and full_far is not None and action_far is not None and action_far < full_far - 0.05)
    selected = choose_source(valid_sources, all_metrics)
    candidate_auc = learning["candidate_replacement"]["full"]["auroc"]; teacher_auc = learning["teacher_forced_primary"]["full"]["auroc"]
    failed = []
    if candidate_auc is None or candidate_auc < config["validation"]["min_replacement_oracle_auroc"]: failed.append("candidate_replacement_oracle_auroc")
    if teacher_auc is None or teacher_auc < config["validation"]["min_teacher_forced_pair_auroc"]: failed.append("teacher_forced_primary_pair_auroc")
    if selected is None: failed.append("no_valid_proposer_pipeline")
    if shortcut_failure: failed.append("action_only_shortcut_failure")
    passed = not failed
    protocol = {"validation_gate_pass": passed, "selected_proposer": selected, "selected_threshold": all_metrics[selected]["threshold"]["threshold"] if selected else None, "selected_score": "fused_transition_score", "full_top_k": 5, "action_top_k": 5, "replacement_score_formula": "clip(score_mean-score_std,0,0.9)/0.9*clip(positive_mean-positive_std,0,1)", "all_validation_metrics": all_metrics, "verifier_learning": learning, "shortcut_failure": shortcut_failure, "failed_rules": failed, "proposer_transfer": transfer["validation"], "artifact_hashes": artifact_hashes()}
    a.output_metrics.parent.mkdir(parents=True, exist_ok=True); a.output_metrics.write_text(json.dumps({"sources": all_metrics, "selected_proposer": selected, "validation_gate_pass": passed, "failed_rules": failed}, indent=2), encoding="utf-8")
    a.output_protocol.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = [{"method": source, **{key: value for key, value in item.items() if key not in {"by_task", "threshold"}}, "threshold": item["threshold"]["threshold"] if item["threshold"] else None, "can_auroc": item["by_task"].get("can", {}).get("auroc"), "square_auroc": item["by_task"].get("square", {}).get("auroc")} for source, item in all_metrics.items()]
    pd.DataFrame(rows).to_csv(a.output_csv, index=False)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__": main()
