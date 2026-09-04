"""Evaluate the blind benchmark with the already-frozen validation protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binary, threshold_metrics


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_metrics(pair: pd.DataFrame, transition: pd.DataFrame, threshold: float) -> dict:
    labels = pair.is_effective_intervention.astype(bool).to_numpy()
    scores = pair.fused_pair_score.astype(float).to_numpy()
    thresholded = threshold_metrics(labels, scores, threshold)
    effective = pair[pair.is_effective_intervention.astype(bool)].copy()
    if len(effective):
        abs_delay = np.abs(effective.fused_predicted_t.astype(int) - effective.intervention_t.astype(int))
        top1 = float((abs_delay <= 1).mean())
        iou = []
        for row in effective.itertuples(index=False):
            start, end = int(row.responsible_start), int(row.responsible_end)
            pred = int(row.fused_predicted_t)
            if start < 0 or end < start:
                iou.append(0.0)
            else:
                intersection = int(start <= pred <= end)
                iou.append(intersection / float((end - start + 1) + 1 - intersection))
        mean_iou = float(np.mean(iou))
        delay = float(abs_delay.mean())
    else:
        top1 = mean_iou = delay = None
    by_task = {task: binary(part.is_effective_intervention.astype(int), part.fused_pair_score) for task, part in pair.groupby("task")}
    recovery = transition[transition.get("is_recovery", False).astype(bool)] if "is_recovery" in transition else transition.iloc[0:0]
    recovery_far = float((recovery.fused_transition_score.astype(float) >= threshold).mean()) if len(recovery) else None
    return {
        **binary(labels.astype(int), scores),
        "threshold": float(threshold),
        "no_effect_far": thresholded["no_effect_far"],
        "effective_recall": thresholded["effective_recall"],
        "predicted_positive_count": thresholded["predicted_positive_count"],
        "top1_within_1": top1,
        "region_iou": mean_iou,
        "mean_abs_localization_delay": delay,
        "recovery_false_attribution": recovery_far,
        "by_task": by_task,
        "mean_candidates_per_pair": float(pair.mean_candidates_per_pair.mean()) if len(pair) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-scores", type=Path, required=True)
    parser.add_argument("--transition-scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = protocol.get("selected_proposer")
    threshold = protocol.get("selected_threshold")
    if source not in {"full_top5", "action_top5", "union_top5"} or threshold is None:
        raise ValueError("frozen validation protocol lacks selected proposer or threshold")
    pairs = pd.read_parquet(args.pair_scores)
    transitions = pd.read_parquet(args.transition_scores)
    labels = pd.read_parquet(args.labels)
    meta = jsonl(args.metadata)
    pair = pairs[pairs.proposer_source.eq(source)].copy()
    transition = transitions[transitions.proposer_source.eq(source)].copy()
    if pair.empty or transition.empty:
        raise ValueError(f"blind score table has no rows for frozen proposer {source}")
    pair_metrics = selected_metrics(pair, transition, float(threshold))
    meta_ids = {str(row["pair_id"]) for row in meta}
    observed_ids = set(pair.pair_id.astype(str))
    blind_check = bool(len(meta_ids) == 320 and observed_ids == meta_ids and len(labels[labels.variant.eq("perturbed")].pair_id.astype(str).unique()) == 320)
    metrics = {
        "benchmark_check_pass": blind_check,
        "benchmark_pair_count": int(len(meta_ids)),
        "selected_proposer": source,
        "selected_threshold": float(threshold),
        "selected_score": protocol.get("selected_score", "fused_transition_score"),
        "pair_metrics": pair_metrics,
        "all_sources": {
            str(name): selected_metrics(
                pairs[pairs.proposer_source.eq(name)],
                transitions[transitions.proposer_source.eq(name)],
                float(threshold),
            )
            for name in ("full_top5", "action_top5", "union_top5")
            if not pairs[pairs.proposer_source.eq(name)].empty
        },
        "protocol_sha256": __import__("hashlib").sha256(args.protocol.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_rows = []
    for name, item in metrics["all_sources"].items():
        csv_rows.append({"proposer_source": name, **{key: value for key, value in item.items() if key not in {"by_task"}}, "can_auroc": item["by_task"].get("can", {}).get("auroc"), "square_auroc": item["by_task"].get("square", {}).get("auroc")})
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(args.csv, index=False)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
