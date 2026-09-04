"""Evaluate learned verifier approximation of candidate and teacher-forced oracle targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binary


def recall_matched_far(frame, score):
    y = frame.is_effective_intervention.astype(bool).to_numpy(); values = frame[score].to_numpy(float)
    if not y.any() or y.all(): return None
    thresholds = np.unique(values)
    feasible = []
    for threshold in thresholds:
        pred = values >= threshold; recall = pred[y].mean()
        if recall >= 0.4: feasible.append((threshold, float(pred[~y].mean())))
    return min(feasible, key=lambda item: item[1])[1] if feasible else None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ensemble", type=Path, required=True); p.add_argument("--action-only", type=Path, required=True); p.add_argument("--samples", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--csv", type=Path, required=True); a = p.parse_args()
    full = pd.read_parquet(a.ensemble); action = pd.read_parquet(a.action_only)[["replacement_id", "pred_score", "pred_positive_probability"]].rename(columns={"pred_score": "action_pred_score", "pred_positive_probability": "action_pred_positive"})
    frame = full.merge(action, on="replacement_id", validate="one_to_one")
    target = frame.target_positive.astype(int)
    # The teacher-forced diagnostic must never fall back to a proposal row
    # that happens to query the same intervention time.  Proposal candidates
    # are evaluated separately at candidate-replacement scope.
    primary = frame[
        frame.is_teacher_forced
        & frame.query_t.eq(frame.intervention_t)
        & frame.replacement_rank.eq(0)
    ].drop_duplicates("pair_id")
    metrics = {
        "candidate_replacement": {"full": binary(target, frame.pred_positive_mean), "action_only": binary(target, frame.action_pred_positive), "full_score_mae": float(np.abs(frame.pred_score_mean - frame.counterfactual_improvement_long).mean()), "full_score_spearman": float(frame.pred_score_mean.corr(frame.counterfactual_improvement_long, method="spearman"))},
        "teacher_forced_primary": {"full": binary(primary.is_effective_intervention.astype(int), primary.pred_score_mean), "action_only": binary(primary.is_effective_intervention.astype(int), primary.action_pred_score), "by_task": {task: binary(part.is_effective_intervention.astype(int), part.pred_score_mean) for task, part in primary.groupby("task")}},
    }
    full_ap, action_ap = metrics["candidate_replacement"]["full"]["auprc"], metrics["candidate_replacement"]["action_only"]["auprc"]
    metrics["full_vs_action_only"] = {"replacement_auprc_difference_full_minus_action": None if full_ap is None or action_ap is None else float(full_ap - action_ap), "teacher_forced_auroc_difference_full_minus_action": None if metrics["teacher_forced_primary"]["full"]["auroc"] is None or metrics["teacher_forced_primary"]["action_only"]["auroc"] is None else float(metrics["teacher_forced_primary"]["full"]["auroc"] - metrics["teacher_forced_primary"]["action_only"]["auroc"]), "full_matched_recall_no_effect_far": recall_matched_far(primary, "pred_score_mean"), "action_only_matched_recall_no_effect_far": recall_matched_far(primary, "action_pred_score")}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([{ "scope": "candidate_replacement", "method": "full", **metrics["candidate_replacement"]["full"]}, {"scope": "candidate_replacement", "method": "action_only", **metrics["candidate_replacement"]["action_only"]}, {"scope": "teacher_forced_primary", "method": "full", **metrics["teacher_forced_primary"]["full"]}, {"scope": "teacher_forced_primary", "method": "action_only", **metrics["teacher_forced_primary"]["action_only"]}]).to_csv(a.csv, index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
