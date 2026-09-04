"""Aggregate frozen proposal-candidate verifier scores to transitions and pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = {"full_top5": "full", "action_top5": "action", "union_top5": "union"}


def best(frame, key):
    return frame.loc[frame[key].astype(float).idxmax()]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--ensemble", type=Path, required=True); p.add_argument("--action-only-predictions", type=Path, required=True); p.add_argument("--samples", type=Path, required=True); p.add_argument("--proposals", type=Path, required=True); p.add_argument("--labels", type=Path, required=True); p.add_argument("--split", required=True); p.add_argument("--frozen-protocol", type=Path); p.add_argument("--output-replacements", type=Path, required=True); p.add_argument("--output-transitions", type=Path, required=True); p.add_argument("--output-pairs", type=Path, required=True); a = p.parse_args()
    if a.frozen_protocol is not None:
        protocol = json.loads(a.frozen_protocol.read_text(encoding="utf-8"))
        if protocol.get("selected_score") not in (None, "fused_transition_score"):
            raise ValueError("pipeline score table is incompatible with frozen selected score")
    ensemble = pd.read_parquet(a.ensemble); action = pd.read_parquet(a.action_only_predictions)[["replacement_id", "pred_score", "pred_positive_probability"]].rename(columns={"pred_score": "action_only_pred_score", "pred_positive_probability": "action_only_pred_positive"})
    candidates = ensemble.merge(action, on="replacement_id", validate="one_to_one"); candidates = candidates[(~candidates.is_teacher_forced) & candidates.split.eq(a.split)].copy()
    proposal = pd.read_parquet(a.proposals); proposal = proposal[proposal.split.eq(a.split)].copy()
    raw_columns = ["pair_id", "t", "full_rank", "action_rank", "union_rank", "raw_full_score", "raw_action_score", "raw_union_score", "proposal_rank_weight", "in_full_top5", "in_action_top5", "in_union_top5"]
    candidates = candidates.merge(proposal[raw_columns], left_on=["pair_id", "query_t"], right_on=["pair_id", "t"], suffixes=("", "_proposal"), validate="many_to_one")
    labels = pd.read_parquet(a.labels)
    labels = labels[labels.variant.eq("perturbed") & labels.split.eq(a.split)][["pair_id", "t", "is_recovery"]].drop_duplicates(["pair_id", "t"])
    candidates = candidates.merge(labels, left_on=["pair_id", "query_t"], right_on=["pair_id", "t"], suffixes=("", "_label"), validate="many_to_one")
    candidates["replacement_cf_score"] = candidates.replacement_cf_score.astype(float)
    candidates["transition_cf_score"] = candidates.groupby(["pair_id", "query_t"]).replacement_cf_score.transform("max")
    candidates.to_parquet(a.output_replacements, index=False)
    transition_rows, pair_rows = [], []
    for source, prefix in SOURCES.items():
        mask = candidates[f"in_{prefix}_top5"].fillna(False)
        part = candidates[mask].copy()
        for (pair_id, t), group in part.groupby(["pair_id", "query_t"], sort=False):
            first = group.iloc[0].to_dict(); cf = float(group.replacement_cf_score.max()); fused = cf * float(first["proposal_rank_weight"]); raw = float(first[f"raw_{prefix}_score"])
            transition_rows.append({**{key: first[key] for key in ("pair_id", "task", "split", "query_t", "failure_type", "is_effective_intervention", "intervention_t", "responsible_start", "responsible_end", "is_recovery") if key in first}, "proposer_source": source, "raw_proposer_score": raw, "counterfactual_only_score": cf, "fused_transition_score": fused, "proposal_rank_weight": float(first["proposal_rank_weight"])})
    transitions = pd.DataFrame(transition_rows)
    for (source, pair_id), group in transitions.groupby(["proposer_source", "pair_id"], sort=False):
        raw = best(group, "raw_proposer_score"); cf = best(group, "counterfactual_only_score"); fused = best(group, "fused_transition_score")
        pair_rows.append({"proposer_source": source, "pair_id": pair_id, "task": str(group.task.iloc[0]), "split": str(group.split.iloc[0]), "failure_type": str(group.failure_type.iloc[0]), "is_effective_intervention": bool(group.is_effective_intervention.iloc[0]), "intervention_t": group.intervention_t.iloc[0], "responsible_start": group.responsible_start.iloc[0], "responsible_end": group.responsible_end.iloc[0], "raw_proposer_pair_score": float(raw.raw_proposer_score), "counterfactual_only_pair_score": float(cf.counterfactual_only_score), "fused_pair_score": float(fused.fused_transition_score), "raw_proposer_predicted_t": int(raw.query_t), "counterfactual_predicted_t": int(cf.query_t), "fused_predicted_t": int(fused.query_t), "mean_candidates_per_pair": int(len(group))})
    pairs = pd.DataFrame(pair_rows)
    for output, frame in ((a.output_transitions, transitions), (a.output_pairs, pairs)):
        output.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(output, index=False)
    print({"replacement_rows": len(candidates), "transition_rows": len(transitions), "pair_rows": len(pairs)})


if __name__ == "__main__": main()
