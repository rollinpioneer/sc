"""Summarize simulator-oracle coverage and pair-level discrimination.

The summary is descriptive: it reports the frozen oracle ceiling and does not
select a model, threshold, or proposal source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _metric(frame: pd.DataFrame) -> dict:
    """Return AUROC/AUPRC when both classes and finite scores are present."""
    if frame.empty:
        return {"n_pairs": 0, "effective_pairs": 0, "no_effect_pairs": 0, "auroc": None, "auprc": None}
    d = frame[np.isfinite(pd.to_numeric(frame["score"], errors="coerce"))].copy()
    y = d["is_effective_intervention"].astype(bool).astype(int)
    out = {
        "n_pairs": int(len(d)),
        "effective_pairs": int(y.sum()),
        "no_effect_pairs": int((y == 0).sum()),
        "auroc": None,
        "auprc": None,
    }
    if len(d) >= 2 and y.nunique() == 2:
        s = d["score"].astype(float).to_numpy()
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    return out


def _pairs(d: pd.DataFrame, replacement_mask: pd.Series | np.ndarray) -> pd.DataFrame:
    """Aggregate replacement scores to all pair IDs, filling absent scores by 0."""
    base_cols = ["pair_id", "task", "is_effective_intervention", "failure_type", "label_status"]
    base = d[base_cols].drop_duplicates("pair_id").copy()
    sub = d.loc[np.asarray(replacement_mask, dtype=bool)].copy()
    if sub.empty:
        base["score"] = 0.0
        return base
    sub["oracle_improvement"] = pd.to_numeric(sub["oracle_improvement"], errors="coerce")
    scores = sub.groupby("pair_id", as_index=False)["oracle_improvement"].max().rename(columns={"oracle_improvement": "score"})
    base = base.merge(scores, on="pair_id", how="left", validate="one_to_one")
    base["score"] = base["score"].fillna(0.0)
    return base


def _source_metrics(d: pd.DataFrame, source: str) -> dict:
    """Compute pair-level metrics for a proposal source or oracle subset."""
    if source == "paired_clean_upper_bound":
        mask = d["oracle_only"].fillna(False) & d["target_valid"].fillna(False)
    elif source == "primary_feasible":
        mask = (
            ~d["oracle_only"].fillna(False)
            & d["target_valid"].fillna(False)
            & d["state_in_domain"].fillna(False)
            & d["action_in_domain"].fillna(False)
        )
    elif source == "teacher_forced_primary":
        mask = (
            d["query_source"].astype(str).str.contains("teacher_forced", na=False)
            & ~d["oracle_only"].fillna(False)
            & d["target_valid"].fillna(False)
            & d["state_in_domain"].fillna(False)
            & d["action_in_domain"].fillna(False)
        )
    else:
        mask = (
            d[source].fillna(False)
            & ~d["oracle_only"].fillna(False)
            & d["target_valid"].fillna(False)
            & d["state_in_domain"].fillna(False)
            & d["action_in_domain"].fillna(False)
        )
    pair = _pairs(d, mask)
    result = {"overall": _metric(pair), "by_task": {}}
    for task, sub in pair.groupby("task", sort=True):
        result["by_task"][str(task)] = _metric(sub)
    return result


def _summarize(path: Path) -> dict:
    d = pd.read_parquet(path)
    split = str(d["split"].dropna().iloc[0]) if len(d) else path.stem
    if not len(d):
        return {"split": split, "rows": 0, "queries": 0, "valid_rows": 0, "verifier_eligible_rows": 0}

    primary = (
        ~d["oracle_only"].fillna(False)
        & d["target_valid"].fillna(False)
        & d["state_in_domain"].fillna(False)
        & d["action_in_domain"].fillna(False)
    )
    query_status = d.groupby("query_id", as_index=False).agg(
        has_primary=("replacement_id", lambda x: bool(primary.loc[x.index].any())),
        task=("task", "first"),
    )
    source_counts = {str(k): int(v) for k, v in d["replacement_source"].value_counts(dropna=False).items()}
    query_ood_rate = float((d.groupby("query_id").size() == 0).mean()) if len(query_status) else 0.0
    return {
        "split": split,
        "rows": int(len(d)),
        "queries": int(d["query_id"].nunique()),
        "valid_rows": int(d["target_valid"].fillna(False).sum()),
        "verifier_eligible_rows": int(d["verifier_eligible"].fillna(False).sum()),
        "primary_feasible_rows": int(primary.sum()),
        # OOD is defined by the planner as a query with no retrieved rows.
        # Keep the stricter verifier-eligibility gap separate so it is not
        # mislabeled as retrieval OOD (tail queries can fail the >=10-frame
        # validity check even when retrieval itself succeeded).
        "query_ood_rate": query_ood_rate,
        "no_primary_feasible_rate": float((~query_status["has_primary"]).mean()) if len(query_status) else None,
        "reference_replay_ok_rate": float(d["reference_replay_ok"].fillna(False).mean()),
        "replacement_source_counts": source_counts,
        "oracle_metrics": {
            "paired_clean_upper_bound": _source_metrics(d, "paired_clean_upper_bound"),
            "primary_feasible_replacement": _source_metrics(d, "primary_feasible"),
            "teacher_forced_primary_feasible": _source_metrics(d, "teacher_forced_primary"),
            "full_top5_proposal": _source_metrics(d, "in_full_top5"),
            "action_top5_proposal": _source_metrics(d, "in_action_top5"),
            "union_top5_proposal": _source_metrics(d, "in_union_top5"),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {"train": _summarize(args.train), "validation": _summarize(args.validation)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
