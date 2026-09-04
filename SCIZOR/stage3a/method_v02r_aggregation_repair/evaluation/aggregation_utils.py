"""Small, deterministic helpers for the bounded aggregation repair."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def binary_metrics(labels: Iterable, scores: Iterable) -> dict:
    y = np.asarray(list(labels), dtype=int); s = np.asarray(list(scores), dtype=float)
    result = {"count": int(len(y)), "positive_count": int(y.sum()), "prevalence": float(y.mean()) if len(y) else 0.0}
    result["auroc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None
    result["auprc"] = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else None
    return result


def threshold_metrics(labels: Iterable, scores: Iterable, threshold: float) -> dict:
    y = np.asarray(list(labels), dtype=bool); s = np.asarray(list(scores), dtype=float); pred = s >= float(threshold)
    no = ~y; effective = y
    return {"threshold": float(threshold), "no_effect_far": float(pred[no].mean()) if no.any() else 0.0,
            "effective_recall": float(pred[effective].mean()) if effective.any() else 0.0,
            "predicted_positive_count": int(pred.sum())}


def build_pair_universe(labels: pd.DataFrame, split: str) -> pd.DataFrame:
    cols = ["pair_id", "task", "split", "failure_type", "is_effective_intervention", "intervention_t",
            "responsible_start", "responsible_end"]
    part = labels[(labels.variant == "perturbed") & (labels.split == split)].copy()
    return part.sort_values("t").drop_duplicates("pair_id")[cols].reset_index(drop=True)


def attach_local_deficit(candidates: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy(); values = []; covered_values = []
    # Convert each evidence group to compact interval arrays once.  Filtering
    # a pandas frame for every candidate is quadratic on the train table.
    grouped = {(str(k[0]), str(k[1])): (g.start_t.to_numpy(int), g.end_t.to_numpy(int), g.V_c.to_numpy(float))
               for k, g in evidence.groupby(["task", "demo_id"], sort=False)}
    for row in out.itertuples(index=False):
        demo = getattr(row, "demo_id", getattr(row, "perturbed_demo_id", "")); g = grouped.get((str(row.task), str(demo)))
        t = int(getattr(row, "query_t", getattr(row, "t", -1)))
        if g is None:
            values.append(0.0); covered_values.append(False); continue
        starts, ends, vc = g; mask = (starts <= t) & (t < ends)
        values.append(float(np.clip(vc[mask].max() if mask.any() else 0.0, 0.0, 1.0))); covered_values.append(bool(mask.any()))
    out["local_deficit"] = values; out["deficit_coverage"] = covered_values
    return out


def replacement_summary(group: pd.DataFrame) -> dict:
    scores = group.replacement_cf_score.astype(float).to_numpy()
    return {"cf_max": float(np.max(scores)) if len(scores) else 0.0,
            "cf_median": float(np.median(scores)) if len(scores) else 0.0,
            "cf_contrast": float(max(np.max(scores) - np.median(scores), 0.0)) if len(scores) else 0.0}


def source_columns(source: str) -> tuple[str, str, str]:
    if source == "action_top5": return "in_action_top5", "raw_action_score", "action_rank"
    if source == "union_top5": return "in_union_top5", "raw_union_score", "union_rank"
    if source == "full_top5": return "in_full_top5", "raw_full_score", "full_rank"
    raise ValueError(source)


def fill_missing_pairs(pair_scores: pd.DataFrame, pair_universe: pd.DataFrame, method_id: str, source: str) -> pd.DataFrame:
    keys = pair_scores[["pair_id"]].drop_duplicates() if len(pair_scores) else pd.DataFrame(columns=["pair_id"])
    missing = pair_universe[~pair_universe.pair_id.isin(keys.pair_id)].copy()
    if len(missing):
        missing["method_id"] = method_id; missing["source"] = source; missing["pair_score"] = 0.0
        missing["predicted_t"] = -1; missing["candidate_count"] = 0; missing["has_valid_candidate"] = False
        pair_scores = pd.concat([pair_scores, missing], ignore_index=True, sort=False)
    return pair_scores.sort_values("pair_id").reset_index(drop=True)
