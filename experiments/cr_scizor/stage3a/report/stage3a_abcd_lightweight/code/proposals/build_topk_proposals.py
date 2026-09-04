"""Merge frozen proposer transition scores into a deterministic candidate table."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["pair_id", "demo_id", "task", "base_demo_id", "split", "t"]
LABEL_COLS = [
    "pair_id", "demo_id", "task", "base_demo_id", "split", "episode_length",
    "t",
    "label_status", "failure_type", "is_effective_intervention", "responsible_t",
    "responsible_start", "responsible_end", "intervention_t", "is_responsibility_region",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def rank_weight(rank: float) -> float:
    return 0.0 if not np.isfinite(rank) else 1.0 / math.log2(float(rank) + 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--full-transition-scores", type=Path, required=True)
    p.add_argument("--action-transition-scores", type=Path, required=True)
    p.add_argument("--transition-labels", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--fit-score-calibration", type=Path)
    p.add_argument("--score-calibration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    return p.parse_args()


def _read_scores(path: Path, name: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    required = set(KEYS + ["score"])
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    d = d[KEYS + ["score"]].copy()
    d = d[d["pair_id"].notna()].copy()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.rename(columns={"score": f"{name}_score"})
    if d.duplicated(KEYS).any():
        raise ValueError(f"duplicate score keys in {path}")
    return d


def _calibration(full: pd.Series, action: pd.Series) -> dict:
    def stats(s: pd.Series) -> dict:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if not len(x):
            raise ValueError("cannot calibrate an empty score column")
        return {"mean": float(x.mean()), "std": float(x.std(ddof=0)), "count": int(len(x))}
    return {"full": stats(full), "action": stats(action), "method": "sigmoid_train_zscore"}


def _apply_calibration(d: pd.DataFrame, cal: dict) -> pd.DataFrame:
    for name in ("full", "action"):
        st = cal[name]
        raw = d[f"{name}_score"].to_numpy(dtype=np.float64)
        z = (raw - float(st["mean"])) / max(float(st["std"]), 1e-8)
        d[f"{name}_score_calibrated"] = sigmoid(z)
    d["raw_full_score"] = d["full_score_calibrated"]
    d["raw_action_score"] = d["action_score_calibrated"]
    d["raw_union_score"] = d[["raw_full_score", "raw_action_score"]].max(axis=1, skipna=True)
    return d


def main() -> None:
    a = parse_args()
    if bool(a.fit_score_calibration) == bool(a.score_calibration):
        raise ValueError("specify exactly one of --fit-score-calibration or --score-calibration")
    full = _read_scores(a.full_transition_scores, "full")
    action = _read_scores(a.action_transition_scores, "action")
    d = full.merge(action, on=KEYS, how="outer", validate="one_to_one")
    if set(d["split"].dropna().unique()) - set(a.splits):
        d = d[d["split"].isin(a.splits)].copy()
    else:
        d = d[d["split"].isin(a.splits)].copy()
    if not len(d):
        raise ValueError(f"no perturbed scores for splits {a.splits}")
    if a.fit_score_calibration:
        cal = _calibration(d["full_score"], d["action_score"])
        a.fit_score_calibration.parent.mkdir(parents=True, exist_ok=True)
        a.fit_score_calibration.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    else:
        cal = json.loads(a.score_calibration.read_text(encoding="utf-8"))
    d = _apply_calibration(d, cal)
    for name in ("full", "action"):
        d[f"{name}_rank"] = d.groupby("pair_id")[f"{name}_score"].rank(method="first", ascending=False)
        d[f"in_{name}_top5"] = d[f"{name}_rank"] <= int(a.top_k)
    d["in_union_top5"] = d["in_full_top5"] | d["in_action_top5"]
    d = d[d["in_union_top5"]].copy()
    d["union_rank"] = d.groupby("pair_id")["raw_union_score"].rank(method="first", ascending=False)
    d["proposal_rank_weight"] = [max(rank_weight(x), rank_weight(y)) for x, y in zip(d.full_rank, d.action_rank)]
    labels = pd.read_parquet(a.transition_labels)
    labels = labels[labels["variant"].eq("perturbed") & labels["split"].isin(a.splits)].copy()
    labels = labels[LABEL_COLS].drop_duplicates(KEYS, keep="first")
    d = d.merge(labels, on=KEYS, how="left", validate="one_to_one")
    d["intervention_t"] = d["intervention_t"].astype("Int64")
    d["responsible_t"] = d["responsible_t"].astype("Int64")
    d["responsible_start"] = d["responsible_start"].astype("Int64")
    d["responsible_end"] = d["responsible_end"].astype("Int64")
    d["is_responsibility_region"] = d["is_responsibility_region"].fillna(False).astype(bool)
    d = d.sort_values(["split", "pair_id", "union_rank", "t"]).reset_index(drop=True)
    cols = ["pair_id", "demo_id", "task", "base_demo_id", "split", "t", "episode_length", "full_score", "full_score_calibrated", "full_rank", "action_score", "action_score_calibrated", "action_rank", "raw_full_score", "raw_action_score", "raw_union_score", "in_full_top5", "in_action_top5", "in_union_top5", "union_rank", "proposal_rank_weight", "label_status", "failure_type", "is_effective_intervention", "responsible_t", "responsible_start", "responsible_end", "intervention_t", "is_responsibility_region"]
    out = d[cols]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.output, index=False)
    summary = {"rows": int(len(out)), "pairs": int(out.pair_id.nunique()), "max_rows_per_pair": int(out.groupby("pair_id").size().max()), "splits": {str(k): int(v) for k, v in out.groupby("split").size().items()}, "calibration": cal, "top_k_per_model": int(a.top_k)}
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
