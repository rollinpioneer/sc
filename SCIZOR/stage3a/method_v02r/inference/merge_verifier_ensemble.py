"""Freeze mean/std lower-confidence-bound ensemble scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(); p.add_argument("--inputs", nargs=3, type=Path, required=True); p.add_argument("--samples", type=Path, required=True); p.add_argument("--std-multiplier", type=float, default=1.0); p.add_argument("--score-max", type=float, default=0.9); p.add_argument("--output", type=Path, required=True); p.add_argument("--summary", type=Path, required=True); a = p.parse_args()
    predictions = [pd.read_parquet(path).set_index("replacement_id") for path in a.inputs]
    ids = predictions[0].index
    if any(not item.index.equals(ids) for item in predictions[1:]): raise ValueError("ensemble prediction IDs differ")
    scores = np.stack([item.loc[ids, "pred_score"].to_numpy(float) for item in predictions], 1); positives = np.stack([item.loc[ids, "pred_positive_probability"].to_numpy(float) for item in predictions], 1)
    frame = pd.read_parquet(a.samples).merge(pd.DataFrame({"replacement_id": ids, "pred_score_mean": scores.mean(1), "pred_score_std": scores.std(1), "pred_positive_mean": positives.mean(1), "pred_positive_std": positives.std(1)}), on="replacement_id", validate="one_to_one")
    frame["pred_score_lcb"] = np.clip(frame.pred_score_mean - a.std_multiplier * frame.pred_score_std, 0, a.score_max)
    frame["pred_positive_lcb"] = np.clip(frame.pred_positive_mean - a.std_multiplier * frame.pred_positive_std, 0, 1)
    frame["replacement_cf_score"] = frame.pred_score_lcb / a.score_max * frame.pred_positive_lcb
    a.output.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(a.output, index=False)
    summary = {"rows": int(len(frame)), "seed_count": 3, "std_multiplier": a.std_multiplier, "score_max": a.score_max, "formula": "clip(score_mean-score_std,0,0.9)/0.9*clip(positive_mean-positive_std,0,1)"}; a.summary.parent.mkdir(parents=True, exist_ok=True); a.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
