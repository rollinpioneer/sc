"""Choose validation-only, matched-deletion operating points."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _active(frame, split):
    return frame[(frame["split"] == split) & (frame["variant"] == "perturbed") & (frame["label_status"] != "ambiguous")].copy()


def _top_budget_threshold(frame, budget):
    ranked = frame.sort_values(["score", "task", "demo_id", "t"], ascending=[False, True, True, True], kind="stable")
    if budget <= 0:
        return float("inf"), 0
    budget = min(budget, len(ranked))
    return float(ranked.iloc[budget - 1]["score"]), budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reference-method", default="original_scizor")
    parser.add_argument("--reference-percentile", type=float, default=0.70)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    active = _active(predictions, args.split)
    reference = active[active["method"] == args.reference_method]
    if reference.empty:
        raise RuntimeError("reference method has no validation transitions")
    threshold = float(np.quantile(reference["score"], args.reference_percentile))
    budget = int((reference["score"] >= threshold).sum())
    operating = {args.reference_method: {"threshold": threshold, "validation_delete_count": budget, "selection": "score >= validation percentile threshold"}}
    for method in sorted(set(active["method"]) - {args.reference_method}):
        value, count = _top_budget_threshold(active[active["method"] == method], budget)
        operating[method] = {"threshold": value, "validation_delete_count": count, "selection": "validation score rank with deterministic task,demo,t tie break"}
    result = {"selection_split": args.split, "reference_percentile": args.reference_percentile, "active_validation_transition_count": len(reference), "methods": operating}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
