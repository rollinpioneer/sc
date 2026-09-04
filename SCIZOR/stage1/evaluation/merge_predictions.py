"""Validate and merge all Stage 1D methods against frozen transition labels."""

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["task", "demo_id", "t"]
LABEL_COLUMNS = ["task", "demo_id", "pair_id", "variant", "base_demo_id", "split", "t", "label_status", "failure_type", "perturbation_type", "is_responsible_point", "is_responsibility_region", "is_recovery", "is_expert", "is_rare", "is_slow_precise", "is_innocent_downstream", "is_no_effect_intervention"]


def _method_frame(path, labels, method):
    frame = pd.read_parquet(path)
    if "subop_score" in frame.columns:
        frame = frame.rename(columns={"subop_score": "score"})
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"{method} predictions duplicate transition keys")
    source = frame[KEYS + ["score"]]
    merged = labels.merge(source, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any() or len(source) != len(labels):
        missing = int((merged["_merge"] != "both").sum())
        raise RuntimeError(f"{method} does not align exactly with labels: missing={missing}, source={len(source)}, labels={len(labels)}")
    merged = merged.drop(columns="_merge")
    merged["method"] = method
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--future", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = pd.read_parquet(args.labels)[LABEL_COLUMNS]
    if labels.duplicated(KEYS).any():
        raise RuntimeError("transition labels duplicate transition keys")
    merged = pd.concat([
        _method_frame(args.original, labels, "original_scizor"),
        _method_frame(args.uniform, labels, "uniform"),
        _method_frame(args.future, labels, "future_discount"),
    ], ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
