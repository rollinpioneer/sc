"""Merge learned and frozen Stage 1 predictions with exact label coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["task", "demo_id", "t"]
LABEL_COLUMNS = ["task", "demo_id", "pair_id", "variant", "base_demo_id", "split", "t", "label_status", "failure_type", "perturbation_type", "is_responsible_point", "is_responsibility_region", "is_recovery", "is_expert", "is_rare", "is_slow_precise", "is_innocent_downstream", "is_no_effect_intervention"]


def checked(labels, frame, method):
    source = frame[KEYS + ["score"]].copy()
    if source.duplicated(KEYS).any(): raise RuntimeError(f"duplicate keys for {method}")
    merged = labels.merge(source, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if len(source) != len(labels) or not merged._merge.eq("both").all(): raise RuntimeError(f"inexact frozen-label coverage for {method}")
    return merged.drop(columns="_merge").assign(method=method)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--baseline-predictions", required=True); p.add_argument("--learned-score-files", nargs="+", required=True); p.add_argument("--transition-labels", required=True); p.add_argument("--splits", nargs="+", required=True); p.add_argument("--output", required=True); args = p.parse_args()
    labels = pd.read_parquet(args.transition_labels)[LABEL_COLUMNS]; labels = labels[labels["split"].isin(args.splits)]
    frames = []
    baseline = pd.read_parquet(args.baseline_predictions); baseline = baseline[baseline["split"].isin(args.splits)]
    for method, frame in baseline.groupby("method", sort=True): frames.append(checked(labels[labels["split"].isin(frame["split"].unique())], frame, method))
    for path in args.learned_score_files:
        frame = pd.read_parquet(path); method = str(frame.method.iloc[0]); frames.append(checked(labels[labels["split"].isin(frame["split"].unique())], frame, method))
    output = pd.concat(frames, ignore_index=True); Path(args.output).parent.mkdir(parents=True, exist_ok=True); output.to_parquet(args.output, index=False); print(f"wrote {len(output)} rows")


if __name__ == "__main__": main()
