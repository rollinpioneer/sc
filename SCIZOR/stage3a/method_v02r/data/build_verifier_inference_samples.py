"""Build target-free verifier input rows for a frozen blind-test plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_PLAN_COLUMNS = {
    "replacement_id",
    "query_id",
    "pair_id",
    "task",
    "split",
    "perturbed_demo_id",
    "query_t",
    "replacement_rank",
    "replacement_action",
    "state_distance",
    "action_delta_l2",
    "state_in_domain",
    "action_in_domain",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    plans = pd.read_parquet(args.plans).copy()
    missing = REQUIRED_PLAN_COLUMNS.difference(plans.columns)
    if missing:
        raise ValueError(f"plan table missing columns: {sorted(missing)}")
    if plans.empty or plans.replacement_id.duplicated().any():
        raise ValueError("blind verifier plans must be non-empty with unique replacement IDs")
    if plans.groupby("query_id").size().ne(4).any():
        raise ValueError("blind verifier plans must contain exactly four replacements per query")

    features = pd.read_parquet(args.feature_index)
    feature_keys = {(str(row.task), str(row.demo_id)) for row in features.itertuples(index=False)}
    missing_features = sorted({(str(row.task), str(row.perturbed_demo_id)) for row in plans.itertuples(index=False)} - feature_keys)
    if missing_features:
        raise ValueError(f"missing feature rows for {missing_features[:5]}")

    # Keep all plan/metadata columns for downstream audit, but make the
    # target fields explicit NaNs. The Dataset and inference code never use
    # these labels; explicit columns prevent accidental schema drift.
    plans["query_group_id"] = plans["query_id"].astype(str)
    plans["target_valid"] = False
    plans["counterfactual_improvement_long"] = float("nan")
    plans["target_positive"] = float("nan")
    plans["is_teacher_forced"] = False
    plans["split"] = plans["split"].astype(str)
    plans = plans.sort_values(["pair_id", "query_t", "replacement_rank", "replacement_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plans.to_parquet(args.output, index=False)
    summary = {
        "rows": int(len(plans)),
        "query_groups": int(plans.query_group_id.nunique()),
        "split_values": sorted(plans.split.unique().tolist()),
        "target_columns_are_unlabeled": True,
        "state_ood_rows": int((~plans.state_in_domain.astype(bool)).sum()),
        "action_ood_rows": int((~plans.action_in_domain.astype(bool)).sum()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
