"""Build the deterministic Stage 2 responsibility chunk sample index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


META_COLUMNS = [
    "task", "demo_id", "pair_id", "variant", "base_demo_id", "split", "failure_type",
    "label_status", "is_effective_intervention", "is_no_effect_negative_control",
    "responsible_t", "responsible_start", "responsible_end", "failure_onset", "intervention_t",
]


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-evidence", type=Path, required=True)
    parser.add_argument("--transition-labels", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def top(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sort_values(["V_c", "start_t", "chunk_id"], ascending=[False, True, True]).head(n)


def spread_select(df: pd.DataFrame, position_col: str, count: int) -> pd.DataFrame:
    if len(df) <= count:
        return df
    selected: list[int] = []
    remaining = df.copy()
    for target in np.linspace(0.05, 0.95, count):
        index = (remaining[position_col] - target).abs().idxmin()
        selected.append(index)
        remaining = remaining.drop(index)
        if remaining.empty:
            break
    return df.loc[selected]


def unique_concat(parts: list[pd.DataFrame], limit: int | None = None) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    joined = pd.concat([part for part in parts if not part.empty], axis=0) if any(not part.empty for part in parts) else pd.DataFrame(columns=parts[0].columns)
    if joined.empty:
        return joined
    joined = joined.drop_duplicates("chunk_id", keep="first")
    return joined.head(limit) if limit is not None else joined


def record(row: pd.Series, sample_type: str, target_effect: int, responsible_t: int | None) -> dict:
    start, end = int(row.start_t), int(row.end_t)
    result = {
        "task": row.task, "demo_id": row.demo_id, "pair_id": row.pair_id,
        "variant": row.variant, "base_demo_id": row.base_demo_id, "split": row.split,
        "chunk_id": row.chunk_id, "start_t": start, "end_t": end,
        "horizon_steps": int(row.horizon_steps), "V_c": float(row.V_c),
        "pred_rank": float(row.pred_rank), "expected_rank": float(row.expected_rank),
        "feature_path": row.feature_path, "sample_type": sample_type, "target_effect": int(target_effect),
        "target_center_t": np.nan, "target_region_start_t": np.nan, "target_region_end_t": np.nan,
    }
    if target_effect:
        assert responsible_t is not None and start <= responsible_t < end
        result.update({
            "target_center_t": int(responsible_t),
            "target_region_start_t": max(start, int(responsible_t) - 1),
            "target_region_end_t": min(end - 1, int(responsible_t) + 1),
        })
    result["sample_id"] = f"{row.task}|{row.demo_id}|{row.chunk_id}|{sample_type}"
    return result


def main() -> None:
    args = options()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    min_horizon = int(config["data"]["min_horizon"])
    positives_per_pair = int(config["data"]["positive_chunks_per_pair"])
    chunks = pd.read_parquet(args.chunk_evidence)
    labels = pd.read_parquet(args.transition_labels)
    features = pd.read_parquet(args.feature_index)
    labels = labels[labels["label_status"] != "ambiguous"]
    meta = labels[META_COLUMNS].drop_duplicates(["task", "demo_id"])
    chunks = chunks[chunks["horizon_steps"] >= min_horizon]
    # Stage 1 represents a clean sequence's evidence pair ID as
    # ``task:base_demo_id:clean`` but labels correctly retain pair_id=null.
    # Therefore identity is (task, demo_id), never pair_id, for clean rows.
    chunks = chunks.merge(
        meta.drop(columns=["pair_id"]),
        on=["task", "demo_id", "variant", "base_demo_id"],
        how="inner",
        validate="many_to_one",
    )
    chunks.loc[chunks["variant"] == "clean", "pair_id"] = pd.NA
    chunks = chunks.merge(features[["task", "demo_id", "feature_path"]], on=["task", "demo_id"], how="inner", validate="many_to_one")
    selected: list[dict] = []
    effective_meta = meta[(meta["variant"] == "perturbed") & meta["is_effective_intervention"].astype(bool)]
    for item in effective_meta.sort_values(["task", "demo_id"]).itertuples(index=False):
        demo = chunks[(chunks.task == item.task) & (chunks.demo_id == item.demo_id)]
        responsible_t = int(item.responsible_t)
        positive = demo[(demo.start_t <= responsible_t) & (responsible_t < demo.end_t)].copy()
        if positive.empty:
            raise RuntimeError(f"effective pair has no positive chunk: {item.demo_id}")
        positive["relative_responsible_position"] = (responsible_t - positive.start_t) / positive.horizon_steps
        parts: list[pd.DataFrame] = []
        if pd.notna(item.failure_onset):
            onset = int(item.failure_onset)
            parts.append(top(positive[(positive.start_t <= onset) & (onset < positive.end_t)], 4))
        parts.append(top(positive, 4))
        initial = unique_concat(parts, positives_per_pair)
        remaining = positive[~positive.chunk_id.isin(initial.chunk_id)]
        parts = [initial, spread_select(remaining, "relative_responsible_position", positives_per_pair - len(initial))]
        positive_selected = unique_concat(parts, positives_per_pair)
        selected.extend(record(row, "positive", 1, responsible_t) for _, row in positive_selected.iterrows())
        negative_parts: list[pd.DataFrame] = []
        outside = demo[~((demo.start_t <= responsible_t) & (responsible_t < demo.end_t))]
        before = outside[outside.end_t <= responsible_t].sort_values(["end_t", "V_c"], ascending=[False, False]).head(1)
        after = outside[outside.start_t > responsible_t].sort_values(["start_t", "V_c"], ascending=[True, False]).head(1)
        negative_parts.extend([before, after])
        if pd.notna(item.failure_onset):
            onset = int(item.failure_onset)
            negative_parts.append(top(outside[(outside.start_t <= onset) & (onset < outside.end_t)], 1))
        negative_parts.append(top(outside, 1))
        hard = unique_concat(negative_parts, int(config["data"]["hard_negative_chunks_per_effective_pair"]))
        selected.extend(record(row, "effective_hard_negative", 0, None) for _, row in hard.iterrows())
    controls = meta[(meta["variant"] == "perturbed") & (meta["failure_type"] == "no_effect") & meta["is_no_effect_negative_control"].astype(bool)]
    for item in controls.sort_values(["task", "demo_id"]).itertuples(index=False):
        demo = chunks[(chunks.task == item.task) & (chunks.demo_id == item.demo_id)]
        intervention_t = int(item.intervention_t)
        candidates = demo[(demo.start_t <= intervention_t) & (intervention_t < demo.end_t)].copy()
        candidates["relative_intervention_position"] = (intervention_t - candidates.start_t) / candidates.horizon_steps
        remaining = candidates.copy(); parts = []
        for target in (.35, .65):
            if remaining.empty:
                break
            chosen_index = (remaining.relative_intervention_position - target).abs().idxmin()
            parts.append(remaining.loc[[chosen_index]])
            remaining = remaining.drop(chosen_index)
        chosen = unique_concat(parts, 2)
        selected.extend(record(row, "no_effect_control", 0, None) for _, row in chosen.iterrows())
    clean_meta = meta[meta["variant"] == "clean"]
    for item in clean_meta.sort_values(["task", "demo_id"]).itertuples(index=False):
        demo = chunks[(chunks.task == item.task) & (chunks.demo_id == item.demo_id)].copy()
        values = np.quantile(demo.start_t.to_numpy(), [.10, .35, .65, .90])
        remaining = demo.copy(); chosen_rows: list[pd.Series] = []
        for value in values:
            if remaining.empty:
                break
            idx = (remaining.start_t - value).abs().idxmin()
            chosen_rows.append(remaining.loc[idx]); remaining = remaining.drop(idx)
        selected.extend(record(row, "clean_control", 0, None) for row in chosen_rows)
    samples = pd.DataFrame(selected)
    samples = samples.sort_values(["task", "demo_id", "sample_type", "chunk_id"]).reset_index(drop=True)
    samples["pair_sample_weight"] = 1.0 / samples.groupby(["task", "demo_id"])["sample_id"].transform("size")
    if samples.sample_id.duplicated().any():
        raise RuntimeError("duplicate sample id")
    found = samples[samples.target_effect == 1].groupby("split").demo_id.nunique().to_dict()
    expected = effective_meta.groupby("split").demo_id.nunique().to_dict()
    if found != expected:
        raise RuntimeError(f"effective pair coverage mismatch: expected {expected}, found {found}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(args.output, index=False)
    summary = {
        "seed": args.seed, "sample_count": int(len(samples)), "sample_type_counts": samples.sample_type.value_counts().sort_index().to_dict(),
        "split_sample_counts": samples.groupby("split").size().to_dict(), "positive_sample_count": int(samples.target_effect.sum()),
        "ambiguous_sample_count": 0, "effective_pair_coverage": {split: {"covered": int(found.get(split, 0)), "expected": int(expected.get(split, 0))} for split in sorted(expected)},
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
