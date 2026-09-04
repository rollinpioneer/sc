"""Summarize frozen full/action proposer transfer on replay-locked v0.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = {
    "full_top5": ("in_full_top5", "full_rank"),
    "action_top5": ("in_action_top5", "action_rank"),
    "union_top5": ("in_union_top5", "union_rank"),
}


def summarize(frame: pd.DataFrame, mask_column: str, rank_column: str) -> dict:
    data = frame[frame[mask_column].fillna(False)].copy()
    pairs = frame.drop_duplicates("pair_id")[["pair_id", "task", "is_effective_intervention", "responsible_t"]].copy()
    grouped = data.groupby("pair_id", sort=False)
    rows = []
    for pair in pairs.itertuples(index=False):
        candidates = grouped.get_group(pair.pair_id) if pair.pair_id in grouped.groups else data.iloc[:0]
        rank = candidates[rank_column].astype(float) if len(candidates) else pd.Series(dtype=float)
        top = candidates.loc[rank.idxmin()] if len(candidates) else None
        region = bool(candidates.is_responsibility_region.any()) if pair.is_effective_intervention and len(candidates) else False
        rows.append({"pair_id": pair.pair_id, "task": pair.task, "effective": bool(pair.is_effective_intervention), "count": len(candidates),
                     "region_hit": region, "top1_within_1": bool(top is not None and pair.is_effective_intervention and abs(int(top.t) - int(pair.responsible_t)) <= 1),
                     "min_abs_delay": float(np.min(np.abs(candidates.t.astype(int) - int(pair.responsible_t)))) if pair.is_effective_intervention and len(candidates) else None})
    per = pd.DataFrame(rows)
    def stats(part: pd.DataFrame) -> dict:
        effective = part[part.effective]
        return {
            "pair_count": int(len(part)), "effective_pair_count": int(len(effective)),
            "mean_candidates_per_pair": float(part["count"].mean()) if len(part) else None,
            "responsibility_region_recall": float(effective.region_hit.mean()) if len(effective) else None,
            "top1_within_1": float(effective.top1_within_1.mean()) if len(effective) else None,
            "top5_hit": float(effective.region_hit.mean()) if len(effective) else None,
            "mean_abs_localization_delay": float(effective.min_abs_delay.dropna().mean()) if effective.min_abs_delay.notna().any() else None,
        }
    return {"overall": stats(per), "by_task": {task: stats(per[per.task.eq(task)]) for task in ("can", "square")}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = {}
    for split, path in (("train", a.train), ("validation", a.validation)):
        frame = pd.read_parquet(path)
        if set(frame.split.unique()) != {split}:
            raise ValueError(f"unexpected split in {path}")
        result[split] = {source: summarize(frame, *columns) for source, columns in SOURCES.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
