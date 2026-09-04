"""Build a compact validation/blind leaderboard from frozen metric JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = ("pair_auroc", "pair_auprc", "no_effect_far", "effective_recall", "top1_within_1", "top5_hit", "region_iou", "mean_abs_delay", "recovery_false_attribution", "can_auroc", "square_auroc", "mean_candidates_per_pair")


def row(split: str, method: str, metrics: dict) -> dict:
    task = metrics.get("by_task", {})
    return {
        "split": split,
        "method": method,
        "pair_auroc": metrics.get("auroc"),
        "pair_auprc": metrics.get("auprc"),
        "no_effect_far": metrics.get("no_effect_far"),
        "effective_recall": metrics.get("effective_recall"),
        "top1_within_1": metrics.get("top1_within_1"),
        "top5_hit": metrics.get("top5_hit"),
        "region_iou": metrics.get("region_iou"),
        "mean_abs_delay": metrics.get("mean_abs_localization_delay", metrics.get("mean_abs_delay")),
        "recovery_false_attribution": metrics.get("recovery_false_attribution"),
        "can_auroc": task.get("can", {}).get("auroc"),
        "square_auroc": task.get("square", {}).get("auroc"),
        "mean_candidates_per_pair": metrics.get("mean_candidates_per_pair"),
    }


def add_sources(result: dict, split: str, selected: str | None, rows: list[dict]) -> None:
    sources = result.get("sources", result)
    for source, metrics in sources.items():
        rows.append(row(split, source, metrics))
    if selected and selected in sources:
        rows.append(row(split, "selected fused pipeline", sources[selected]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--blind", type=Path)
    parser.add_argument("--teacher-forced", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    blind = json.loads(args.blind.read_text(encoding="utf-8")) if args.blind and args.blind.exists() else None
    teacher = json.loads(args.teacher_forced.read_text(encoding="utf-8")) if args.teacher_forced and args.teacher_forced.exists() else None
    records: list[dict] = []
    add_sources(validation, "validation", validation.get("selected_proposer"), records)
    if blind is not None:
        add_sources(blind.get("all_sources", {}), "blind_test", blind.get("selected_proposer"), records)
    # The simulator primary feasible result is an upper-reference diagnostic,
    # not a learned pipeline score.
    if teacher is not None:
        primary = teacher.get("primary_effective_vs_no_effect", {})
        records.append(row("blind_test", "simulator primary feasible oracle (teacher-forced)", {"auroc": primary.get("auroc"), "auprc": primary.get("auprc"), "by_task": teacher.get("primary_by_task", {})}))
    frame = pd.DataFrame(records)
    for field in ("split", "method", *FIELDS):
        if field not in frame:
            frame[field] = None
    frame = frame[["split", "method", *FIELDS]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(frame), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
