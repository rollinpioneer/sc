"""Select one Stage 2 iteration checkpoint per run using validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATES = ("best_localization", "best_effect_bacc", "best_gate_gap")
FULL_RUNS = ("full_seed_0", "full_seed_1", "full_seed_2")
RUN_METHODS = {
    "full_seed_0": "responsibility_iter1_seed0",
    "full_seed_1": "responsibility_iter1_seed1",
    "full_seed_2": "responsibility_iter1_seed2",
    "action_only_seed_0": "action_only_iter1_seed0",
}
BASELINES = ("original_scizor", "uniform", "future_discount")


def value(metrics: dict, key: str, default: float) -> float:
    item = metrics.get(key)
    return default if item is None else float(item)


def candidate_record(methods: dict, normalized: str, candidate: str) -> dict:
    method = f"{normalized}__{candidate}"
    if method not in methods:
        raise KeyError(f"missing validation method {method}")
    return {"method": method, "candidate_name": candidate, "metrics": methods[method]["overall"]}


def choose_candidate(records: list[dict], baseline_iou: float, baseline_top1: float, tolerance: float) -> tuple[dict, bool]:
    floor = [record for record in records if value(record["metrics"], "responsibility_region_iou", float("-inf")) > baseline_iou and value(record["metrics"], "top1_within_1", float("-inf")) > baseline_top1]
    localization_floor_pass = bool(floor)
    pool = floor if floor else records
    min_far = min(value(record["metrics"], "no_effect_false_attribution_rate", float("inf")) for record in pool)
    pool = [record for record in pool if value(record["metrics"], "no_effect_false_attribution_rate", float("inf")) <= min_far + tolerance]
    selected = sorted(
        pool,
        key=lambda record: (
            -value(record["metrics"], "responsibility_region_iou", float("-inf")),
            -value(record["metrics"], "top1_within_1", float("-inf")),
            -value(record["metrics"], "recovery_retention", float("-inf")),
            value(record["metrics"], "mean_abs_localization_delay", float("inf")),
            record["candidate_name"],
        ),
    )[0]
    return selected, localization_floor_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--far-tolerance", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.validation_metrics.read_text(encoding="utf-8"))
    methods = report["methods"]
    baseline_metrics = [methods[name]["overall"] for name in BASELINES]
    baseline_iou = max(value(item, "responsibility_region_iou", float("-inf")) for item in baseline_metrics)
    baseline_top1 = max(value(item, "top1_within_1", float("-inf")) for item in baseline_metrics)

    runs = {}
    for run_name, normalized in RUN_METHODS.items():
        records = [candidate_record(methods, normalized, candidate) for candidate in CANDIDATES]
        selected, floor_pass = choose_candidate(records, baseline_iou, baseline_top1, args.far_tolerance)
        checkpoint = args.runs_root / run_name / f"{selected['candidate_name']}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        runs[run_name] = {
            "normalized_method": normalized,
            "selected_candidate_method": selected["method"],
            "candidate_name": selected["candidate_name"],
            "checkpoint": str(checkpoint.resolve()),
            "localization_floor_pass": floor_pass,
            "validation_metrics": selected["metrics"],
        }

    canonical_pool = [runs[name] for name in FULL_RUNS if runs[name]["localization_floor_pass"]]
    if not canonical_pool:
        canonical_pool = [runs[name] for name in FULL_RUNS]
    min_far = min(value(item["validation_metrics"], "no_effect_false_attribution_rate", float("inf")) for item in canonical_pool)
    canonical_pool = [item for item in canonical_pool if value(item["validation_metrics"], "no_effect_false_attribution_rate", float("inf")) <= min_far + args.far_tolerance]
    canonical = sorted(
        canonical_pool,
        key=lambda item: (
            -value(item["validation_metrics"], "responsibility_region_iou", float("-inf")),
            -value(item["validation_metrics"], "top1_within_1", float("-inf")),
            -value(item["validation_metrics"], "recovery_retention", float("-inf")),
            value(item["validation_metrics"], "mean_abs_localization_delay", float("inf")),
            int(item["normalized_method"].rsplit("seed", 1)[-1]),
        ),
    )[0]
    canonical_run = next(name for name, item in runs.items() if item is canonical)
    payload = {
        "selection_split": "validation",
        "far_tolerance": args.far_tolerance,
        "strongest_validation_baselines": {
            "methods": list(BASELINES),
            "responsibility_region_iou": baseline_iou,
            "top1_within_1": baseline_top1,
        },
        "runs": runs,
        "canonical": {
            "run_name": canonical_run,
            "seed": int(canonical["normalized_method"].rsplit("seed", 1)[-1]),
            "normalized_method": canonical["normalized_method"],
            "checkpoint": canonical["checkpoint"],
            "candidate_name": canonical["candidate_name"],
            "validation_metrics": canonical["validation_metrics"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
