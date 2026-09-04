"""Apply the fixed Stage 2 iteration test decision rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINES = ("original_scizor", "uniform", "future_discount")
FULL_METHODS = ("responsibility_iter1_seed0", "responsibility_iter1_seed1", "responsibility_iter1_seed2")


def metric(metrics: dict, key: str, default: float) -> float:
    value = metrics.get(key)
    return default if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--canonical-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.test_metrics.read_text(encoding="utf-8"))
    methods = report["methods"]
    canonical = json.loads(args.canonical_seed.read_text(encoding="utf-8"))
    canonical_method = canonical["canonical_method"]
    if canonical_method not in methods:
        raise KeyError(f"canonical method missing from test metrics: {canonical_method}")

    baseline = [methods[name]["overall"] for name in BASELINES]
    iou_baseline = max(metric(item, "responsibility_region_iou", float("-inf")) for item in baseline)
    top1_baseline = max(metric(item, "top1_within_1", float("-inf")) for item in baseline)
    delay_baseline = min(metric(item, "mean_abs_localization_delay", float("inf")) for item in baseline)
    recovery_baseline = max(metric(item, "recovery_retention", float("-inf")) for item in baseline)
    far_baseline = min(metric(item, "no_effect_false_attribution_rate", float("inf")) for item in baseline)
    far_limit = far_baseline + 0.03
    full = [methods[name]["overall"] for name in FULL_METHODS]
    canonical_metrics = methods[canonical_method]["overall"]
    action_metrics = methods["action_only_iter1_seed0"]["overall"]
    full_iou_count = sum(metric(item, "responsibility_region_iou", float("-inf")) > iou_baseline for item in full)
    full_top1_count = sum(metric(item, "top1_within_1", float("-inf")) > top1_baseline for item in full)
    rules = {
        "full_iou_count_at_least_2": {
            "passed": full_iou_count >= 2,
            "actual": full_iou_count,
            "threshold": 2,
            "strongest_baseline": iou_baseline,
        },
        "full_top1_count_at_least_2": {
            "passed": full_top1_count >= 2,
            "actual": full_top1_count,
            "threshold": 2,
            "strongest_baseline": top1_baseline,
        },
        "canonical_delay_below_baseline": {
            "passed": metric(canonical_metrics, "mean_abs_localization_delay", float("inf")) < delay_baseline,
            "actual": metric(canonical_metrics, "mean_abs_localization_delay", float("inf")),
            "threshold": delay_baseline,
        },
        "canonical_recovery_within_tolerance": {
            "passed": metric(canonical_metrics, "recovery_retention", float("-inf")) >= recovery_baseline - 0.10,
            "actual": metric(canonical_metrics, "recovery_retention", float("-inf")),
            "threshold": recovery_baseline - 0.10,
            "baseline": recovery_baseline,
        },
        "canonical_no_effect_far_within_limit": {
            "passed": metric(canonical_metrics, "no_effect_false_attribution_rate", float("inf")) <= far_limit,
            "actual": metric(canonical_metrics, "no_effect_false_attribution_rate", float("inf")),
            "threshold": far_limit,
            "baseline": far_baseline,
        },
        "canonical_beats_action_only": {
            "passed": (
                metric(canonical_metrics, "responsibility_region_iou", float("-inf")) > metric(action_metrics, "responsibility_region_iou", float("-inf"))
                or metric(canonical_metrics, "top1_within_1", float("-inf")) > metric(action_metrics, "top1_within_1", float("-inf"))
                or metric(canonical_metrics, "no_effect_false_attribution_rate", float("inf")) <= metric(action_metrics, "no_effect_false_attribution_rate", float("inf")) - 0.02
            ),
            "actual": {
                "canonical_iou": metric(canonical_metrics, "responsibility_region_iou", float("-inf")),
                "action_only_iou": metric(action_metrics, "responsibility_region_iou", float("-inf")),
                "canonical_top1": metric(canonical_metrics, "top1_within_1", float("-inf")),
                "action_only_top1": metric(action_metrics, "top1_within_1", float("-inf")),
                "canonical_no_effect_far": metric(canonical_metrics, "no_effect_false_attribution_rate", float("inf")),
                "action_only_no_effect_far": metric(action_metrics, "no_effect_false_attribution_rate", float("inf")),
            },
            "threshold": "any canonical advantage (FAR margin >= 0.02)",
        },
    }
    failed = [name for name, item in rules.items() if not item["passed"]]
    payload = {
        "decision": "GO_STAGE3" if not failed else "NO_GO_SWITCH_DIRECTION",
        "canonical_method": canonical_method,
        "canonical_seed": canonical["canonical_seed"],
        "baselines": {
            "methods": list(BASELINES),
            "iou_baseline": iou_baseline,
            "top1_baseline": top1_baseline,
            "delay_baseline": delay_baseline,
            "recovery_baseline": recovery_baseline,
            "no_effect_far_baseline": far_baseline,
            "no_effect_far_limit": far_limit,
        },
        "rules": rules,
        "failed_rules": failed,
        "test_metrics_file": str(args.test_metrics.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
