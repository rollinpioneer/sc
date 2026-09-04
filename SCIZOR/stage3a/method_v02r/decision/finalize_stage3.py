"""Apply the frozen Stage 3H decision gates without tuning any parameter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


GO = "GO_STAGE4_POLICY_VALIDATION"
SWITCH = "SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR"


def value(mapping: dict, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def at_least(item, minimum):
    return item is not None and float(item) >= float(minimum)


def at_most(item, maximum):
    return item is not None and float(item) <= float(maximum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--blind-pipeline", type=Path)
    parser.add_argument("--blind-teacher-forced", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    confirmation = json.loads(args.confirmation.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    failed: list[str] = []
    checks: dict[str, bool] = {}

    checks["confirmation_engineering_pass"] = bool(confirmation.get("engineering_pass"))
    checks["confirmation_method_pass"] = bool(confirmation.get("method_pass"))
    checks["validation_gate_pass"] = bool(validation.get("validation_gate_pass"))
    if not checks["confirmation_engineering_pass"]:
        failed.append("confirmation_engineering")
    if not checks["confirmation_method_pass"]:
        failed.append("confirmation_method")
    if not checks["validation_gate_pass"]:
        failed.append("validation_gate")

    blind_pipeline = None
    blind_teacher = None
    if not args.validation_only:
        if args.blind_pipeline is None or args.blind_teacher_forced is None:
            raise ValueError("blind metrics are required unless --validation-only is set")
        blind_pipeline = json.loads(args.blind_pipeline.read_text(encoding="utf-8"))
        blind_teacher = json.loads(args.blind_teacher_forced.read_text(encoding="utf-8"))
        pair = blind_pipeline.get("pair_metrics", {})
        engineering = blind_teacher.get("engineering", {})
        teacher_positive = blind_teacher.get("replacement_oracle_positive", {})
        primary = blind_teacher.get("primary_effective_vs_no_effect", {})
        bcfg = config["blind_test"]
        checks.update({
            "blind_benchmark_check": bool(blind_pipeline.get("benchmark_check_pass")),
            "blind_branch_pre_state_equal": at_least(engineering.get("branch_pre_state_equal_rate"), 1.0),
            "blind_reference_exact": at_least(engineering.get("reference_exact_rate"), 0.999),
            "blind_finite_target": at_least(engineering.get("finite_target_rate"), 0.99),
            "blind_teacher_forced_oracle_auroc": at_least(teacher_positive.get("auroc"), 0.70),
            "blind_primary_effective_auroc": at_least(primary.get("auroc"), 0.70),
            "blind_pair_auroc": at_least(pair.get("auroc"), 0.70),
            "blind_pair_auprc": at_least(pair.get("auprc"), 2.0 * float(pair.get("prevalence") or 0.0)),
            "blind_no_effect_far": at_most(pair.get("no_effect_far"), bcfg["max_no_effect_far"]),
            "blind_effective_recall": at_least(pair.get("effective_recall"), bcfg["min_effective_recall"]),
            "blind_top1_within_1": at_least(pair.get("top1_within_1"), bcfg["min_top1_within_1"]),
            "blind_can_auroc": at_least(value(pair, "by_task", "can", "auroc"), bcfg["min_task_pair_auroc"]),
            "blind_square_auroc": at_least(value(pair, "by_task", "square", "auroc"), bcfg["min_task_pair_auroc"]),
        })
        for name, passed in checks.items():
            if not passed and name not in {"confirmation_engineering_pass", "confirmation_method_pass", "validation_gate_pass"}:
                failed.append(name)

    decision = GO if not failed else SWITCH
    payload = {
        "schema": "stage3_v02r_final_decision_v1",
        "decision": decision,
        "validation_only": bool(args.validation_only),
        "checks": checks,
        "failed_rules": failed,
        "selected_proposer": validation.get("selected_proposer"),
        "selected_threshold": validation.get("selected_threshold"),
        "confirmation_decision": confirmation.get("decision"),
    }
    if blind_pipeline is not None:
        payload["blind_pipeline_summary"] = {"benchmark_check_pass": blind_pipeline.get("benchmark_check_pass"), "selected_proposer": blind_pipeline.get("selected_proposer"), "selected_threshold": blind_pipeline.get("selected_threshold"), "pair_metrics": blind_pipeline.get("pair_metrics")}
    if blind_teacher is not None:
        payload["blind_teacher_forced_summary"] = blind_teacher
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
