"""Derive the bounded HB1-R decision from capability and formal-probe evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump


ALLOWED_OVERALL = {
    "READY_HB2",
    "READY_HB2_SINGLE_TASK",
    "HOLD_CAPABILITY_REPAIR",
    "NO_LOCAL_HANDOFF_EVIDENCE_FOR_CURRENT_PAIR_AND_GRID",
}


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_rescues(summary: dict | None) -> int:
    if not summary:
        return 0
    selection = summary.get("selection", {})
    return int(selection.get("base_failed_repair_succeeded_roots", 0) or 0)


def _candidate_successes(summary: dict | None) -> int:
    if not summary:
        return 0
    selection = summary.get("selection", {})
    return int(selection.get("successes", 0) or 0)


def _formal_task(formal: dict | None, task: str) -> dict | None:
    if not formal:
        return None
    value = formal.get("tasks", {}).get(task)
    return value if isinstance(value, dict) else None


def build_decision(root: Path) -> dict:
    direct = _read(root / "square/teacher/evaluation/qualification/summary.json")
    residual = _read(root / "square/repair_rl/qualification/summary.json")
    can_lowdim = _read(root / "can/diagnostic/evaluation/summary.json")
    can_visual = _read(root / "can/visual/evaluation/summary.json")
    formal = _read(root / "metrics/hb1_decision.json")
    coverage = _read(root / "metrics/coverage.json")

    square_pair = (root / "assets/policy_pair_square.json").is_file()
    can_base = (root / "assets/base_policy_can.json").is_file()
    square_formal = _formal_task(formal, "square")
    can_formal = _formal_task(formal, "can")

    if square_formal and square_formal.get("status") in {
        "READY_HB2",
        "NO_LOCAL_HANDOFF_EVIDENCE_FOR_CURRENT_PAIR_AND_GRID",
    }:
        square_status = "READY"
    elif residual and residual.get("status") != "QUALIFIED":
        square_status = "NEED_STRONGER_REPAIRER_AFTER_HB1_R"
    elif direct and direct.get("status") == "NEED_STRONGER_REPAIRER" and not residual:
        square_status = "NOT_RUN"
    elif square_pair:
        square_status = "READY"
    elif direct or residual:
        square_status = "NEED_STRONGER_REPAIRER_AFTER_HB1_R"
    else:
        square_status = "NOT_RUN"

    if can_formal and can_formal.get("status") == "READY_HB2":
        can_status = "READY"
    elif can_base or (can_visual and can_visual.get("status") == "QUALIFIED"):
        can_status = "READY"
    elif can_visual:
        can_status = "NEED_BASE_POLICY_AFTER_HB1_R"
    elif can_lowdim and can_lowdim.get("status") == "HOLD_ENGINEERING_FIX":
        can_status = "HOLD_CAN_TRAINING_PIPELINE"
    elif can_lowdim and can_lowdim.get("status") != "QUALIFIED":
        can_status = "NEED_BASE_POLICY_AFTER_HB1_R"
    else:
        can_status = "NOT_RUN"

    formal_status = formal.get("status") if formal else None
    formal_task_statuses = {
        task: details.get("status")
        for task, details in (formal or {}).get("tasks", {}).items()
        if isinstance(details, dict)
    }
    if formal_status in {"READY_HB2", "READY_HB2_SINGLE_TASK"}:
        overall = formal_status
    elif "NO_LOCAL_HANDOFF_EVIDENCE_FOR_CURRENT_PAIR_AND_GRID" in formal_task_statuses.values():
        overall = "NO_LOCAL_HANDOFF_EVIDENCE_FOR_CURRENT_PAIR_AND_GRID"
    else:
        overall = "HOLD_CAPABILITY_REPAIR"
    if overall not in ALLOWED_OVERALL:
        raise RuntimeError(f"invalid HB1-R overall decision: {overall}")

    ready_tasks = list((formal or {}).get("ready_tasks", []))
    return {
        "schema_version": "hb1_capability_repair_decision_v1",
        "status": overall,
        "ready_tasks": ready_tasks,
        "tasks": {
            "square": {
                "status": square_status,
                "direct_teacher_status": (direct or {}).get("status", "NOT_RUN"),
                "direct_teacher_rescued_roots": _selected_rescues(direct),
                "teacher_residual_status": (residual or {}).get("status", "NOT_RUN"),
                "teacher_residual_rescued_roots": _selected_rescues(residual),
                "policy_pair_frozen": square_pair,
                "formal_probe": square_formal or {"status": "NOT_RUN"},
            },
            "can": {
                "status": can_status,
                "lowdim_status": (can_lowdim or {}).get("status", "NOT_RUN"),
                "lowdim_base_val_successes": _candidate_successes(can_lowdim),
                "visual_status": (can_visual or {}).get("status", "NOT_RUN"),
                "visual_base_val_successes": _candidate_successes(can_visual),
                "base_policy_frozen": can_base,
                "formal_probe": can_formal or {"status": "NOT_RUN"},
            },
        },
        "formal_hb1_status": formal_status or "NOT_RUN",
        "coverage": coverage or {
            "authoritative_anchors": 0,
            "complete_anchors": 0,
            "coverage_fraction": 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = build_decision(args.repair_root.resolve())
    atomic_json_dump(decision, args.output)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
