"""Reference HB1 timing and success labels shared by execution and analysis."""
from __future__ import annotations

from typing import Any


BRANCH_SPECS = (
    ("none", 0),
    ("l5", 5),
    ("l20", 20),
    ("l80", 80),
    ("full", "full"),
)


def helper_active(repair_length: int | str, anchor_t: int, absolute_t: int) -> bool:
    if repair_length == "full":
        return True
    length = int(repair_length)
    return length > 0 and absolute_t < anchor_t + length


def apply_success_labels(result: dict[str, Any], minimum_autonomous_steps: int = 20) -> dict[str, Any]:
    length = result.get("repair_length")
    handoff_state = result.get("handoff_state_index")
    first_success = result.get("first_raw_success_state")
    finite_help = length not in (0, "full", None)
    handoff_executed = bool(result.get("handoff_executed"))
    system_success = bool(result.get("system_success"))
    success_under_helper = bool(result.get("success_seen_under_helper"))
    calls_after = int(result.get("repair_calls_after_handoff") or 0)
    autonomous = None
    if handoff_state is not None and first_success is not None:
        autonomous = int(first_success) - int(handoff_state)
    autonomous_execution = None
    if handoff_state is not None and result.get("episode_end_state_index") is not None:
        autonomous_execution = max(
            0, int(result["episode_end_state_index"]) - int(handoff_state)
        )
    raw = bool(
        finite_help
        and system_success
        and handoff_executed
        and handoff_state is not None
        and first_success is not None
        and int(first_success) > int(handoff_state)
    )
    genuine = bool(
        raw
        and not success_under_helper
        and autonomous is not None
        and autonomous >= int(minimum_autonomous_steps)
        and calls_after == 0
    )
    result.update(
        handoff_success_raw=raw,
        genuine_handoff_success=genuine,
        autonomous_steps_after_handoff=autonomous,
        first_success_wait_after_handoff=autonomous,
        autonomous_execution_steps_after_handoff=autonomous_execution,
    )
    return result


def diagnostic_shortest(branches: dict[str, dict[str, Any]]) -> str | int:
    none = branches.get("none", {})
    if bool(none.get("system_success")):
        return 0
    for length in (5, 20, 80):
        if bool(branches.get(f"l{length}", {}).get("genuine_handoff_success")):
            return length
    if bool(branches.get("full", {}).get("system_success")):
        return "helper_only"
    return "unresolved_by_this_pair_and_grid"


def fixture_cases() -> list[dict[str, Any]]:
    return [
        {"case": "A", "y0": True, "branch": "l5", "system_success": True,
         "handoff_executed": True, "handoff_state_index": 25,
         "first_raw_success_state": 60, "success_seen_under_helper": False},
        {"case": "B", "y0": False, "branch": "l20", "system_success": True,
         "handoff_executed": True, "handoff_state_index": 40,
         "first_raw_success_state": 80, "success_seen_under_helper": False},
        {"case": "C", "y0": False, "branch": "l20", "system_success": True,
         "handoff_executed": False, "handoff_state_index": None,
         "first_raw_success_state": 35, "success_seen_under_helper": True},
        {"case": "D", "y0": False, "branch": "l20", "system_success": True,
         "handoff_executed": True, "handoff_state_index": 40,
         "first_raw_success_state": 41, "success_seen_under_helper": False},
        {"case": "E", "y0": False, "branch": "l20", "system_success": True,
         "handoff_executed": True, "handoff_state_index": 40,
         "first_raw_success_state": 70, "success_seen_under_helper": False,
         "l80_system_success": False},
        {"case": "F", "y0": False, "branch": "full", "system_success": True,
         "handoff_executed": False, "handoff_state_index": None,
         "first_raw_success_state": 70, "success_seen_under_helper": True},
    ]
