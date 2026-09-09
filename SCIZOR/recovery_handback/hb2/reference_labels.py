"""HB-2 label and root-count reference checks from the experiment plan."""
from __future__ import annotations

import math
import unittest
from collections import defaultdict
from typing import Any, Iterable, Mapping

BRANCH_NAMES = ("none", "l5", "l20", "l80", "full")
FINITE_NAMES = ("l5", "l20", "l80")
FAILURE, HELPER_COMPLETED, OTHER_SUCCESS, GENUINE_HANDOFF = range(4)


def _boolean(row: Mapping[str, Any], key: str) -> bool:
    if key not in row:
        raise ValueError(f"missing label: {key}")
    value = row[key]
    if isinstance(value, str) or value is None:
        raise ValueError(f"invalid boolean label {key}: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid boolean label {key}: {value!r}") from exc
    if not math.isfinite(number) or number not in (0.0, 1.0):
        raise ValueError(f"invalid boolean label {key}: {value!r}")
    return bool(number)


def finite_category(row: Mapping[str, Any]) -> int:
    if row.get("branch_name") not in FINITE_NAMES:
        raise ValueError("finite_category requires l5, l20, or l80")
    if not _boolean(row, "engineering_ok"):
        raise ValueError("engineering failure is missing/invalid, not task failure")
    system = _boolean(row, "system_success")
    helper = _boolean(row, "helper_completed_task")
    genuine = _boolean(row, "genuine_handoff_success")
    if (helper or genuine) and not system:
        raise ValueError("success subtype without system_success")
    if helper and genuine:
        raise ValueError("helper completion and genuine handoff cannot both hold")
    if genuine:
        if _boolean(row, "success_seen_under_helper"):
            raise ValueError("genuine handoff cannot include helper-period success")
        if int(row.get("repair_calls_after_handoff", -1)) != 0:
            raise ValueError("genuine handoff requires zero post-handoff repair calls")
    if not system:
        return FAILURE
    if helper:
        return HELPER_COMPLETED
    if genuine:
        return GENUINE_HANDOFF
    return OTHER_SUCCESS


def anchor_outcomes(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    indexed: dict[str, Mapping[str, Any]] = {}
    anchor_ids = set()
    for row in rows:
        name = row.get("branch_name")
        if name not in BRANCH_NAMES:
            raise ValueError(f"unexpected branch: {name!r}")
        if name in indexed:
            raise ValueError(f"duplicate branch: {name}")
        anchor_ids.add(row.get("anchor_id"))
        if not _boolean(row, "engineering_ok"):
            raise ValueError("incomplete engineering record")
        indexed[name] = row
    if len(anchor_ids) != 1 or None in anchor_ids:
        raise ValueError("expected exactly one non-null anchor_id")
    if set(indexed) != set(BRANCH_NAMES):
        raise ValueError("incomplete paired anchor")
    y0 = int(_boolean(indexed["none"], "system_success"))
    output: dict[str, Any] = {
        "anchor_id": next(iter(anchor_ids)),
        "y0": y0,
        "y_full": int(_boolean(indexed["full"], "system_success")),
    }
    for name in FINITE_NAMES:
        category = finite_category(indexed[name])
        y_sys = int(category != FAILURE)
        y_genuine = int(category == GENUINE_HANDOFF)
        output.update({
            f"category_{name}": category,
            f"y_sys_{name}": y_sys,
            f"y_genuine_{name}": y_genuine,
            f"gain_sys_{name}": y_sys - y0,
            f"gain_autonomy_{name}": y_genuine - y0,
            f"rescue_{name}": int(not y0 and y_genuine),
            f"harm_{name}": int(y0 and not y_sys),
        })
    return output


def full_completion_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("branch_name") != "full":
            continue
        if not _boolean(row, "engineering_ok"):
            raise ValueError("invalid full branch must be reported, not discarded")
        key = row.get("anchor_id")
        if key is None or key in seen:
            raise ValueError("missing or duplicate full anchor_id")
        seen.add(key)
        group = row.get("stat_group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("global stat_group_id required")
        completed = _boolean(row, "helper_completed_task")
        if completed and not _boolean(row, "system_success"):
            raise ValueError("helper completion requires system success")
        groups[group].append(completed)
    if not groups:
        return {"anchors": 0, "roots": 0, "helper_completed_anchors": 0,
                "unique_helper_completed_roots": 0, "anchor_rate": None,
                "mean_within_root_rate": None}
    count = sum(map(len, groups.values()))
    successes = sum(sum(values) for values in groups.values())
    return {
        "anchors": count,
        "roots": len(groups),
        "helper_completed_anchors": successes,
        "unique_helper_completed_roots": sum(any(values) for values in groups.values()),
        "anchor_rate": successes / count,
        "mean_within_root_rate": sum(sum(v) / len(v) for v in groups.values()) / len(groups),
    }


class ReferenceTests(unittest.TestCase):
    def row(self, branch: str, system: bool = False, *, genuine: bool = False,
            helper: bool = False, root: str = "r1", anchor: str = "r1:20") -> dict:
        return {"branch_name": branch, "engineering_ok": True,
                "system_success": system, "helper_completed_task": helper,
                "genuine_handoff_success": genuine, "success_seen_under_helper": helper,
                "repair_calls_after_handoff": 0, "stat_group_id": root,
                "anchor_id": anchor}

    def test_four_categories(self):
        self.assertEqual(finite_category(self.row("l5")), FAILURE)
        self.assertEqual(finite_category(self.row("l5", True, helper=True)), HELPER_COMPLETED)
        self.assertEqual(finite_category(self.row("l5", True)), OTHER_SUCCESS)
        self.assertEqual(finite_category(self.row("l5", True, genuine=True)), GENUINE_HANDOFF)

    def test_negative_gain_kept(self):
        result = anchor_outcomes([self.row(name, name == "none") for name in BRANCH_NAMES])
        self.assertEqual(result["gain_autonomy_l5"], -1)
        self.assertEqual(result["harm_l5"], 1)

    def test_nonmonotonic_success_allowed(self):
        result = anchor_outcomes([
            self.row(name, name == "l5", genuine=name == "l5") for name in BRANCH_NAMES
        ])
        self.assertEqual(result["rescue_l5"], 1)
        self.assertEqual(result["rescue_l80"], 0)

    def test_full_and_fast_success_are_not_genuine(self):
        result = anchor_outcomes([
            self.row("none"), self.row("l5", True), self.row("l20"),
            self.row("l80", True, helper=True), self.row("full", True, helper=True),
        ])
        self.assertEqual(result["y_full"], 1)
        self.assertEqual(result["rescue_l5"], 0)
        self.assertEqual(result["rescue_l80"], 0)

    def test_missing_not_failure(self):
        with self.assertRaises(ValueError):
            anchor_outcomes([self.row("none")])
        invalid = self.row("l5")
        invalid["engineering_ok"] = False
        with self.assertRaises(ValueError):
            finite_category(invalid)

    def test_distinct_roots_not_rate_times_count(self):
        rows = []
        for root, results in (("r1", (True, False, False)), ("r2", (True, False, False))):
            for index, success in enumerate(results):
                rows.append(self.row("full", success, helper=success,
                                     root=root, anchor=f"{root}:{index}"))
        summary = full_completion_summary(rows)
        self.assertEqual(summary["unique_helper_completed_roots"], 2)
        self.assertAlmostEqual(summary["anchor_rate"], 1 / 3)
        self.assertNotEqual(summary["anchor_rate"] * summary["roots"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

