"""Evaluate frozen HB3-P methods on one complete episode per independent root."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, sha256_file
from recovery_handback.hb3p.io import write_csv
from recovery_handback.hb3p.metrics import paired_bootstrap


PAIRS = (
    ("M1_GRID", "S_STAR"),
    ("M1_GRID", "NONE"),
    ("M1_GRID", "M0_GRID"),
    ("M4_GRID", "S_STAR"),
    ("M4_GRID", "M1_GRID"),
    ("S_STAR", "NONE"),
)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _root_map(rows: list[dict], key: str) -> dict[str, float]:
    return {str(row["stat_group_id"]): float(row[key]) for row in rows}


def _list_value(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return list(json.loads(value))
    return list(value)


def _fourfold(left: list[dict], right: list[dict], key: str) -> dict:
    left_map = _root_map(left, key)
    right_map = _root_map(right, key)
    if set(left_map) != set(right_map):
        raise ValueError("paired outcome table has mismatched roots")
    counts = Counter((bool(left_map[root]), bool(right_map[root])) for root in left_map)
    return {
        "both_success": counts[(True, True)],
        "left_only": counts[(True, False)],
        "right_only": counts[(False, True)],
        "both_failure": counts[(False, False)],
    }


def _method_summary(method: str, rows: list[dict], baseline: dict[str, dict]) -> tuple[dict, dict, dict]:
    if len(rows) != len(baseline) or len({row["stat_group_id"] for row in rows}) != len(rows):
        raise ValueError(f"method does not have one record per root: {method}")
    rescues = sum(
        not bool(baseline[str(row["stat_group_id"])]["system_success"])
        and int(row["takeover_count"]) > 0 and bool(row["genuine_handoff_success"])
        for row in rows
    )
    baseline_success = sum(bool(row["system_success"]) for row in baseline.values())
    helped_baseline_success = sum(
        bool(baseline[str(row["stat_group_id"])]["system_success"])
        and int(row["takeover_count"]) > 0 for row in rows
    )
    interference = sum(
        bool(baseline[str(row["stat_group_id"])]["system_success"])
        and not bool(row["system_success"]) for row in rows
    )
    retained = sum(
        bool(baseline[str(row["stat_group_id"])]["system_success"])
        and bool(row["system_success"]) for row in rows
    )
    summary = {
        "method_id": method,
        "valid_roots": len(rows),
        "system_success_count": sum(bool(row["system_success"]) for row in rows),
        "system_success_rate": _mean(rows, "system_success"),
        "autonomous_completion_count": sum(bool(row["autonomous_completion"]) for row in rows),
        "autonomous_completion_rate": _mean(rows, "autonomous_completion"),
        "primary_utility_U_lambda_0p25": _mean(rows, "utility"),
        "genuine_rescued_roots": int(rescues),
        "retained_baseline_success_roots": int(retained),
        "interference_roots": int(interference),
        "mean_helper_steps_actual": _mean(rows, "helper_steps_actual"),
        "mean_changed_action_steps": _mean(rows, "changed_action_steps"),
        "mean_takeover_count": _mean(rows, "takeover_count"),
        "mean_query_count": _mean(rows, "query_count"),
        "repair_calls_after_handoff_total": sum(int(row["repair_calls_after_handoff"]) for row in rows),
    }
    interference_row = {
        "method_id": method,
        "count": int(interference),
        "all_baseline_success_roots": int(baseline_success),
        "all_baseline_success_rate": interference / baseline_success if baseline_success else None,
        "helped_baseline_success_roots": int(helped_baseline_success),
        "helped_baseline_success_rate": interference / helped_baseline_success if helped_baseline_success else None,
        "small_sample": baseline_success < 5,
    }
    cost = {
        "method_id": method,
        "mean_helper_steps_actual": _mean(rows, "helper_steps_actual"),
        "mean_changed_action_steps": _mean(rows, "changed_action_steps"),
        "mean_takeover_count": _mean(rows, "takeover_count"),
        "mean_query_count": _mean(rows, "query_count"),
        "mean_inference_wall_seconds": _mean(rows, "inference_wall_seconds"),
        "mean_backbone_wall_seconds": _mean(rows, "backbone_wall_seconds"),
        "mean_prediction_head_wall_seconds": _mean(rows, "prediction_head_wall_seconds"),
        "mean_ipc_roundtrip_wall_seconds": _mean(rows, "ipc_roundtrip_wall_seconds"),
        "mean_simulation_wall_seconds": _mean(rows, "simulation_wall_seconds"),
        "mean_episode_wall_seconds": _mean(rows, "wall_seconds"),
        "repair_calls_after_handoff_total": summary["repair_calls_after_handoff_total"],
    }
    return summary, interference_row, cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--episode-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(args.protocol)
    rows = read_table(args.episode_table)
    grouped = defaultdict(list)
    for row in rows:
        if row.get("protocol_hash") != protocol_hash or not row.get("engineering_ok"):
            raise RuntimeError("evaluation input includes failed or wrong-protocol episodes")
        grouped[str(row["method_id"])].append(row)
    expected_methods = set(protocol["methods"])
    if set(grouped) != expected_methods:
        raise RuntimeError(f"logical method coverage mismatch: {sorted(grouped)}")
    if any(len(values) != 80 for values in grouped.values()):
        raise RuntimeError("every logical method must cover exactly 80 roots")
    baseline = {str(row["stat_group_id"]): row for row in grouped["NONE"]}

    methods, interference, costs = [], [], []
    for method in protocol["methods"]:
        summary, interference_row, cost = _method_summary(method, grouped[method], baseline)
        methods.append(summary)
        interference.append(interference_row)
        costs.append(cost)
    method_by_id = {row["method_id"]: row for row in methods}

    comparisons = {}
    outcome_rows = []
    for left, right in PAIRS:
        key = f"{left}_minus_{right}"
        comparisons[key] = {
            "left": left,
            "right": right,
            "utility": paired_bootstrap(
                _root_map(grouped[left], "utility"),
                _root_map(grouped[right], "utility"),
                int(protocol["statistics"]["bootstrap_repeats"]),
                int(protocol["statistics"]["seed"]),
            ),
            "system_success": _fourfold(grouped[left], grouped[right], "system_success"),
            "autonomous_completion": _fourfold(grouped[left], grouped[right], "autonomous_completion"),
        }
        for outcome in ("system_success", "autonomous_completion"):
            outcome_rows.append({"comparison": key, "outcome": outcome, **comparisons[key][outcome]})

    decision_rows = []
    for method, values in grouped.items():
        takeover_counts = Counter((row.get("takeover_t"), int(row["selected_length"])) for row in values)
        for (absolute_t, length), count in sorted(takeover_counts.items(), key=lambda item: (item[0][0] is None, item[0][0] or -1, item[0][1])):
            decision_rows.append({
                "method_id": method, "event": "takeover_choice",
                "absolute_t": absolute_t, "selected_length": length, "count": count,
            })
        query_counts = Counter(int(t) for row in values for t in _list_value(row["queried_times"]))
        for absolute_t, count in sorted(query_counts.items()):
            decision_rows.append({
                "method_id": method, "event": "query",
                "absolute_t": absolute_t, "selected_length": None, "count": count,
            })

    primary = comparisons["M1_GRID_minus_S_STAR"]["utility"]
    m1 = method_by_id["M1_GRID"]
    scheduled = method_by_id["S_STAR"]
    none = method_by_id["NONE"]
    if any(row["repair_calls_after_handoff_total"] for row in methods):
        status = "HOLD_HB3P_ENGINEERING_OR_INPUT"
    elif (
        primary["point_difference"] > 0 and primary["ci95_percentile"][0] > 0
        and m1["genuine_rescued_roots"] >= 3
        and m1["system_success_rate"] >= scheduled["system_success_rate"]
    ):
        status = "STATE_ENTRY_GAIN_SUPPORTED"
    elif (
        primary["point_difference"] > 0 and m1["genuine_rescued_roots"] >= 3
        and m1["system_success_rate"] >= scheduled["system_success_rate"]
    ):
        status = "STATE_ENTRY_SIGNAL_ONLY"
    elif primary["point_difference"] > 0 and m1["system_success_rate"] < scheduled["system_success_rate"]:
        status = "UTILITY_GAIN_WITH_SUCCESS_TRADEOFF"
    elif scheduled["primary_utility_U_lambda_0p25"] > none["primary_utility_U_lambda_0p25"]:
        status = "TIME_ONLY_BASELINE_RETAINED"
    else:
        status = "NO_EPISODE_LEVEL_GAIN_CURRENT_GRID"
    s_vs_none = comparisons["S_STAR_minus_NONE"]["utility"]
    m4_vs_m1 = comparisons["M4_GRID_minus_M1_GRID"]["utility"]
    m4_vs_s = comparisons["M4_GRID_minus_S_STAR"]["utility"]
    decision = {
        "schema_version": "hb3p_decision_v1",
        "hb3p_status": status,
        "hb2_status_preserved": "HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN",
        "stage": "HB3-P", "task": "square",
        "scope": "fixed_candidate_times_single_takeover_fixed_exit",
        "state_entry_gain": "supported" if status == "STATE_ENTRY_GAIN_SUPPORTED" else (
            "directional" if status == "STATE_ENTRY_SIGNAL_ONLY" else "unproven"
        ),
        "time_schedule_gain": "supported" if s_vs_none["ci95_percentile"][0] > 0 else (
            "directional" if s_vs_none["point_difference"] > 0 else "unproven"
        ),
        "visual_increment": "secondary_evidence" if (
            m4_vs_m1["point_difference"] > 0 and m4_vs_s["point_difference"] > 0
        ) else "unproven",
        "learned_exit_enabled": False,
        "repeated_takeover_tested": False,
        "arbitrary_query_times_tested": False,
        "baseline_preservation_evidence": "adequate" if method_by_id["NONE"]["system_success_count"] >= 5 else "small_sample",
        "primary_method": "M1_GRID", "primary_comparator": "S_STAR",
        "primary_comparison": primary,
        "methods": method_by_id,
        "visual_secondary_comparisons": {
            "M4_minus_M1": m4_vs_m1,
            "M4_minus_S_STAR": m4_vs_s,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(methods, args.output_dir / "methods.csv")
    write_csv(outcome_rows, args.output_dir / "paired_outcome_counts.csv")
    write_csv(interference, args.output_dir / "interference.csv")
    write_csv(costs, args.output_dir / "costs.csv")
    write_csv(decision_rows, args.output_dir / "decision_time_counts.csv")
    atomic_json_dump(comparisons, args.output_dir / "paired_comparisons.json")
    atomic_json_dump(decision, args.output_dir.parent / "hb3p_decision.json")
    print(json.dumps({"status": status, "primary": primary, "methods": len(methods)}, indent=2))


if __name__ == "__main__":
    main()
