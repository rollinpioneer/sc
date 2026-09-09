"""Evaluate the bounded fixed-entry exit probe without training a selector."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, sha256_array, sha256_file
from recovery_handback.hb3p.io import write_csv
from recovery_handback.hb3p.metrics import paired_bootstrap


METHODS = ("NONE", "FIXED_L40", "FIXED_L60", "FIXED_L80")
LENGTHS = {"NONE": 0, "FIXED_L40": 40, "FIXED_L60": 60, "FIXED_L80": 80}
SHORT = ("FIXED_L40", "FIXED_L60")


def _pair_case(stop: bool, continue_: bool) -> str:
    if stop and continue_:
        return "stop_success_continue_success"
    if not stop and continue_:
        return "stop_failure_continue_success"
    if stop and not continue_:
        return "stop_success_continue_failure"
    return "both_failure"


def _oracle_component(oracle: dict, fixed: dict) -> str:
    delta = float(oracle["utility"]) - float(fixed["utility"])
    if abs(delta) <= 1e-12:
        return "no_gain"
    oracle_success = bool(oracle["autonomous_completion"])
    fixed_success = bool(fixed["autonomous_completion"])
    if oracle_success and fixed_success:
        return "success_with_lower_cost" if delta > 0 else "success_with_higher_cost"
    if oracle_success and not fixed_success:
        return "success_improvement"
    if not oracle_success and not fixed_success:
        return "both_failed_cost_only"
    return "failure_with_lower_cost" if delta > 0 else "fixed_success_better"


def _prefix_key(row: dict) -> tuple[int, str, str | None]:
    path = Path(str(row.get("trajectory_path", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        states = np.asarray(archive["states"])
        action_name = "base_actions" if "base_actions" in archive.files else "suggestions"
        actions = np.asarray(archive[action_name]) if action_name in archive.files else None
    state_count = min(21, len(states))
    action_count = min(20, len(actions)) if actions is not None else 0
    state_hash = sha256_array(states[:state_count])
    action_hash = sha256_array(actions[:action_count]) if actions is not None else None
    return state_count, state_hash, action_hash


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _fourfold(left: list[dict], right: list[dict], key: str) -> dict:
    left_map = {str(row["stat_group_id"]): bool(row[key]) for row in left}
    right_map = {str(row["stat_group_id"]): bool(row[key]) for row in right}
    if set(left_map) != set(right_map):
        raise ValueError("paired outcome table has mismatched roots")
    counts = Counter((left_map[root], right_map[root]) for root in left_map)
    return {
        "both_success": counts[(True, True)],
        "left_only": counts[(True, False)],
        "right_only": counts[(False, True)],
        "both_failure": counts[(False, False)],
    }


def evaluate(protocol_path: Path, episode_table: Path, output_dir: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    expected_roots = int(protocol["new_test_roots"])
    rows = read_table(episode_table)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("protocol_hash") != protocol_hash or not row.get("engineering_ok"):
            raise RuntimeError("evaluation input includes failed or wrong-protocol episodes")
        grouped[str(row["method_id"])].append(row)
    if set(grouped) != set(METHODS):
        raise RuntimeError(f"probe method coverage mismatch: {sorted(grouped)}")
    if any(len(grouped[method]) != expected_roots for method in METHODS):
        raise RuntimeError("every probe method must cover every root exactly once")

    baseline = {str(row["stat_group_id"]): row for row in grouped["NONE"]}
    if len(baseline) != expected_roots:
        raise RuntimeError("probe baseline root IDs are not unique")
    summaries = []
    for method in METHODS:
        values = grouped[method]
        summary = {
            "method_id": method,
            "planned_length": LENGTHS[method],
            "roots": len(values),
            "system_success_count": sum(bool(row["system_success"]) for row in values),
            "autonomous_completion_count": sum(bool(row["autonomous_completion"]) for row in values),
            "system_success_rate": _mean(values, "system_success"),
            "autonomous_completion_rate": _mean(values, "autonomous_completion"),
            "mean_utility": _mean(values, "utility"),
            "mean_helper_steps_actual": _mean(values, "helper_steps_actual"),
            "genuine_rescued_roots": sum(
                not bool(baseline[str(row["stat_group_id"])]["system_success"])
                and bool(row["genuine_handoff_success"])
                for row in values
            ),
            "interference_roots": sum(
                bool(baseline[str(row["stat_group_id"])]["system_success"])
                and not bool(row["system_success"])
                for row in values
            ),
            "mean_takeover_count": _mean(values, "takeover_count"),
        }
        summaries.append(summary)

    prefix_rows = []
    prefix_failures = []
    for root_id in sorted(baseline):
        keys = {method: _prefix_key(next(row for row in grouped[method] if str(row["stat_group_id"]) == root_id)) for method in METHODS}
        verified = len(set(keys.values())) == 1
        prefix_rows.append({"stat_group_id": root_id, "prefix_verified": verified, **{
            f"{method.lower()}_prefix_state_count": keys[method][0] for method in METHODS
        }})
        if not verified:
            prefix_failures.append({"stat_group_id": root_id, "prefix_keys": keys})

    comparisons = {}
    pair_rows = []
    for left, right in (("FIXED_L40", "FIXED_L80"), ("FIXED_L60", "FIXED_L80"), ("FIXED_L40", "FIXED_L60"), ("FIXED_L40", "NONE"), ("FIXED_L60", "NONE"), ("FIXED_L80", "NONE")):
        name = f"{left}_minus_{right}"
        comparisons[name] = {
            "left": left,
            "right": right,
            "utility": paired_bootstrap(
                {str(row["stat_group_id"]): float(row["utility"]) for row in grouped[left]},
                {str(row["stat_group_id"]): float(row["utility"]) for row in grouped[right]},
                int(protocol["statistics"]["bootstrap_repeats"]),
                int(protocol["statistics"]["seed"]),
            ),
            "system_success": _fourfold(grouped[left], grouped[right], "system_success"),
            "autonomous_completion": _fourfold(grouped[left], grouped[right], "autonomous_completion"),
        }
        for root_id in sorted(baseline):
            left_row = next(row for row in grouped[left] if str(row["stat_group_id"]) == root_id)
            right_row = next(row for row in grouped[right] if str(row["stat_group_id"]) == root_id)
            if right in SHORT or left in SHORT:
                pair_rows.append({
                    "stat_group_id": root_id,
                    "comparison": name,
                    "left_method": left,
                    "right_method": right,
                    "autonomous_case": _pair_case(bool(left_row["autonomous_completion"]), bool(right_row["autonomous_completion"])),
                    "system_case": _pair_case(bool(left_row["system_success"]), bool(right_row["system_success"])),
                    "utility_delta": float(left_row["utility"]) - float(right_row["utility"]),
                })

    branch_by_root = {
        str(row["stat_group_id"]): {
            method: next(value for value in grouped[method] if str(value["stat_group_id"]) == str(row["stat_group_id"]))
            for method in METHODS
        }
        for row in grouped["NONE"]
    }
    oracle_components = Counter()
    oracle_rows = []
    for root_id, branches in branch_by_root.items():
        oracle_name = min(
            (method for method in METHODS if method != "NONE"),
            key=lambda method: (-float(branches[method]["utility"]), LENGTHS[method]),
        )
        component = _oracle_component(branches[oracle_name], branches["FIXED_L80"])
        oracle_components[component] += 1
        oracle_rows.append({
            "stat_group_id": root_id,
            "oracle_method": oracle_name,
            "oracle_utility": branches[oracle_name]["utility"],
            "fixed_l80_utility": branches["FIXED_L80"]["utility"],
            "oracle_minus_fixed_l80_utility": float(branches[oracle_name]["utility"]) - float(branches["FIXED_L80"]["utility"]),
            "oracle_component": component,
        })

    short_success = {method: next(row for row in summaries if row["method_id"] == method)["autonomous_completion_count"] for method in SHORT}
    continue_cases = {
        method: sum(
            not bool(branches[method]["autonomous_completion"])
            and bool(branches["FIXED_L80"]["autonomous_completion"])
            for branches in branch_by_root.values()
        )
        for method in SHORT
    }
    opportunity_confirmed = not prefix_failures and max(short_success.values()) >= 5 and max(continue_cases.values()) >= 5
    decision = {
        "schema_version": "hb3p_exit_probe_decision_v1",
        "decision": "INTERMEDIATE_EXIT_OPPORTUNITY_CONFIRMED" if opportunity_confirmed else "NO_STOP_CONTINUE_OPPORTUNITY_CONFIRMED",
        "training_allowed": bool(opportunity_confirmed),
        "training_performed": False,
        "prefix_verified_roots": sum(bool(row["prefix_verified"]) for row in prefix_rows),
        "prefix_failures": prefix_failures,
        "short_exit_autonomous_success_counts": short_success,
        "continue_to_l80_autonomous_success_cases": continue_cases,
        "oracle_components": dict(sorted(oracle_components.items())),
        "oracle_mean_delta_vs_fixed_l80": float(np.mean([row["oracle_minus_fixed_l80_utility"] for row in oracle_rows])),
        "protocol_hash": protocol_hash,
        "new_rollout_count": expected_roots * (len(METHODS) - 1),
        "scope": "fixed_entry_t20_exit_lengths_40_60_80",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(prefix_rows, output_dir / "prefix_checks.csv")
    write_csv(pair_rows, output_dir / "pair_cases.csv")
    write_csv(oracle_rows, output_dir / "oracle_rows.csv")
    atomic_json_dump({"schema_version": "hb3p_exit_probe_summary_v1", "methods": summaries, "comparisons": comparisons}, output_dir / "summary.json")
    atomic_json_dump(decision, output_dir / "decision.json")
    return {"decision": decision, "summary": summaries, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--episode-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.protocol, args.episode_table, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
