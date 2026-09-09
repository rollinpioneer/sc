"""Read-only HB3-P per-root and fixed-entry exit opportunity diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, sha256_array, sha256_file
from recovery_handback.hb3p.io import write_csv


SHORT_BRANCHES = ("l5", "l20")
EXIT_BRANCHES = ("l5", "l20", "l80")
ALL_BRANCHES = ("none", "l5", "l20", "l80", "full")
PLANNED_LENGTH = {"none": 0, "l5": 5, "l20": 20, "l80": 80}
LAMBDA = 0.25
DENOMINATOR = 400.0


def _bool(value: Any) -> bool:
    return bool(value)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _present_number(value: Any, default: float) -> float:
    number = _number(value)
    return default if number is None else number


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_unique_branch_rows(hb2_root: Path, role: str) -> list[dict]:
    role_root = hb2_root / "branches" / role
    if not role_root.is_dir():
        raise FileNotFoundError(role_root)
    rows: list[dict] = []
    for shard in sorted(role_root.glob("shard_*")):
        preferred = shard / "branch_results.parquet"
        paths = [preferred] if preferred.is_file() else sorted(shard.glob("branch_results*.parquet"))[:1]
        for path in paths:
            rows.extend(read_table(path))
    deduped: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        key = (str(row["root_id"]), int(row["anchor_t"]), str(row["branch_name"]))
        previous = deduped.get(key)
        if previous is not None:
            comparable = (previous.get("trajectory_path"), row.get("trajectory_path"), previous.get("system_success"), row.get("system_success"))
            if comparable[0] != comparable[1] or comparable[2] != comparable[3]:
                raise RuntimeError(f"conflicting duplicate HB2 branch row: {key}")
            continue
        deduped[key] = row
    return list(deduped.values())


def _branch_utility(row: dict, branch: str) -> tuple[bool, float]:
    if branch == "none":
        autonomous = _bool(row.get("system_success"))
        helper_steps = 0.0
    else:
        autonomous = _bool(row.get("genuine_handoff_success"))
        helper_steps = _number(row.get("helper_steps_actual")) or 0.0
    return autonomous, autonomous - LAMBDA * helper_steps / DENOMINATOR


def _load_trajectory_prefix(row: dict) -> tuple[str | None, str | None]:
    path = Path(str(row.get("trajectory_path", "")))
    if not path.is_file():
        return None, None
    try:
        with np.load(path, allow_pickle=False) as archive:
            states = np.asarray(archive["states"])
    except (OSError, KeyError, ValueError):
        return None, None
    if states.ndim != 2 or len(states) == 0:
        return None, None
    return sha256_array(states[0]), sha256_file(path)


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


def _trace_for(row: dict) -> list[dict]:
    path = Path(str(row.get("decision_trace_path", "")))
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"invalid decision trace records: {path}")
    return records


def _diagnose_hb3p(root: Path) -> tuple[list[dict], dict]:
    rows = read_table(root / "metrics/test/episodes.parquet")
    if len(rows) != 480:
        raise RuntimeError(f"expected 480 HB3-P logical rows, got {len(rows)}")
    by_key = {(str(row["root_id"]), str(row["method_id"])): row for row in rows}
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate HB3-P root-method records")
    methods = {str(row["method_id"]) for row in rows}
    expected = {"NONE", "S_STAR", "M0_GRID", "RISK_GRID", "M1_GRID", "M4_GRID"}
    if methods != expected:
        raise RuntimeError(f"unexpected HB3-P methods: {sorted(methods)}")
    roots = sorted({str(row["root_id"]) for row in rows})
    output: list[dict] = []
    for root_id in roots:
        baseline = by_key[(root_id, "NONE")]
        scheduled = by_key[(root_id, "S_STAR")]
        for method in ("NONE", "S_STAR", "M0_GRID", "RISK_GRID", "M1_GRID", "M4_GRID"):
            row = by_key[(root_id, method)]
            trace = _trace_for(row)
            decision_summary = ";".join(
                f"t{int(item['absolute_t'])}:L{int(item.get('reported_selected_length') or 0)}"
                for item in trace
            )
            baseline_success = _bool(baseline.get("system_success"))
            system_success = _bool(row.get("system_success"))
            autonomous = _bool(row.get("autonomous_completion"))
            if method == "NONE":
                relation = "baseline"
            elif baseline_success and not system_success:
                relation = "baseline_interference"
            elif not baseline_success and autonomous:
                relation = "genuine_rescue"
            elif not baseline_success and system_success and not autonomous:
                relation = "helper_only_success"
            elif baseline_success and system_success and int(row.get("takeover_count") or 0) == 0:
                relation = "baseline_preserved_without_takeover"
            else:
                relation = "no_rescue_or_change"
            method_outcome_diff = (
                system_success != _bool(scheduled.get("system_success"))
                or autonomous != _bool(scheduled.get("autonomous_completion"))
            )
            focus = []
            if method == "M1_GRID" and int(row.get("takeover_count") or 0) == 0:
                focus.append("m1_no_takeover")
            if method == "M1_GRID" and method_outcome_diff:
                focus.append("m1_vs_s_star_outcome_difference")
            if method == "M1_GRID" and relation == "genuine_rescue":
                focus.append("m1_genuine_rescue")
            if method == "M1_GRID" and relation == "baseline_interference":
                focus.append("m1_interference")
            output.append({
                "root_id": root_id,
                "stat_group_id": row.get("stat_group_id"),
                "root_seed": row.get("root_seed"),
                "method_id": method,
                "canonical_execution": row.get("canonical_execution"),
                "baseline_system_success": baseline_success,
                "relation_to_none": relation,
                "takeover_t": row.get("takeover_t"),
                "selected_length": row.get("selected_length"),
                "helper_steps_actual": row.get("helper_steps_actual"),
                "changed_action_steps": row.get("changed_action_steps"),
                "utility": row.get("utility"),
                "system_success": system_success,
                "autonomous_completion": autonomous,
                "genuine_handoff_success": _bool(row.get("genuine_handoff_success")),
                "helper_completed_task": _bool(row.get("helper_completed_task")),
                "success_seen_under_helper": _bool(row.get("success_seen_under_helper")),
                "handoff_executed": _bool(row.get("handoff_executed")),
                "first_raw_success_state": row.get("first_raw_success_state"),
                "stable_success_state": row.get("stable_success_state"),
                "handoff_t": row.get("handoff_t"),
                "first_success_wait_after_handoff": row.get("first_success_wait_after_handoff"),
                "query_count": row.get("query_count"),
                "queried_times": row.get("queried_times"),
                "decision_summary": decision_summary,
                "decision_trace_available": bool(trace),
                "query_trace_json": _json_text(trace),
                "method_vs_s_star_outcome_difference": method_outcome_diff,
                "method_vs_s_star_utility_delta": (
                    float(row.get("utility") or 0.0) - float(scheduled.get("utility") or 0.0)
                ),
                "diagnostic_focus": ";".join(focus),
            })
    m1_rows = [row for row in output if row["method_id"] == "M1_GRID"]
    summary = {
        "schema_version": "hb3p_per_root_diagnosis_v1",
        "roots": len(roots),
        "logical_records": len(output),
        "m1_no_takeover_roots": sum(_number(row["takeover_t"]) is None for row in m1_rows),
        "m1_vs_s_star_outcome_difference_roots": sum(bool(row["method_vs_s_star_outcome_difference"]) for row in m1_rows),
        "m1_genuine_rescue_roots": sum(row["relation_to_none"] == "genuine_rescue" for row in m1_rows),
        "m1_interference_roots": sum(row["relation_to_none"] == "baseline_interference" for row in m1_rows),
        "query_trace_rows": sum(bool(row["decision_trace_available"]) for row in output),
    }
    return output, summary


def _diagnose_fixed_entry(hb2_root: Path, role: str, anchor_t: int) -> tuple[list[dict], dict]:
    rows = [row for row in _read_unique_branch_rows(hb2_root, role) if int(row["anchor_t"]) == anchor_t]
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["root_id"])][str(row["branch_name"])] = row
    if len(grouped) != 40:
        raise RuntimeError(f"expected 40 HB2 validation roots at t={anchor_t}, got {len(grouped)}")
    output: list[dict] = []
    prefix_failures: list[dict] = []
    pair_counts: Counter[str] = Counter()
    oracle_components: Counter[str] = Counter()
    branch_summaries: dict[str, dict[str, float]] = {}
    for root_id in sorted(grouped):
        branches = grouped[root_id]
        missing = sorted(set(ALL_BRANCHES) - set(branches))
        if missing:
            raise RuntimeError(f"missing HB2 fixed-entry branches for {root_id}: {missing}")
        prefix_rows = [branches[name] for name in ALL_BRANCHES]
        anchor_hashes = []
        state_hashes = []
        trajectory_hashes = []
        for row in prefix_rows:
            anchor_path = Path(str(row.get("anchor_history_path", "")))
            anchor_hashes.append(sha256_file(anchor_path) if anchor_path.is_file() else None)
            state_hash, trajectory_hash = _load_trajectory_prefix(row)
            state_hashes.append(state_hash)
            trajectory_hashes.append(trajectory_hash)
        metadata_ok = (
            len({row.get("config_hash") for row in prefix_rows}) == 1
            and len({row.get("policy_pair_hash") for row in prefix_rows}) == 1
            and len({row.get("memory_capture_phase") for row in prefix_rows}) == 1
            and {int(_present_number(row.get("prefix_env_steps"), -1)) for row in prefix_rows} == {anchor_t}
            and all(abs(_present_number(row.get("prefix_action_max_abs"), 1.0)) <= 1e-12 for row in prefix_rows)
            and all(abs(_present_number(row.get("prefix_state_max_abs"), 1.0)) <= 1e-12 for row in prefix_rows)
        )
        prefix_verified = metadata_ok and len(set(anchor_hashes)) == 1 and None not in anchor_hashes and len(set(state_hashes)) == 1 and None not in state_hashes
        if not prefix_verified:
            prefix_failures.append({"root_id": root_id, "anchor_hashes": anchor_hashes, "state_hashes": state_hashes})
        branch_values: dict[str, dict] = {}
        for name in ("none",) + EXIT_BRANCHES:
            row = branches[name]
            autonomous, utility = _branch_utility(row, name)
            branch_values[name] = {
                "system_success": _bool(row.get("system_success")),
                "autonomous_completion": autonomous,
                "helper_completed_task": _bool(row.get("helper_completed_task")),
                "success_seen_under_helper": _bool(row.get("success_seen_under_helper")),
                "handoff_executed": _bool(row.get("handoff_executed")),
                "genuine_handoff_success": _bool(row.get("genuine_handoff_success")),
                "helper_steps_actual": _number(row.get("helper_steps_actual")) or 0.0,
                "changed_action_steps": _number(row.get("changed_action_steps")) or 0.0,
                "utility": utility,
                "first_raw_success_state": row.get("first_raw_success_state"),
                "stable_success_state": row.get("stable_success_state"),
                "handoff_state_index": row.get("handoff_state_index"),
                "raw_return": row.get("raw_return"),
                "trajectory_path": row.get("trajectory_path"),
            }
        oracle_name = min(
            EXIT_BRANCHES,
            key=lambda name: (-branch_values[name]["utility"], branch_values[name]["helper_steps_actual"], PLANNED_LENGTH[name]),
        )
        oracle = dict(branch_values[oracle_name])
        oracle["branch"] = oracle_name
        component = _oracle_component(oracle, branch_values["l80"])
        oracle_components[component] += 1
        for short in SHORT_BRANCHES:
            pair_counts[f"{short}_vs_l80_auto::{_pair_case(branch_values[short]['autonomous_completion'], branch_values['l80']['autonomous_completion'])}"] += 1
            pair_counts[f"{short}_vs_l80_system::{_pair_case(branch_values[short]['system_success'], branch_values['l80']['system_success'])}"] += 1
        record = {
            "root_id": root_id,
            "stat_group_id": branches["none"].get("stat_group_id"),
            "role": role,
            "anchor_t": anchor_t,
            "prefix_verified": prefix_verified,
            "prefix_config_hash": branches["none"].get("config_hash"),
            "prefix_policy_pair_hash": branches["none"].get("policy_pair_hash"),
            "prefix_memory_capture_phase": branches["none"].get("memory_capture_phase"),
            "prefix_env_steps": branches["none"].get("prefix_env_steps"),
            "prefix_action_max_abs": max(_present_number(row.get("prefix_action_max_abs"), 0.0) for row in prefix_rows),
            "prefix_state_max_abs": max(_present_number(row.get("prefix_state_max_abs"), 0.0) for row in prefix_rows),
            "prefix_anchor_history_sha256": _json_text(anchor_hashes),
            "prefix_initial_state_sha256": _json_text(state_hashes),
            "branch_trajectory_sha256": _json_text(trajectory_hashes),
            "oracle_branch": oracle_name,
            "oracle_planned_length": PLANNED_LENGTH[oracle_name],
            "oracle_utility": oracle["utility"],
            "l80_utility": branch_values["l80"]["utility"],
            "oracle_minus_l80_utility": oracle["utility"] - branch_values["l80"]["utility"],
            "oracle_autonomous_completion": oracle["autonomous_completion"],
            "oracle_component": component,
        }
        for name in ("none",) + EXIT_BRANCHES:
            value = branch_values[name]
            prefix = f"{name}_"
            for key in (
                "system_success", "autonomous_completion", "helper_completed_task", "success_seen_under_helper",
                "handoff_executed", "genuine_handoff_success", "helper_steps_actual", "changed_action_steps",
                "utility", "first_raw_success_state", "stable_success_state", "handoff_state_index", "raw_return",
            ):
                record[prefix + key] = value[key]
        for short in SHORT_BRANCHES:
            record[f"{short}_vs_l80_auto_case"] = _pair_case(
                branch_values[short]["autonomous_completion"], branch_values["l80"]["autonomous_completion"]
            )
            record[f"{short}_vs_l80_system_case"] = _pair_case(
                branch_values[short]["system_success"], branch_values["l80"]["system_success"]
            )
            record[f"{short}_vs_l80_utility_delta"] = branch_values[short]["utility"] - branch_values["l80"]["utility"]
        output.append(record)
    for name in ("none",) + EXIT_BRANCHES:
        values = [float(row[f"{name}_utility"]) for row in output]
        branch_summaries[name] = {
            "roots": float(len(values)),
            "system_success_count": float(sum(bool(row[f"{name}_system_success"]) for row in output)),
            "autonomous_completion_count": float(sum(bool(row[f"{name}_autonomous_completion"]) for row in output)),
            "mean_utility": float(np.mean(values)),
            "mean_helper_steps_actual": float(np.mean([float(row[f"{name}_helper_steps_actual"]) for row in output])),
        }
    summary = {
        "schema_version": "hb3p_fixed_entry_exit_diagnosis_v1",
        "role": role,
        "anchor_t": anchor_t,
        "roots": len(output),
        "prefix_verified_roots": sum(bool(row["prefix_verified"]) for row in output),
        "prefix_failures": prefix_failures,
        "branches": branch_summaries,
        "pair_counts": dict(sorted(pair_counts.items())),
        "oracle_components": dict(sorted(oracle_components.items())),
        "oracle_mean_utility": float(np.mean([row["oracle_utility"] for row in output])),
        "l80_mean_utility": float(np.mean([row["l80_utility"] for row in output])),
        "oracle_mean_delta_vs_l80": float(np.mean([row["oracle_minus_l80_utility"] for row in output])),
        "oracle_success_with_lower_cost_roots": oracle_components["success_with_lower_cost"],
        "oracle_both_failed_cost_only_roots": oracle_components["both_failed_cost_only"],
    }
    return output, summary


def _write_markdown(output: Path, hb3p_summary: dict, fixed_summary: dict, decision: dict) -> None:
    branches = fixed_summary["branches"]
    lines = [
        "# HB3-P Exit Diagnosis",
        "",
        "This is a read-only diagnosis. It adds no training and no rollout.",
        "",
        "## HB3-P Per-Root Attribution",
        "",
        f"- HB3-P logical records: {hb3p_summary['logical_records']} across {hb3p_summary['roots']} roots.",
        f"- M1 no-takeover roots: {hb3p_summary['m1_no_takeover_roots']}.",
        f"- M1 versus S_STAR outcome-difference roots: {hb3p_summary['m1_vs_s_star_outcome_difference_roots']}.",
        f"- M1 genuine rescue roots: {hb3p_summary['m1_genuine_rescue_roots']}.",
        f"- M1 interference roots: {hb3p_summary['m1_interference_roots']}.",
        "- `hb3p_per_root_diagnosis.csv` preserves actual query predictions and selected decisions as compact JSON.",
        "",
        "## Fixed Entry: t=20",
        "",
        f"- Source: `{fixed_summary['role']}`; roots: {fixed_summary['roots']}.",
        f"- Shared-prefix verification: {fixed_summary['prefix_verified_roots']}/{fixed_summary['roots']} roots passed.",
        "",
        "| Branch | System success | Autonomous completion | Mean utility | Mean helper steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("none", "l5", "l20", "l80"):
        row = branches[name]
        lines.append(
            f"| {name} | {int(row['system_success_count'])}/{int(row['roots'])} | "
            f"{int(row['autonomous_completion_count'])}/{int(row['roots'])} | "
            f"{row['mean_utility']:.6f} | {row['mean_helper_steps_actual']:.3f} |"
        )
    lines.extend([
        "",
        "### Stop-versus-continue cases",
        "",
        f"- `l5` stop versus `l80` continue, autonomous: `{fixed_summary['pair_counts'].get('l5_vs_l80_auto::stop_failure_continue_success', 0)}` short-failure/continue-success roots; `{fixed_summary['pair_counts'].get('l5_vs_l80_auto::stop_success_continue_success', 0)}` both-success roots.",
        f"- `l20` stop versus `l80` continue, autonomous: `{fixed_summary['pair_counts'].get('l20_vs_l80_auto::stop_failure_continue_success', 0)}` short-failure/continue-success roots; `{fixed_summary['pair_counts'].get('l20_vs_l80_auto::stop_success_continue_success', 0)}` both-success roots.",
        f"- Observed exit-grid oracle minus fixed `l80` mean utility: `{fixed_summary['oracle_mean_delta_vs_l80']:.6f}`.",
        f"- Oracle success-with-lower-cost roots: `{fixed_summary['oracle_success_with_lower_cost_roots']}`.",
        f"- Oracle gains from both-failed cost-only cases: `{fixed_summary['oracle_both_failed_cost_only_roots']}`.",
        "",
        "## Route Decision",
        "",
        f"- Decision: **{decision['decision']}**.",
        f"- Training allowed: `{decision['training_allowed']}`.",
        f"- HB3-P status preserved: `{decision['hb3p_status_preserved']}`.",
        "- The oracle is retrospective and is not a deployable selector or a formal success-rate upper bound.",
        "- The next bounded probe, if executed, must use a new frozen protocol with fixed t=20 and 40/60/80-step exits plus NONE.",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hb3p-root", type=Path, required=True)
    parser.add_argument("--hb2-root", type=Path, required=True)
    parser.add_argument("--hb2-role", default="hb2_val")
    parser.add_argument("--anchor-t", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.anchor_t != 20:
        raise ValueError("the bounded exit diagnosis is frozen at t=20")
    per_root, hb3p_summary = _diagnose_hb3p(args.hb3p_root)
    fixed, fixed_summary = _diagnose_fixed_entry(args.hb2_root, args.hb2_role, args.anchor_t)
    if fixed_summary["prefix_failures"]:
        decision_name = "HOLD_PREFIX_INCOMPATIBLE"
        training_allowed = False
    else:
        short_success = max(
            fixed_summary["branches"][name]["autonomous_completion_count"] for name in SHORT_BRANCHES
        )
        continue_cases = max(
            fixed_summary["pair_counts"].get(f"{name}_vs_l80_auto::stop_failure_continue_success", 0)
            for name in SHORT_BRANCHES
        )
        decision_name = (
            "RUN_BOUNDED_INTERMEDIATE_EXIT_PROBE"
            if short_success < 5 and continue_cases >= 5
            else "NO_TRAINING_OPPORTUNITY_ESTABLISHED"
        )
        training_allowed = False
    decision = {
        "schema_version": "hb3p_exit_route_decision_v1",
        "decision": decision_name,
        "training_allowed": training_allowed,
        "new_rollout_performed": False,
        "hb3p_status_preserved": "STATE_ENTRY_SIGNAL_ONLY",
        "hb3p_summary": hb3p_summary,
        "fixed_entry_summary": fixed_summary,
        "next_probe": {
            "roots": 40,
            "entry_t": 20,
            "methods": ["NONE", "FIXED_L40", "FIXED_L60", "FIXED_L80"],
            "planned_exit_lengths": [40, 60, 80],
            "maximum_complete_method_records": 160,
            "maximum_environment_steps": 64000,
            "separate_interface_pilot": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_root, args.output_dir / "hb3p_per_root_diagnosis.csv")
    write_csv(fixed, args.output_dir / "fixed_entry_exit_opportunities.csv")
    atomic_json_dump(hb3p_summary, args.output_dir / "hb3p_per_root_summary.json")
    atomic_json_dump(fixed_summary, args.output_dir / "fixed_entry_exit_summary.json")
    atomic_json_dump(decision, args.output_dir / "exit_diagnosis_decision.json")
    _write_markdown(args.output_dir / "HB3P_EXIT_DIAGNOSIS.md", hb3p_summary, fixed_summary, decision)
    print(json.dumps({"decision": decision_name, "hb3p": hb3p_summary, "fixed_entry": fixed_summary}, indent=2))


if __name__ == "__main__":
    main()
