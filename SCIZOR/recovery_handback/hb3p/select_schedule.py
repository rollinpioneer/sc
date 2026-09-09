"""Select the fixed validation schedule and pre-register HB3-P methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb2.metrics import choose_length
from recovery_handback.hb3p.controller import Rule, select_episode_from_cached_predictions
from recovery_handback.hb3p.io import mark, read_jsonl, write_csv, write_jsonl
from recovery_handback.hb3p.metrics import episode_value


def _baseline(root: dict) -> dict:
    return {
        "engineering_ok": bool(root["baseline_engineering_ok"]),
        "system_success": bool(root["baseline_system_success"]),
        "genuine_handoff_success": False,
        "helper_steps_actual": 0,
        "takeover_count": 0,
        "takeover_t": None,
        "selected_length": 0,
        "query_count": 0,
    }


def _branch_choice(root: dict, branches: dict, takeover_t: int | None, length: int, queries: int = 0) -> dict:
    if takeover_t is None or length == 0:
        result = _baseline(root)
        result["query_count"] = int(queries)
        return result
    branch = branches[str(takeover_t)][f"l{length}"]
    return {
        "engineering_ok": bool(branch["engineering_ok"]),
        "system_success": bool(branch["system_success"]),
        "genuine_handoff_success": bool(branch["genuine_handoff_success"]),
        "helper_steps_actual": int(branch["helper_steps_actual"]),
        "changed_action_steps": int(branch.get("changed_action_steps") or 0),
        "repair_calls_after_handoff": int(branch.get("repair_calls_after_handoff") or 0),
        "takeover_count": 1,
        "takeover_t": int(takeover_t),
        "selected_length": int(length),
        "query_count": int(queries),
        "source_trajectory_path": branch.get("trajectory_path"),
    }


def _summarize(method_id: str, rows: list[dict], penalty: float) -> dict:
    values = [episode_value(row, penalty) for row in rows]
    return {
        "method_id": method_id,
        "roots": len(rows),
        "root_mean_utility": sum(values) / len(values),
        "system_success_rate": sum(bool(row["system_success"]) for row in rows) / len(rows),
        "autonomous_completion_rate": sum(
            bool(row["genuine_handoff_success"]) if row["takeover_count"] else bool(row["system_success"])
            for row in rows
        ) / len(rows),
        "mean_actual_helper_steps": sum(int(row["helper_steps_actual"]) for row in rows) / len(rows),
        "takeover_rate": sum(int(row["takeover_count"]) for row in rows) / len(rows),
        "mean_query_count": sum(int(row.get("query_count", 0)) for row in rows) / len(rows),
    }


def select_schedule(config_path: Path, development_dir: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hb3p = config["hb3p"]
    roots = read_jsonl(development_dir / "development_roots.jsonl")
    branch_index = json.loads((development_dir / "branch_index.json").read_text())
    predictions = json.loads((development_dir / "cached_predictions.json").read_text())
    penalty = float(hb3p["primary_lambda"])
    candidates = [("NONE", None, 0)] + [
        (f"S_{t}_{length}", t, length)
        for t in hb3p["candidate_times"]
        for length in hb3p["helper_lengths"]
    ]
    candidate_rows = []
    all_choices = []
    candidate_outputs = {}
    for candidate_id, scheduled_t, length in candidates:
        rows = []
        for root in roots:
            legal = set(int(value) for value in root["legal_anchor_times"])
            takeover_t = scheduled_t if scheduled_t in legal else None
            row = {
                "root_id": root["root_id"],
                "stat_group_id": root["stat_group_id"],
                "method_id": candidate_id,
                **_branch_choice(root, branch_index.get(root["root_id"], {}), takeover_t, length),
            }
            rows.append(row)
        summary = _summarize(candidate_id, rows, penalty)
        summary.update({"scheduled_t": scheduled_t, "planned_length": length})
        candidate_rows.append(summary)
        candidate_outputs[candidate_id] = rows
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            -row["root_mean_utility"],
            row["mean_actual_helper_steps"],
            row["planned_length"],
            -1 if row["scheduled_t"] is None else row["scheduled_t"],
        ),
    )
    selected = ranked[0]
    s_star = selected["method_id"]

    m0_schedule = {}
    for t in hb3p["candidate_times"]:
        options = {
            choose_length(
                values,
                penalty,
                denominator=float(hb3p["cost_denominator"]),
                tolerance=float(hb3p["tie_tolerance"]),
            )
            for key, values in predictions["M0_time"].items()
            if key.endswith(f":{t}")
        }
        if len(options) != 1:
            raise ValueError(f"M0 is not a time-only rule at t={t}: {sorted(options)}")
        m0_schedule[str(t)] = options.pop()
    atomic_json_dump({
        "schema_version": "hb3p_m0_compiled_schedule_v1",
        "temperature": json.loads(Path(config["hb3p"]["hb2_frozen_protocol"]).read_text())["models"]["M0_time"]["seed0"]["temperature"],
        "schedule": m0_schedule,
    }, development_dir / "m0_calibrated_schedule.json")

    rules = {
        "NONE": Rule("none"),
        "S_STAR": Rule("none") if s_star == "NONE" else Rule(
            "scheduled", scheduled_t=int(selected["scheduled_t"]), scheduled_length=int(selected["planned_length"])
        ),
        "M0_GRID": Rule("model", model="M0_time", penalty=penalty),
        "RISK_GRID": Rule("risk", model="M0_time", risk_threshold=0.0, risk_length=80),
        "M1_GRID": Rule("model", model="M1_proprio", penalty=penalty),
        "M4_GRID": Rule("model", model="M4_paired", penalty=penalty),
    }
    mechanism_rows = {}
    for method_id, rule in rules.items():
        rows = []
        for root in roots:
            if method_id == "NONE":
                selection = {"takeover_t": None, "length": 0, "queries_before_takeover": 0}
            elif method_id == "S_STAR":
                selection = select_episode_from_cached_predictions(
                    rule, {}, root["first_raw_success_state"], root["episode_end_state_index"]
                )
            else:
                model = rule.model
                root_predictions = {
                    int(key.rsplit(":", 1)[1]): value
                    for key, value in predictions[model].items()
                    if key.startswith(root["root_id"] + ":")
                }
                selection = select_episode_from_cached_predictions(
                    rule, root_predictions, root["first_raw_success_state"], root["episode_end_state_index"]
                )
            row = {
                "root_id": root["root_id"],
                "stat_group_id": root["stat_group_id"],
                "method_id": method_id,
                **_branch_choice(
                    root,
                    branch_index.get(root["root_id"], {}),
                    selection["takeover_t"],
                    selection["length"],
                    selection["queries_before_takeover"],
                ),
            }
            rows.append(row)
            all_choices.append(row)
        mechanism_rows[method_id] = rows

    aliases = []
    signatures = {
        method: [(row["root_id"], row["takeover_t"], row["selected_length"]) for row in rows]
        for method, rows in mechanism_rows.items()
    }
    canonical = {}
    for method in hb3p["logical_methods"]:
        match = next((other for other in canonical if signatures[method] == signatures[other]), None)
        if match is None:
            canonical[method] = method
        else:
            canonical[method] = canonical[match]
            aliases.append({
                "alias": method,
                "canonical_execution": canonical[match],
                "reason": "same_complete_validation_execution_under_frozen_rules",
            })
    atomic_json_dump({"schema_version": "hb3p_method_aliases_v1", "aliases": aliases}, development_dir / "method_aliases.json")
    write_csv(candidate_rows, development_dir / "fixed_schedule_candidates.csv")
    write_jsonl(all_choices, development_dir / "root_episode_choices.jsonl")
    method_metrics = [_summarize(method, rows, penalty) for method, rows in mechanism_rows.items()]
    write_csv(method_metrics, development_dir / "root_episode_metrics.csv")

    f_protocol_path = Path(hb3p["hb2_frozen_protocol"])
    f_protocol = json.loads(f_protocol_path.read_text(encoding="utf-8"))
    protocol = {
        "schema_version": "hb3p_protocol_draft_v1",
        "frozen": False,
        "source_ref": config["source_ref"],
        "task": "square",
        "semantic_pair_id": f_protocol["semantic_pair_id"],
        "scope": hb3p["scope"],
        "hb2_status_preserved": hb3p["hb2_status_preserved"],
        "hb2_f_protocol": str(f_protocol_path.resolve()),
        "hb2_f_protocol_sha256": sha256_file(f_protocol_path),
        "models": {
            model: f_protocol["models"][model]["seed0"]
            for model in ("M0_time", "M1_proprio", "M4_paired")
        },
        "normalizer": f_protocol["normalizer"],
        "input_schema": f_protocol["input_schema"],
        "selected_fixed_schedule": {
            "candidate_id": s_star,
            "scheduled_t": selected["scheduled_t"],
            "length": selected["planned_length"],
            "validation_root_mean_utility": selected["root_mean_utility"],
            "selection_split": "hb2_validation_only",
        },
        "m0_compiled_schedule": m0_schedule,
        "risk_rule": {"p0_model": "M0_time", "threshold": 0.0, "length": 80},
        "methods": {
            "NONE": {"kind": "none"},
            "S_STAR": {"kind": rules["S_STAR"].kind, "scheduled_t": rules["S_STAR"].scheduled_t, "length": rules["S_STAR"].scheduled_length},
            "M0_GRID": {"kind": "model", "model": "M0_time"},
            "RISK_GRID": {"kind": "risk", "model": "M0_time", "threshold": 0.0, "length": 80},
            "M1_GRID": {"kind": "model", "model": "M1_proprio"},
            "M4_GRID": {"kind": "model", "model": "M4_paired"},
        },
        "method_aliases": aliases,
        "primary_method": "M1_GRID",
        "primary_comparator": "S_STAR",
        "candidate_times": hb3p["candidate_times"],
        "helper_lengths": hb3p["helper_lengths"],
        "decision": {
            "primary_lambda": penalty,
            "cost_denominator": hb3p["cost_denominator"],
            "tie_tolerance": hb3p["tie_tolerance"],
            "rule": hb3p["entry_rule"],
        },
        "horizon_steps": int(config["horizon_steps"]),
        "minimum_autonomous_steps": int(config["minimum_autonomous_steps"]),
        "success_consecutive_steps": int(config["success_consecutive_steps"]),
        "learned_exit_enabled": False,
        "allow_retakeover": False,
        "max_takeovers": 1,
        "statistics": hb3p["statistics"],
        "new_test_roots": hb3p["new_test_roots"],
        "test_seed_start": int(config["roles"]["hb3p_test"]["seed_start"]) + int(config["square_seed_offset"]),
    }
    atomic_json_dump(protocol, output)
    mark(output.parents[1], "hb3p-B.done", f"selected {s_star} from HB2 validation")
    return {"selected_schedule": s_star, "aliases": aliases, "methods": method_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select_schedule(args.config, args.development_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
