"""Aggregate complete baseline and online records into one root-method table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, write_table
from recovery_handback.hb3p.metrics import episode_value


def _json_files(root: Path, method: str) -> list[Path]:
    return sorted(
        path for path in root.glob(f"**/{method}/*.json")
        if not path.name.endswith("_decision_trace.json") and not path.name.endswith("_handoff.json")
    )


def _trace_costs(row: dict) -> dict:
    path = Path(row["decision_trace_path"])
    records = json.loads(path.read_text(encoding="utf-8"))["records"] if path.is_file() else []
    return {
        "backbone_wall_seconds": sum(float(item.get("backbone_seconds", 0.0)) for item in records),
        "prediction_head_wall_seconds": sum(float(item.get("prediction_head_seconds", 0.0)) for item in records),
        "ipc_roundtrip_wall_seconds": sum(float(item.get("ipc_roundtrip_seconds", 0.0)) for item in records),
    }


def _baseline(root: dict, protocol_hash: str, semantic_pair_id: str) -> dict:
    return {
        "schema_version": "hb3p_aggregate_episode_v1",
        "task": "square", "role": str(root.get("role", root.get("exposure_role", "hb3p_test"))), "root_id": root["root_id"],
        "stat_group_id": root["stat_group_id"], "method_id": "NONE",
        "canonical_execution": "NONE", "protocol_hash": protocol_hash,
        "semantic_pair_id": semantic_pair_id, "engineering_ok": root.get("exception_reason") is None,
        "exception_reason": root.get("exception_reason"), "system_success": bool(root["baseline_success"]),
        "first_raw_success_state": root.get("first_raw_success_state"),
        "stable_success_state": root.get("stable_success_state"),
        "takeover_count": 0, "takeover_t": None, "selected_length": 0,
        "handoff_t": None, "handoff_executed": False, "genuine_handoff_success": False,
        "handoff_success_raw": False, "success_seen_under_helper": False,
        "helper_completed_task": False, "helper_steps_actual": 0,
        "changed_action_steps": 0, "repair_calls_after_handoff": 0,
        "base_policy_calls": int(root["actual_steps"]), "repair_policy_calls": 0,
        "episode_end_state_index": int(root["actual_steps"]),
        "first_success_wait_after_handoff": None,
        "autonomous_execution_steps_after_handoff": None,
        "query_count": 0, "queried_times": [], "inference_wall_seconds": 0.0,
        "backbone_wall_seconds": 0.0, "prediction_head_wall_seconds": 0.0,
        "ipc_roundtrip_wall_seconds": 0.0, "simulation_wall_seconds": None,
        "wall_seconds": None, "root_seed": int(root["seed"]),
        "initial_state_hash": root["initial_state_hash"],
        "trajectory_path": root["rollout_path"], "decision_trace_path": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(args.protocol)
    roots = read_table(args.roots)
    expected_roots = int(protocol["new_test_roots"])
    if len(roots) != expected_roots:
        raise RuntimeError(f"expected {expected_roots} test roots, got {len(roots)}")
    roots_by_id = {str(row["root_id"]): row for row in roots}
    if len(roots_by_id) != expected_roots:
        raise RuntimeError("test root IDs are not unique")
    aliases = {row["alias"]: row["canonical_execution"] for row in protocol.get("method_aliases", [])}
    canonical_methods = sorted({
        aliases.get(method, method) for method in protocol["methods"] if method != "NONE"
    })
    canonical = {}
    failures = []
    for method in canonical_methods:
        for path in _json_files(args.episodes_root, method):
            row = json.loads(path.read_text(encoding="utf-8"))
            root_id = str(row.get("root_id"))
            key = (root_id, method)
            if key in canonical:
                raise RuntimeError(f"duplicate online record: {key}")
            root = roots_by_id.get(root_id)
            if root is None:
                failures.append({"root_id": root_id, "method": method, "reason": "unknown_root"})
            elif row.get("method_id") != method:
                failures.append({"root_id": root_id, "method": method, "reason": "method_id_mismatch"})
            elif row.get("protocol_hash") != protocol_hash:
                failures.append({"root_id": row.get("root_id"), "method": method, "reason": "protocol_hash_mismatch"})
            elif row.get("semantic_pair_id") != protocol["semantic_pair_id"]:
                failures.append({"root_id": root_id, "method": method, "reason": "semantic_pair_id_mismatch"})
            elif int(row.get("root_seed", -1)) != int(root["seed"]):
                failures.append({"root_id": root_id, "method": method, "reason": "root_seed_mismatch"})
            elif row.get("initial_state_hash") != root.get("initial_state_hash"):
                failures.append({"root_id": root_id, "method": method, "reason": "initial_state_hash_mismatch"})
            elif not row.get("engineering_ok"):
                failures.append({"root_id": row.get("root_id"), "method": method, "reason": row.get("exception_reason")})
            elif not Path(row.get("trajectory_path", "")).is_file():
                failures.append({"root_id": row.get("root_id"), "method": method, "reason": "trajectory_missing"})
            canonical[key] = row
    missing = [
        {"root_id": root_id, "method": method}
        for root_id in roots_by_id for method in canonical_methods
        if (root_id, method) not in canonical
    ]
    rows = []
    for root_id, root in roots_by_id.items():
        rows.append(_baseline(root, protocol_hash, protocol["semantic_pair_id"]))
        for method_id in protocol["methods"]:
            if method_id == "NONE":
                continue
            execution = aliases.get(method_id, method_id)
            source = canonical.get((root_id, execution))
            if source is None:
                continue
            row = dict(source)
            row.update(_trace_costs(row))
            row["canonical_execution"] = execution
            row["method_id"] = method_id
            rows.append(row)
    for row in rows:
        if row["engineering_ok"]:
            row["autonomous_completion"] = (
                bool(row["genuine_handoff_success"])
                if int(row["takeover_count"]) > 0 else bool(row["system_success"])
            )
            row["utility"] = episode_value(
                row, float(protocol["decision"]["primary_lambda"]),
                float(protocol["decision"]["cost_denominator"]),
            )
    logical_methods = list(protocol["methods"])
    coverage = {
        "schema_version": "hb3p_test_coverage_v1",
        "protocol_hash": protocol_hash,
        "preregistered_roots": expected_roots,
        "logical_methods": logical_methods,
        "canonical_online_methods": canonical_methods,
        "expected_logical_records": expected_roots * len(logical_methods),
        "complete_logical_records": len(rows),
        "actual_unique_rollouts": expected_roots + len(canonical),
        "total_env_steps_actual": sum(
            int(row["actual_steps"]) for row in roots
        ) + sum(
            int(row.get("episode_end_state_index", 0)) for row in canonical.values()
        ),
        "missing_records": missing,
        "engineering_failures": failures,
        "complete": not missing and not failures and len(rows) == expected_roots * len(logical_methods),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(coverage, args.output_dir / "coverage.json")
    atomic_json_dump({"aliases": protocol.get("method_aliases", [])}, args.output_dir / "equivalent_methods.json")
    write_table(rows, args.output_dir / "episodes.parquet")
    if not coverage["complete"]:
        raise RuntimeError("HB3-P online coverage is incomplete; see coverage.json")
    print(json.dumps({"records": len(rows), "unique_rollouts": coverage["actual_unique_rollouts"], "complete": True}, indent=2))


if __name__ == "__main__":
    main()
