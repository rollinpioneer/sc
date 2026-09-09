"""Aggregate stop/continue online records and verify shared prefixes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.common import sha256_array, sha256_file
from recovery_handback.hb3p.metrics import episode_value
from recovery_handback.hb3p_stop.io import atomic_json_dump, read_jsonl, write_csv, write_jsonl


METHODS = ("NONE", "FIXED_L60", "FIXED_L80", "LEARNED_STOP_CONTINUE")


def _online_files(root: Path, method: str) -> list[Path]:
    return sorted(
        path
        for path in Path(root).glob(f"**/{method}/*.json")
        if not path.name.endswith("_decision_trace.json")
        and not path.name.endswith("_handoff.json")
    )


def _prefix_key(path: Path) -> tuple[str, str]:
    with np.load(path, allow_pickle=False) as archive:
        states = np.asarray(archive["states"], dtype=np.float64)
        actions = np.asarray(archive["base_actions"], dtype=np.float32)
    return sha256_array(states[:21]), sha256_array(actions[:20])


def _predecision_key(path: Path) -> tuple[int, str, int, str]:
    with np.load(path, allow_pickle=False) as archive:
        states = np.asarray(archive["states"], dtype=np.float64)
        base_actions = np.asarray(archive["base_actions"], dtype=np.float32)
    state_count = min(81, len(states))
    action_count = min(81, len(base_actions))
    return state_count, sha256_array(states[:state_count]), action_count, sha256_array(base_actions[:action_count])


def _trajectory_key(path: Path) -> tuple:
    with np.load(path, allow_pickle=False) as archive:
        names = ("states", "actions", "base_actions", "helper_mask", "success")
        return tuple((name, tuple(archive[name].shape), sha256_array(np.asarray(archive[name]))) for name in names)


def aggregate(roots_path: Path, episodes_root: Path, output_dir: Path, protocol: dict) -> dict:
    roots = read_jsonl(Path(roots_path))
    expected = list(map(int, protocol["test_seeds"]))
    if [int(row["seed"]) for row in roots] != expected:
        raise RuntimeError("root manifest seed order mismatch")
    roots_by_id = {str(row["root_id"]): row for row in roots}
    if len(roots_by_id) != len(roots):
        raise RuntimeError("test root IDs are not unique")
    records = {}
    failures = []
    for method in METHODS[1:]:
        files = _online_files(episodes_root, method)
        for path in files:
            row = json.loads(path.read_text(encoding="utf-8"))
            key = (str(row.get("root_id")), method)
            if key in records:
                raise RuntimeError(f"duplicate online record: {key}")
            root = roots_by_id.get(key[0])
            if root is None:
                failures.append({"method": method, "path": str(path), "reason": "unknown_root"})
                continue
            if row.get("method_id") != method or row.get("protocol_hash") != protocol["protocol_sha256"]:
                failures.append({"method": method, "root_id": key[0], "reason": "identity_mismatch"})
            elif not row.get("engineering_ok"):
                failures.append({"method": method, "root_id": key[0], "reason": row.get("exception_reason")})
            elif not Path(row.get("trajectory_path", "")).is_file():
                failures.append({"method": method, "root_id": key[0], "reason": "trajectory_missing"})
            records[key] = row
    missing = [{"root_id": root_id, "method": method} for root_id in roots_by_id for method in METHODS[1:] if (root_id, method) not in records]
    rows = []
    for root_id, root in roots_by_id.items():
        rows.append({
            "schema_version": "hb3p_stop_continue_aggregate_episode_v1", "task": "square",
            "role": root.get("role", protocol.get("role")), "root_id": root_id,
            "stat_group_id": root.get("stat_group_id", root_id), "method_id": "NONE",
            "protocol_hash": protocol["protocol_sha256"], "semantic_pair_id": protocol["semantic_pair_id"],
            "engineering_ok": root.get("exception_reason") is None, "system_success": bool(root["baseline_success"]),
            "autonomous_completion": bool(root["baseline_success"]), "genuine_handoff_success": False,
            "takeover_count": 0, "takeover_t": None, "selected_length": 0, "helper_steps_actual": 0,
            "repair_policy_calls": 0, "repair_calls_after_handoff": 0, "changed_action_steps": 0,
            "first_raw_success_state": root.get("first_raw_success_state"), "stable_success_state": root.get("stable_success_state"),
            "handoff_executed": False, "handoff_t": None, "helper_completed_task": False,
            "first_success_wait_after_handoff": None, "autonomous_execution_steps_after_handoff": None,
            "query_count": 0, "queried_times": [], "inference_wall_seconds": 0.0,
            "root_seed": int(root["seed"]), "initial_state_hash": root.get("initial_state_hash"),
            "trajectory_path": root.get("rollout_path"), "decision_trace_path": None,
            "utility": float(root["baseline_success"]),
        })
        for method in METHODS[1:]:
            source = records.get((root_id, method))
            if source is None:
                continue
            row = dict(source)
            row["autonomous_completion"] = bool(row.get("genuine_handoff_success")) if int(row.get("takeover_count", 0)) else bool(row.get("system_success"))
            row["utility"] = episode_value(row, float(protocol["lambda"]), float(protocol["horizon_steps"]))
            row["root_id"] = root_id
            rows.append(row)
    prefix_rows, prefix_failures, parity_rows, parity_failures = [], [], [], []
    for root_id, root in roots_by_id.items():
        with np.load(root["rollout_path"], allow_pickle=False) as archive:
            baseline_states = np.asarray(archive["states"], dtype=np.float64)
            baseline_actions = np.asarray(archive["suggestions"], dtype=np.float32)
        keys = {"NONE": (sha256_array(baseline_states[:21]), sha256_array(baseline_actions[:20]))}
        for method in METHODS[1:]:
            source = records.get((root_id, method))
            if source is not None:
                keys[method] = _prefix_key(Path(source["trajectory_path"]))
        shared_t20 = len(keys) == 4 and len(set(keys.values())) == 1
        helper_predecision = {
            method: _predecision_key(Path(records[(root_id, method)]["trajectory_path"]))
            for method in METHODS[1:] if (root_id, method) in records
        }
        shared_t80 = len(helper_predecision) == 3 and len(set(helper_predecision.values())) == 1
        prefix_rows.append({
            "root_id": root_id, "shared_t20_verified": shared_t20,
            "helper_predecision_t80_verified": shared_t80,
            "state_hash_t20": keys["NONE"][0], "action_hash_t20": keys["NONE"][1],
        })
        if not shared_t20 or not shared_t80:
            prefix_failures.append({"root_id": root_id, "keys": keys})
        learned = records.get((root_id, "LEARNED_STOP_CONTINUE"))
        comparator = None
        parity = False
        if learned is not None:
            comparator = "FIXED_L60" if int(learned["selected_length"]) == 60 else "FIXED_L80"
            fixed = records.get((root_id, comparator))
            parity = fixed is not None and _trajectory_key(Path(learned["trajectory_path"])) == _trajectory_key(Path(fixed["trajectory_path"]))
        parity_rows.append({
            "root_id": root_id, "learned_decision": learned.get("stop_continue_decision") if learned else None,
            "selected_length": learned.get("selected_length") if learned else None,
            "matched_fixed_method": comparator, "full_trajectory_parity": parity,
        })
        if not parity:
            parity_failures.append({"root_id": root_id, "matched_fixed_method": comparator})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_dir / "episodes.jsonl")
    write_csv(prefix_rows, output_dir / "prefix_checks.csv")
    write_csv(parity_rows, output_dir / "branch_parity.csv")
    atomic_json_dump({
        "schema_version": "hb3p_stop_continue_coverage_v1",
        "protocol_sha256": protocol["protocol_sha256"], "expected_roots": len(roots),
        "methods": list(METHODS), "expected_records": len(roots) * len(METHODS), "complete_records": len(rows),
        "unique_rollouts": len(roots) + len(records), "missing_records": missing, "engineering_failures": failures,
        "shared_t20_verified_roots": sum(bool(row["shared_t20_verified"]) for row in prefix_rows),
        "helper_predecision_t80_verified_roots": sum(bool(row["helper_predecision_t80_verified"]) for row in prefix_rows),
        "prefix_verified_roots": sum(bool(row["shared_t20_verified"] and row["helper_predecision_t80_verified"]) for row in prefix_rows),
        "branch_parity_verified_roots": sum(bool(row["full_trajectory_parity"]) for row in parity_rows),
        "prefix_failures": prefix_failures, "branch_parity_failures": parity_failures,
        "complete": not missing and not failures and not prefix_failures and not parity_failures and len(rows) == len(roots) * len(METHODS),
    }, output_dir / "coverage.json")
    if missing or failures or prefix_failures or parity_failures:
        raise RuntimeError("stop/continue aggregation failed coverage or prefix checks")
    return {"records": len(rows), "unique_rollouts": len(roots) + len(records), "prefix_verified_roots": len(roots), "branch_parity_verified_roots": len(roots)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol["protocol_sha256"] = sha256_file(args.protocol)
    print(json.dumps(aggregate(args.roots, args.episodes_root, args.output_dir, protocol), indent=2))


if __name__ == "__main__":
    main()
