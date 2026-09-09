"""Validate the HB3-P online loop on the four frozen HB2 pilot roots."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.hb2.features import _anchor_history
from recovery_handback.hb2.metrics import choose_length
from recovery_handback.hb3p.online_episode import OnlineRunner
from recovery_handback.hb3p.predictors import arrays_sha256, history_arrays


PROBABILITY_FIELDS = (
    "p0", "p_full", "p_sys_5", "p_genuine_5", "p_sys_20",
    "p_genuine_20", "p_sys_80", "p_genuine_80",
)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _safe(value: str) -> str:
    return value.replace(":", "__").replace("/", "_")


def _history(row: dict, horizon: int) -> list[dict]:
    frames, actions, times = _anchor_history(row, horizon)
    return [
        {
            "absolute_t": int(round(float(times[index]) * horizon)),
            "obs": frames[index],
            "base_action": actions[index],
        }
        for index in range(len(frames))
    ]


def _branch_record(hb2_root: Path, root_id: str, absolute_t: int, length: int) -> tuple[Path, Path]:
    directory = f"{_safe(root_id)}__{absolute_t}"
    matches = sorted((hb2_root / "branches/hb2_pilot").glob(f"shard_*/records/{directory}/l{length}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one HB2 pilot branch for {root_id}:{absolute_t}:l{length}, got {matches}")
    return matches[0], matches[0].with_suffix(".npz")


def _baseline_checks(root: dict, result: dict, limits: dict) -> dict:
    with np.load(root["rollout_path"], allow_pickle=False) as old, np.load(result["trajectory_path"], allow_pickle=False) as new:
        action_diff = _max_abs(new["actions"], old["actions"])
        suggestion_diff = _max_abs(new["base_actions"], old["suggestions"])
        state_diff = _max_abs(new["states"], old["states"])
        success_equal = np.array_equal(new["success"], old["success"])
    passed = (
        action_diff <= float(limits["prefix_action_max_abs"])
        and suggestion_diff <= float(limits["prefix_action_max_abs"])
        and state_diff <= float(limits["prefix_state_max_abs"])
        and success_equal
    )
    return {
        "pass": passed,
        "action_max_abs": action_diff,
        "base_suggestion_max_abs": suggestion_diff,
        "state_max_abs": state_diff,
        "success_sequence_equal": success_equal,
    }


def _branch_checks(hb2_root: Path, root: dict, result: dict, limits: dict) -> dict:
    if not result["takeover_count"]:
        return {"applicable": False, "pass": True}
    takeover = int(result["takeover_t"])
    length = int(result["selected_length"])
    record_path, trajectory_path = _branch_record(hb2_root, result["root_id"], takeover, length)
    old_record = json.loads(record_path.read_text(encoding="utf-8"))
    with (
        np.load(root["rollout_path"], allow_pickle=False) as baseline,
        np.load(trajectory_path, allow_pickle=False) as old,
        np.load(result["trajectory_path"], allow_pickle=False) as new,
    ):
        prefix_action_diff = _max_abs(new["actions"][:takeover], baseline["actions"][:takeover])
        prefix_base_diff = _max_abs(new["base_actions"][:takeover], baseline["suggestions"][:takeover])
        prefix_state_diff = _max_abs(new["states"][:takeover + 1], baseline["states"][:takeover + 1])
        replay_action_diff = _max_abs(new["actions"][takeover:], old["actions"])
        replay_base_diff = _max_abs(new["base_actions"][takeover:], old["base_actions"])
        replay_state_diff = _max_abs(new["states"][takeover:], old["states"])
        online_active = (np.flatnonzero(new["helper_mask"]) + 0).tolist()
        reference_active = (np.flatnonzero(old["helper_mask"]) + takeover).tolist()
        success_equal = np.array_equal(new["success"][takeover:], old["success"])
    online_interval_valid = online_active == list(
        range(takeover, takeover + int(result["helper_steps_actual"]))
    )
    reference_interval_valid = reference_active == list(
        range(takeover, takeover + int(old_record["helper_steps_actual"]))
    )
    planned_control_equal = (
        int(old_record["anchor_t"]) == takeover
        and int(old_record["repair_length"]) == length
    )
    handoff_boundaries_valid = all(
        not row.get("handoff_executed")
        or int(row["handoff_state_index"]) == takeover + length
        for row in (result, old_record)
    )
    control_semantics_equal = (
        planned_control_equal
        and online_interval_valid
        and reference_interval_valid
        and handoff_boundaries_valid
        and int(result["repair_policy_calls"]) == int(result["helper_steps_actual"])
        and int(old_record["repair_policy_calls"]) == int(old_record["helper_steps_actual"])
        and int(result["repair_calls_after_handoff"]) == 0
        and int(old_record["repair_calls_after_handoff"]) == 0
    )
    task_outcome_equal = all(
        result.get(key) == old_record.get(key) for key in (
            "system_success", "first_raw_success_state", "stable_success_state",
            "genuine_handoff_success",
        )
    )
    passed = (
        bool(old_record.get("engineering_ok"))
        and prefix_action_diff <= float(limits["prefix_action_max_abs"])
        and prefix_base_diff <= float(limits["prefix_action_max_abs"])
        and prefix_state_diff <= float(limits["prefix_state_max_abs"])
        and control_semantics_equal
    )
    return {
        "applicable": True,
        "pass": passed,
        "reference_record": str(record_path.resolve()),
        "prefix_action_max_abs": prefix_action_diff,
        "prefix_base_action_max_abs": prefix_base_diff,
        "prefix_state_max_abs": prefix_state_diff,
        "planned_control_equal": planned_control_equal,
        "online_helper_interval": online_active,
        "reference_helper_interval": reference_active,
        "online_interval_valid": online_interval_valid,
        "reference_interval_valid": reference_interval_valid,
        "handoff_boundaries_valid": handoff_boundaries_valid,
        "control_semantics_equal": control_semantics_equal,
        "reference_observation_tape_hits": old_record.get("observation_tape_hits"),
        "online_observation_tape_hits": result.get("observation_tape_hits"),
        "post_takeover_replay_diagnostic": {
            "gated": False,
            "reason": "old fixed branches reused renderings accumulated by earlier branch executions",
            "action_max_abs": replay_action_diff,
            "base_action_max_abs": replay_base_diff,
            "state_max_abs": replay_state_diff,
        },
        "success_sequence_equal": success_equal,
        "task_outcome_equal": task_outcome_equal,
    }


def _prediction_checks(
    runner: OnlineRunner,
    result: dict,
    anchor_by_key: dict[tuple[str, int], dict],
    horizon: int,
) -> list[dict]:
    trace = json.loads(Path(result["decision_trace_path"]).read_text(encoding="utf-8"))["records"]
    output = []
    for record in trace:
        absolute_t = int(record["absolute_t"])
        row = anchor_by_key[(result["root_id"], absolute_t)]
        history = _history(row, horizon)
        expected_hash = arrays_sha256(history_arrays(history, horizon))
        model = str(record["model"])
        if model == "M1_proprio":
            offline = runner.m1.predict(history, absolute_t)
        elif model == "M4_paired":
            if runner.m4 is None:
                raise RuntimeError("M4 pilot parity requires the feature worker")
            offline = runner.m4.predict(
                history, absolute_t, "M4_GRID_PILOT_PARITY", result["root_id"]
            )
        else:
            offline = runner.m0.predict(history, absolute_t)
        differences = {
            key: abs(float(record["probabilities"][key]) - float(offline[key]))
            for key in PROBABILITY_FIELDS
        }
        decision = runner.protocol["decision"]
        offline_length = choose_length(
            offline,
            float(decision["primary_lambda"]),
            denominator=float(decision["cost_denominator"]),
            tolerance=float(decision["tie_tolerance"]),
        )
        max_diff = max(differences.values())
        tolerance = 5e-5 if model == "M1_proprio" else 1e-6
        output.append({
            "absolute_t": absolute_t,
            "model": model,
            "pass": (
                record.get("history_sha256") == expected_hash
                and max_diff <= tolerance
                and int(record["reported_selected_length"]) == int(offline_length)
            ),
            "history_sha256_online": record.get("history_sha256"),
            "history_sha256_offline": expected_hash,
            "max_probability_abs_diff": max_diff,
            "probability_tolerance": tolerance,
            "online_selected_length": record.get("reported_selected_length"),
            "offline_selected_length": offline_length,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pilot-roots", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    roots = read_table(args.pilot_roots / "roots.parquet")
    if len(roots) != 4:
        raise ValueError(f"HB3-P pilot requires exactly four frozen HB2 roots, got {len(roots)}")
    anchors = read_table(args.pilot_roots / "anchors.parquet")
    anchor_by_key = {(str(row["root_id"]), int(row["anchor_t"])): row for row in anchors}
    hb2_root = args.pilot_roots.parents[1]
    methods = ["NONE", "S_STAR", "M1_GRID", "M4_GRID"]
    runner = OnlineRunner(args.config, args.protocol, args.queue_dir)
    records = []
    checks = []
    for root in roots:
        for method in methods:
            output_path = args.output_dir / "episodes" / method / f"{_safe(root['root_id'])}.json"
            result = runner.run(root, method, output_path)
            baseline = _baseline_checks(root, result, config["engineering"]) if method == "NONE" else {
                "applicable": False, "pass": True,
            }
            branch = _branch_checks(hb2_root, root, result, config["engineering"])
            prediction = _prediction_checks(
                runner, result, anchor_by_key, int(protocol["horizon_steps"])
            )
            invariant_pass = (
                bool(result["engineering_ok"])
                and int(result["takeover_count"]) <= 1
                and int(result["repair_calls_after_handoff"]) == 0
                and int(result["base_policy_calls"]) == int(result["episode_end_state_index"])
                and int(result["repair_policy_calls"]) == int(result["helper_steps_actual"])
            )
            check = {
                "root_id": root["root_id"],
                "method_id": method,
                "engineering_invariants": invariant_pass,
                "baseline_parity": baseline,
                "fixed_branch_parity": branch,
                "prediction_parity": prediction,
            }
            check["pass"] = (
                invariant_pass and bool(baseline["pass"]) and bool(branch["pass"])
                and all(item["pass"] for item in prediction)
            )
            checks.append(check)
            records.append(result)
            print(json.dumps({
                "root_id": root["root_id"], "method": method, "pass": check["pass"],
                "takeover_t": result["takeover_t"], "length": result["selected_length"],
            }), flush=True)
            if not check["pass"]:
                raise RuntimeError(f"HB3-P pilot parity failed for {root['root_id']} {method}")
    summary = {
        "schema_version": "hb3p_pilot_summary_v1",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_hash": runner.protocol_hash,
        "pilot_roots": len(roots),
        "methods_executed": methods,
        "complete_records": len(records),
        "all_engineering_ok": all(row["engineering_ok"] for row in records),
        "all_checks_passed": all(row["pass"] for row in checks),
        "takeovers": sum(int(row["takeover_count"]) for row in records),
        "repair_calls_after_handoff": sum(int(row["repair_calls_after_handoff"]) for row in records),
        "mean_episode_wall_seconds": float(np.mean([row["wall_seconds"] for row in records])),
        "mean_m4_inference_wall_seconds": float(np.mean([
            row["inference_wall_seconds"] for row in records if row["method_id"] == "M4_GRID"
        ])),
        "checks": checks,
        "records": records,
        "wall_seconds": time.time() - started,
    }
    atomic_json_dump(summary, args.output_dir / "summary.json")
    experiment_root = args.output_dir.parent.parent
    if summary["all_checks_passed"]:
        atomic_json_dump({"completed": True, "summary": str((args.output_dir / 'summary.json').resolve())},
                         experiment_root / "status/hb3p-C.done")
        atomic_json_dump({"completed": True, "summary": str((args.output_dir / 'summary.json').resolve())},
                         experiment_root / "status/hb3p-D-pilot.done")


if __name__ == "__main__":
    main()
