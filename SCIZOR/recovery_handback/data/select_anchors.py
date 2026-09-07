"""Select fixed-time anchors from complete natural policy rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.common import read_table, write_table

def _read_roots(path: Path):
    return read_table(path)


def select(config, task, role, roots_path, output):
    rows = []
    for root in _read_roots(roots_path):
        rollout = np.load(root["rollout_path"], allow_pickle=False)
        successes = np.asarray(rollout["success"], dtype=bool)
        for anchor_t in sorted(int(x) for x in config["anchor_times"]):
            if anchor_t >= len(successes) or bool(successes[:anchor_t].any()):
                continue
            if int(config["horizon_steps"]) - anchor_t < int(config["minimum_remaining_steps"]):
                continue
            rows.append({
                "anchor_id": f"{root['root_id']}:{anchor_t}", "task": task, "role": role,
                "root_id": root["root_id"], "stat_group_id": root.get("stat_group_id", root["root_id"]),
                "anchor_t": anchor_t, "remaining_steps": int(config["horizon_steps"]) - anchor_t,
                "rollout_path": root["rollout_path"],
                "canonical_payload_path": root.get("canonical_payload_path", root["payload_path"]),
                "anchor_history_path": root["anchor_history_path"],
                "policy_memory_check_path": root["policy_memory_path"],
                "baseline_success": bool(root["baseline_success"]),
                "initial_state_hash": root["initial_state_hash"],
                "selectable_reason": "fixed_time_before_any_success",
            })
    rows.sort(key=lambda row: (row["task"], row["role"], row["root_id"], row["anchor_t"]))
    write_table(rows, output)
    summary_path = output.with_name("anchors_summary.json")
    summary_path.write_text(json.dumps({"schema_version": "hb1_anchors_v1", "rows": len(rows),
                                        "baseline_successes": sum(bool(row["baseline_success"]) for row in rows),
                                        "task": task, "role": role}, indent=2), encoding="utf-8")
    print(json.dumps({"task": task, "role": role, "anchors": len(rows)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    select(json.loads(args.config.read_text()), args.task, args.role, args.roots, args.output)


if __name__ == "__main__":
    main()
