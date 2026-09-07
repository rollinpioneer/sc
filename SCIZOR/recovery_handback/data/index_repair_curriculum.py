"""Index fixed repair-train prefixes for the teacher-centered SAC curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, read_table, write_table
from recovery_handback.train.residual_env import load_payload
from recovery_handback.train.stage_potential import stage_potential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("square",), required=True)
    parser.add_argument("--base-policy-json", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    base_record = json.loads(args.base_policy_json.read_text(encoding="utf-8"))
    checkpoint = base_record.get("checkpoint") or base_record["base"]["checkpoint"]
    roots = [row for row in read_table(args.roots) if not row.get("exception_reason")]
    source = Path(assets["tasks"][args.task]["source_hdf5"])
    env = EnvAdapter(source, **load_observation_spec(source))
    base = BasePolicyAdapter(checkpoint, device=args.device)
    horizon = int(config["horizon_steps"])
    minimum_remaining = max(100, int(config["minimum_remaining_steps"]))
    rows = []
    try:
        for root in roots:
            with np.load(root["rollout_path"], allow_pickle=False) as handle:
                actions = np.asarray(handle["actions"], dtype=np.float32)
                successes = np.asarray(handle["success"], dtype=bool)
            obs = env.reset_canonical(load_payload(root), int(root["seed"]))
            base.start_episode()
            stable_history = []
            maximum = min(len(actions), horizon - minimum_remaining)
            for t in range(maximum + 1):
                stable_already = (
                    len(stable_history) >= int(config["success_consecutive_steps"])
                    and all(stable_history[-int(config["success_consecutive_steps"]):])
                )
                if t % 10 == 0 and not stable_already:
                    stage = env.staged_rewards()
                    rows.append({
                        "root_id": root["root_id"],
                        "stat_group_id": root.get("stat_group_id", root["root_id"]),
                        "seed": int(root["seed"]),
                        "prefix_t": t,
                        "stage_vector": stage.tolist(),
                        "stage_potential": stage_potential(stage),
                        "remaining_steps": horizon - t,
                        "rollout_path": root["rollout_path"],
                        "canonical_payload_path": root.get(
                            "canonical_payload_path", root["payload_path"]
                        ),
                    })
                if t == maximum:
                    break
                base.suggest_once(obs, t, root["root_id"])
                obs, _reward, success, _info = env.step(actions[t])
                stable_history.append(bool(successes[t]) if t < len(successes) else bool(success))
    finally:
        env.close()

    rows.sort(key=lambda row: (-row["stage_potential"], row["root_id"], row["prefix_t"]))
    count = len(rows)
    easy_count = max(1, int(np.ceil(0.25 * count))) if count else 0
    medium_count = max(easy_count, int(np.ceil(0.60 * count))) if count else 0
    for index, row in enumerate(rows):
        row["rank"] = index
        row["rank_fraction"] = index / max(1, count - 1)
        row["easy"] = index < easy_count
        row["medium"] = index < medium_count
    if not rows:
        raise RuntimeError("repair curriculum index is empty")
    write_table(rows, args.output)
    summary = {
        "schema_version": "hb1_repair_curriculum_v1",
        "task": args.task,
        "roots": len({row["root_id"] for row in rows}),
        "states": count,
        "easy_states": easy_count,
        "medium_states": medium_count,
        "stage_dimensions": sorted({len(row["stage_vector"]) for row in rows}),
        "potential_min": min(row["stage_potential"] for row in rows),
        "potential_max": max(row["stage_potential"] for row in rows),
        "source_role": "repair_train",
    }
    atomic_json_dump(summary, args.output.with_name("curriculum_summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
