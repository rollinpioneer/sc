"""Collect natural closed-loop rollouts for HB1 training and validation roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_array, sha256_file, sha256_json, write_table


def _stable(history, value, required=5):
    history.append(bool(value))
    if len(history) > required:
        del history[:-required]
    return len(history) == required and all(history)


def collect(config, assets, task, role, base_policy_json, output_dir, *, start_index=0, count=None,
            base_device=None):
    pair = json.loads(base_policy_json.read_text(encoding="utf-8"))
    checkpoint = Path(pair["base"]["checkpoint"] if "base" in pair else pair["checkpoint"])
    role_cfg = config["roles"][role]
    role_count = int(role_cfg["roots_per_task"])
    start_index = int(start_index)
    count = role_count - start_index if count is None else int(count)
    if start_index < 0 or count < 0 or start_index + count > role_count:
        raise ValueError(
            f"requested root range [{start_index}, {start_index + count}) exceeds role size {role_count}"
        )
    seed_start = int(role_cfg["seed_start"]) + (int(config.get("square_seed_offset", 0)) if task == "square" else 0)
    source = Path(assets["tasks"][task]["source_hdf5"])
    env = EnvAdapter(source, **load_observation_spec(source))
    policy = BasePolicyAdapter(checkpoint, device=base_device or config.get("base_device", "cuda"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for index in range(start_index, start_index + count):
            seed = seed_start + index
            root_id = f"{task}:{role}:{seed}"
            obs, payload = env.new_episode(seed)
            policy.start_episode()
            states_pre, states_post, actions, rewards, successes, suggestions = [], [], [], [], [], []
            policy_observations = {}
            anchor_history = {}
            anchor_memory = {}
            history = []
            stable_history = []
            stable_state = None
            exception_reason = None
            try:
                for t in range(int(config["horizon_steps"])):
                    states_pre.append(env.physical_state())
                    for key, value in obs.items():
                        if key.endswith("_image"):
                            policy_observations.setdefault(key, []).append(
                                np.asarray(value).copy()
                            )
                    if t in set(int(x) for x in config["anchor_times"]):
                        current = {
                            "obs": {key: np.asarray(value).copy() for key, value in obs.items()
                                    if key.endswith("_image") or key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")},
                            "action": np.full(7, np.nan, dtype=np.float32),
                        }
                        keep = int(config["outputs"]["store_anchor_and_handoff_history"])
                        anchor_history[t] = (history + [current])[-keep:]
                        anchor_memory[t] = policy.memory_snapshot()
                    action = policy.suggest_once(obs, t, root_id)
                    suggestions.append(action.copy())
                    history.append({
                        "obs": {key: np.asarray(value).copy() for key, value in obs.items()
                                if key.endswith("_image") or key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")},
                        "action": action.copy(),
                    })
                    obs_next, reward, raw_success, _info = env.step(action)
                    states_post.append(env.physical_state())
                    actions.append(action.copy())
                    rewards.append(float(reward))
                    successes.append(bool(raw_success))
                    next_state_index = t + 1
                    if _stable(stable_history, raw_success):
                        stable_state = next_state_index
                        obs = obs_next
                        break
                    obs = obs_next
            except Exception as exc:  # abnormal roots remain visible and are never relabeled as failures
                exception_reason = f"{type(exc).__name__}: {exc}"

            payload_dir = output_dir / "payloads"
            payload_dir.mkdir(exist_ok=True)
            stem = f"{index:03d}"
            payload_path = payload_dir / f"{stem}.npz"
            np.savez_compressed(payload_path, states=np.asarray(payload["states"], dtype=np.float64))
            (payload_dir / f"{stem}.xml").write_text(payload["model"], encoding="utf-8")
            (payload_dir / f"{stem}.json").write_text(json.dumps({
                "seed": seed, "env_name": payload["env_name"],
                "control_freq": payload["control_freq"], "ep_meta": payload.get("ep_meta"),
            }, sort_keys=True, indent=2), encoding="utf-8")
            rollout_path = output_dir / f"rollout_{stem}.npz"
            state_sequence = (
                np.concatenate((np.asarray(states_pre[:1], np.float64), np.asarray(states_post, np.float64)), axis=0)
                if states_pre and states_post else np.asarray(states_pre, np.float64)
            )
            np.savez_compressed(
                rollout_path,
                actions=np.asarray(actions, np.float32),
                suggestions=np.asarray(suggestions, np.float32),
                states_pre=np.asarray(states_pre, np.float64),
                states_post=np.asarray(states_post, np.float64),
                states=state_sequence,
                rewards=np.asarray(rewards, np.float32),
                success=np.asarray(successes, np.bool_),
                **{
                    f"policy_obs__{key}": np.asarray(values)
                    for key, values in policy_observations.items()
                },
            )
            history_path = output_dir / f"anchor_history_{stem}.npz"
            history_arrays = {}
            history_meta = {}
            for anchor_t, frames in anchor_history.items():
                history_meta[str(anchor_t)] = []
                for frame_index, frame in enumerate(frames):
                    history_meta[str(anchor_t)].append({"frame_index": frame_index, "keys": sorted(frame["obs"])})
                    for key, value in frame["obs"].items():
                        history_arrays[f"{anchor_t}_{frame_index}_{key}"] = value
                    history_arrays[f"{anchor_t}_{frame_index}_action"] = frame["action"]
            np.savez_compressed(history_path, **history_arrays)
            (history_path.with_suffix(".json")).write_text(json.dumps(history_meta, sort_keys=True, indent=2), encoding="utf-8")
            memory_path = output_dir / f"policy_memory_{stem}.npz"
            memory_arrays = {}
            for anchor_t, snapshot in anchor_memory.items():
                hidden = snapshot.get("_rnn_hidden_state")
                if hidden is not None:
                    tensors = hidden if isinstance(hidden, (tuple, list)) else (hidden,)
                    for hidden_index, tensor in enumerate(tensors):
                        memory_arrays[f"{anchor_t}_hidden_{hidden_index}"] = np.asarray(tensor)
                memory_arrays[f"{anchor_t}_counter"] = np.asarray(snapshot.get("_rnn_counter", 0), dtype=np.int64)
            np.savez_compressed(memory_path, **memory_arrays)
            model_path = payload_dir / f"{stem}.xml"
            ep_meta_path = payload_dir / f"{stem}.json"
            initial_state_hash = sha256_array(np.asarray(payload["states"], dtype=np.float64))
            model_hash = sha256_file(model_path)
            rows.append({
                "schema_version": "hb1_root_v1", "task": task, "role": role,
                "root_id": root_id, "stat_group_id": root_id, "seed": seed,
                "initial_model_path": str(model_path.resolve()),
                "initial_state_path": str(payload_path.resolve()),
                "ep_meta_path": str(ep_meta_path.resolve()),
                "payload_path": str(payload_path.resolve()),
                "canonical_payload_path": str(payload_path.resolve()),
                "rollout_path": str(rollout_path.resolve()),
                "policy_observation_tape": "rollout_policy_obs_by_exact_state_v1",
                "anchor_history_path": str(history_path.resolve()),
                "policy_memory_path": str(memory_path.resolve()),
                "policy_checkpoint_sha256": sha256_file(checkpoint),
                "runtime_fingerprint_id": sha256_json({
                    "env_name": payload["env_name"], "control_freq": payload["control_freq"],
                    "env_args": payload.get("env_args", {}),
                }),
                "horizon_steps": int(config["horizon_steps"]), "control_freq": env.control_freq,
                "actual_steps": len(actions),
                "first_raw_success_state": next((i + 1 for i, value in enumerate(successes) if value), None),
                "stable_success_state": stable_state, "baseline_success": stable_state is not None,
                "initial_state_hash": initial_state_hash, "model_hash": model_hash,
                "exception_reason": exception_reason, "exposure_role": role,
                "used_for_policy_selection": False,
            })
        group_ids = {}
        for row in rows:
            key = (row["model_hash"], row["initial_state_hash"])
            row["stat_group_id"] = group_ids.setdefault(key, row["root_id"])
    finally:
        env.close()
    atomic_json_dump({
        "schema_version": "hb1_roots_summary_v1", "task": task, "role": role,
        "checkpoint": str(checkpoint.resolve()), "roots": len(rows),
        "baseline_successes": sum(bool(row["baseline_success"]) for row in rows), "rows": rows,
    }, output_dir / "summary.json")
    write_table(rows, output_dir / "roots.parquet")
    write_table(rows, output_dir / "roots.jsonl")
    print(json.dumps({"task": task, "role": role, "roots": len(rows),
                      "baseline_successes": sum(bool(row["baseline_success"]) for row in rows)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--role", choices=("repair_train", "repair_val", "pilot", "probe"), required=True)
    parser.add_argument("--base-policy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--base-device")
    args = parser.parse_args()
    collect(
        json.loads(args.config.read_text()), json.loads(args.assets.read_text()), args.task, args.role,
        args.base_policy_json, args.output_dir, start_index=args.start_index, count=args.count,
        base_device=args.base_device,
    )


if __name__ == "__main__":
    main()
