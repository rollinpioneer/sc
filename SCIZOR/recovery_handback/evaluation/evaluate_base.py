"""Evaluate one frozen Robomimic action policy in the HB1 runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_json, write_table


def _stable_step(success_history, value: bool, state_index: int, required: int = 5):
    success_history.append(bool(value))
    if len(success_history) > required:
        del success_history[:-required]
    return len(success_history) == required and all(success_history)


def _write_image(image: np.ndarray, path: Path):
    try:
        from PIL import Image
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    except ImportError:
        # PNG is diagnostic only; the numerical rollout remains the source of truth.
        pass


def evaluate(config: dict, assets: dict, task: str, role: str, checkpoint: Path, output_dir: Path, device: str = "cuda"):
    task_cfg = config["roles"][role]
    n_roots = int(task_cfg["roots_per_task"])
    seed_start = int(task_cfg["seed_start"]) + (int(config.get("square_seed_offset", 0)) if task == "square" else 0)
    source = Path(assets["tasks"][task]["source_hdf5"])
    env = EnvAdapter(source, **load_observation_spec(source))
    policy = BasePolicyAdapter(checkpoint, device=device)
    action_low, action_high = env.action_bounds()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for index in range(n_roots):
            seed = seed_start + index
            root_id = f"{task}:{role}:{seed}"
            root_key = root_id
            obs, payload = env.new_episode(seed)
            policy.start_episode()
            states_pre, states_post, actions, base_actions, rewards, successes = [], [], [], [], [], []
            stable_history = []
            first_image_written = False
            stable_state = None
            exception_reason = None
            try:
                for absolute_t in range(int(config["horizon_steps"])):
                    states_pre.append(env.physical_state())
                    base_action = policy.suggest_once(obs, absolute_t, root_key)
                    if not first_image_written:
                        for key in ("agentview_image", "robot0_eye_in_hand_image"):
                            if key in obs:
                                _write_image(obs[key], output_dir / f"first_{key}.png")
                        first_image_written = True
                    obs_next, reward, raw_success, _info = env.step(base_action)
                    states_post.append(env.physical_state())
                    actions.append(base_action.copy())
                    base_actions.append(base_action.copy())
                    rewards.append(float(reward))
                    successes.append(bool(raw_success))
                    if _stable_step(stable_history, raw_success, absolute_t + 1):
                        stable_state = absolute_t + 1
                        obs = obs_next
                        break
                    obs = obs_next
            except Exception as exc:
                exception_reason = f"{type(exc).__name__}: {exc}"

            payload_dir = output_dir / "payloads"
            payload_dir.mkdir(exist_ok=True)
            payload_path = payload_dir / f"{index:03d}.npz"
            np.savez_compressed(payload_path, states=np.asarray(payload["states"], dtype=np.float64))
            (payload_dir / f"{index:03d}.xml").write_text(payload["model"], encoding="utf-8")
            (payload_dir / f"{index:03d}.json").write_text(json.dumps({
                "seed": seed,
                "env_name": payload["env_name"],
                "control_freq": payload["control_freq"],
                "ep_meta": payload.get("ep_meta"),
            }, sort_keys=True, indent=2), encoding="utf-8")
            rollout_path = output_dir / f"rollout_{index:03d}.npz"
            np.savez_compressed(
                rollout_path,
                actions=np.asarray(actions, dtype=np.float32),
                base_actions=np.asarray(base_actions, dtype=np.float32),
                states_pre=np.asarray(states_pre, dtype=np.float64),
                states_post=np.asarray(states_post, dtype=np.float64),
                states=(
                    np.concatenate((np.asarray(states_pre[:1], np.float64), np.asarray(states_post, np.float64)), axis=0)
                    if states_pre and states_post else np.asarray(states_pre, np.float64)
                ),
                rewards=np.asarray(rewards, dtype=np.float32),
                success=np.asarray(successes, dtype=np.bool_),
            )
            rows.append({
                "schema_version": "hb1_base_eval_v1",
                "task": task,
                "role": role,
                "root_id": root_id,
                "seed": seed,
                "payload_path": str(payload_path.resolve()),
                "rollout_path": str(rollout_path.resolve()),
                "actual_steps": len(actions),
                "first_raw_success_state": next((i + 1 for i, value in enumerate(successes) if value), None),
                "stable_success_state": stable_state,
                "baseline_success": stable_state is not None,
                "action_dim": int(actions[0].shape[0]) if actions else 0,
                "policy_memory_calls": policy._calls,
                "exception_reason": exception_reason,
            })
    finally:
        env.close()
    identity = policy.actor_identity()
    summary = {
        "schema_version": "hb1_base_eval_summary_v1",
        "task": task,
        "role": role,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "algorithm": identity.get("algo_name"),
        "observation_shapes": identity.get("observation_shapes", {}),
        "action_dim": identity.get("action_dim"),
        "action_bounds": [action_low.tolist(), action_high.tolist()],
        "runtime_fingerprint_id": sha256_json({"source": str(source.resolve()), "control_freq": env.control_freq}),
        "roots": len(rows),
        "successes": sum(bool(row["baseline_success"]) for row in rows),
        "success_rate": float(sum(bool(row["baseline_success"]) for row in rows) / len(rows)) if rows else 0.0,
        "rows": rows,
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    write_table(rows, output_dir / "rows.parquet")
    write_table(rows, output_dir / "rows.jsonl")
    print(json.dumps({k: summary[k] for k in ("task", "checkpoint", "roots", "successes", "success_rate")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--role", default="base_val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    evaluate(config, assets, args.task, args.role, args.checkpoint, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
