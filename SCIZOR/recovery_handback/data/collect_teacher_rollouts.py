"""Collect successful runtime rollouts from a privileged direct teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, write_table
from recovery_handback.data.student_schema import STUDENT_OBS_KEYS


def _write_dataset(group, name, value):
    array = np.asarray(value)
    compression = "gzip" if array.size > 1024 else None
    kwargs = {"compression": compression, "compression_opts": 1} if compression else {}
    group.create_dataset(name, data=array, **kwargs)


def _existing_records(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {
        row["root_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("can",), required=True)
    parser.add_argument("--teacher-meta", type=Path, required=True)
    parser.add_argument("--role", choices=("distill_train",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    teacher_meta = json.loads(args.teacher_meta.read_text(encoding="utf-8"))
    role = config["roles"][args.role]
    requested = int(config["capability_repair"]["can_distill"]["requested_roots"])
    expansion = int(config["capability_repair"]["can_distill"]["single_allowed_expansion_roots"])
    if args.start_index < 0 or args.count <= 0 or args.start_index + args.count > requested + expansion:
        raise ValueError("requested rollout range exceeds the one allowed expansion")
    seed_start = int(role["seed_start"])
    source = Path(assets["tasks"][args.task]["source_hdf5"])
    env = EnvAdapter(source, **load_observation_spec(source))
    teacher = BasePolicyAdapter(teacher_meta["checkpoint"], device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = args.output_dir / "successful_rollouts.hdf5"
    records_path = args.output_dir / "records.jsonl"
    records = _existing_records(records_path)
    source_env_args = None
    with h5py.File(source, "r") as source_file:
        source_env_args = source_file["data"].attrs.get("env_args")

    try:
        with h5py.File(hdf5_path, "a") as output:
            data = output.require_group("data")
            if source_env_args is not None:
                data.attrs["env_args"] = source_env_args
            for index in range(args.start_index, args.start_index + args.count):
                seed = seed_start + index
                root_id = f"{args.task}:{args.role}:{seed}"
                demo_id = f"runtime_teacher_{index:04d}"
                observations = {key: [] for key in STUDENT_OBS_KEYS}
                next_observations = {key: [] for key in STUDENT_OBS_KEYS}
                states = []
                actions = []
                rewards = []
                successes = []
                exception_reason = None
                stable_state = None
                payload = None
                try:
                    obs, payload = env.new_episode(seed)
                    teacher.start_episode()
                    stable_history = []
                    for t in range(int(config["horizon_steps"])):
                        action = teacher.suggest_once(obs, t, root_id)
                        if action.shape != (7,):
                            raise RuntimeError(f"teacher action shape is {action.shape}, expected (7,)")
                        states.append(env.physical_state())
                        for key in STUDENT_OBS_KEYS:
                            observations[key].append(np.asarray(obs[key]).copy())
                        obs_next, reward, success, _info = env.step(action)
                        for key in STUDENT_OBS_KEYS:
                            next_observations[key].append(np.asarray(obs_next[key]).copy())
                        actions.append(action.copy())
                        rewards.append(float(reward))
                        successes.append(bool(success))
                        stable_history.append(bool(success))
                        required = int(config["success_consecutive_steps"])
                        if len(stable_history) > required:
                            stable_history.pop(0)
                        if len(stable_history) == required and all(stable_history):
                            stable_state = t + 1
                            break
                        obs = obs_next
                except Exception as exc:
                    exception_reason = f"{type(exc).__name__}: {exc}"

                if demo_id in data:
                    del data[demo_id]
                if stable_state is not None and exception_reason is None:
                    demo = data.create_group(demo_id)
                    _write_dataset(demo, "actions", np.asarray(actions, dtype=np.float32))
                    _write_dataset(demo, "states", np.asarray(states, dtype=np.float64))
                    _write_dataset(demo, "rewards", np.asarray(rewards, dtype=np.float32))
                    dones = np.zeros(len(actions), dtype=np.int64)
                    if len(dones):
                        dones[-1] = 1
                    _write_dataset(demo, "dones", dones)
                    obs_group = demo.create_group("obs")
                    next_group = demo.create_group("next_obs")
                    for key in STUDENT_OBS_KEYS:
                        _write_dataset(obs_group, key, observations[key])
                        _write_dataset(next_group, key, next_observations[key])
                    demo.attrs["num_samples"] = len(actions)
                    demo.attrs["model_file"] = payload["model"]
                    demo.attrs["hb1_root_id"] = root_id
                    demo.attrs["hb1_seed"] = seed
                records[root_id] = {
                    "schema_version": "hb1_teacher_rollout_v1",
                    "task": args.task,
                    "role": args.role,
                    "root_id": root_id,
                    "seed": seed,
                    "demo_id": demo_id if stable_state is not None and exception_reason is None else None,
                    "steps": len(actions),
                    "stable_success_state": stable_state,
                    "success": stable_state is not None,
                    "exception_reason": exception_reason,
                    "used_for_student_supervision": stable_state is not None and exception_reason is None,
                }
                data.attrs["total"] = sum(
                    int(data[key].attrs["num_samples"]) for key in data
                )
                output.flush()
                ordered = sorted(records.values(), key=lambda row: row["seed"])
                write_table(ordered, records_path)
                atomic_json_dump({
                    "schema_version": "hb1_teacher_rollout_summary_v1",
                    "task": args.task,
                    "role": args.role,
                    "roots": len(ordered),
                    "successes": sum(bool(row["success"]) for row in ordered),
                    "engineering_failures": sum(bool(row["exception_reason"]) for row in ordered),
                    "successful_rollouts_hdf5": str(hdf5_path.resolve()),
                    "student_observation_keys": list(STUDENT_OBS_KEYS),
                    "teacher_uses_privileged_input": True,
                    "student_supervision_excludes_object": True,
                    "rows": ordered,
                }, args.output_dir / "summary.json")
    finally:
        env.close()

    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({k: summary[k] for k in ("roots", "successes", "engineering_failures")}, indent=2))


if __name__ == "__main__":
    main()
