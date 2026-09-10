"""Collect the pre-registered HB4 ordinary-success teacher cohort."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_file, write_table
from recovery_handback.data.student_schema import STUDENT_OBS_KEYS


BASE_CHECKPOINT = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth")
TEACHER_CHECKPOINT = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_teacher_epoch_100.pth")
SOURCE_HDF5 = Path("/home/__compress_data/xushijie/work/cr_scizor/data/robomimic/square/ph/image.hdf5")
RUN_ROOT = Path("/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1")
SEED_START = 920000
ROOT_COUNT = 200
HORIZON = 400
STABLE_STEPS = 5


def _write_dataset(group, name: str, values) -> None:
    array = np.asarray(values)
    kwargs = {"compression": "gzip", "compression_opts": 1} if array.size > 1024 else {}
    group.create_dataset(name, data=array, **kwargs)


def _load_records(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    return {
        int(row["seed"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def collect(start_index: int, count: int, output_dir: Path, device: str) -> dict:
    if start_index < 0 or count <= 0 or start_index + count > ROOT_COUNT:
        raise ValueError("requested range is outside the frozen 920000-920199 cohort")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    hdf5_path = output_dir / "ordinary_teacher.hdf5"
    records = _load_records(records_path)
    env = EnvAdapter(SOURCE_HDF5, **load_observation_spec(SOURCE_HDF5))
    teacher = BasePolicyAdapter(TEACHER_CHECKPOINT, device=device)
    started = time.time()
    try:
        with h5py.File(hdf5_path, "a") as output:
            data = output.require_group("data")
            for index in range(start_index, start_index + count):
                seed = SEED_START + index
                if seed in records and records[seed].get("complete_record"):
                    continue
                demo_id = f"runtime_teacher_{index:04d}"
                root_id = f"square:HB4_standard_source:{seed}"
                observations = {key: [] for key in STUDENT_OBS_KEYS}
                actions, states, rewards, successes = [], [], [], []
                first_raw_success = None
                stable_state = None
                exception_reason = None
                payload = None
                try:
                    obs, payload = env.new_episode(seed)
                    teacher.start_episode()
                    success_history = []
                    for t in range(HORIZON):
                        action = teacher.suggest_once(obs, t, root_id)
                        if action.shape != (7,):
                            raise RuntimeError(f"teacher action shape is {action.shape}, expected (7,)")
                        states.append(env.physical_state())
                        for key in STUDENT_OBS_KEYS:
                            observations[key].append(np.asarray(obs[key]).copy())
                        obs, reward, success, _ = env.step(action)
                        actions.append(action.copy())
                        rewards.append(float(reward))
                        successes.append(bool(success))
                        if first_raw_success is None and success:
                            first_raw_success = t + 1
                        success_history.append(bool(success))
                        if len(success_history) > STABLE_STEPS:
                            success_history.pop(0)
                        if len(success_history) == STABLE_STEPS and all(success_history):
                            stable_state = t + 1
                            break
                except Exception as exc:
                    exception_reason = f"{type(exc).__name__}: {exc}"

                usable_steps = first_raw_success if first_raw_success is not None else len(actions)
                usable_steps = min(int(usable_steps), len(actions))
                block_starts = [start for start in range(0, usable_steps - 9, 10)]
                enough_blocks = len(block_starts) >= 4
                valid = exception_reason is None and stable_state is not None and enough_blocks
                if demo_id in data:
                    del data[demo_id]
                if valid:
                    demo = data.create_group(demo_id)
                    _write_dataset(demo, "actions", actions)
                    _write_dataset(demo, "states", states)
                    _write_dataset(demo, "rewards", rewards)
                    dones = np.zeros(len(actions), dtype=np.int64)
                    if len(dones):
                        dones[-1] = 1
                    _write_dataset(demo, "dones", dones)
                    obs_group = demo.create_group("obs")
                    for key in STUDENT_OBS_KEYS:
                        _write_dataset(obs_group, key, observations[key])
                    demo.attrs["num_samples"] = len(actions)
                    demo.attrs["model_file"] = payload["model"]
                    demo.attrs["hb4_seed"] = seed
                    demo.attrs["first_raw_success_state"] = int(first_raw_success)
                    demo.attrs["stable_success_state"] = int(stable_state)
                records[seed] = {
                    "schema_version": "hb4_standard_teacher_rollout_v1",
                    "task": "square", "role": "HB4_standard_source", "root_id": root_id,
                    "seed": seed, "index": index, "demo_id": demo_id if valid else None,
                    "steps": len(actions), "first_raw_success_state": first_raw_success,
                    "stable_success_state": stable_state, "usable_steps_before_first_raw_success": usable_steps,
                    "candidate_block_starts": block_starts, "valid_for_matching": valid,
                    "complete_record": True, "exception_reason": exception_reason,
                    "teacher_checkpoint_sha256": sha256_file(TEACHER_CHECKPOINT),
                }
                ordered = sorted(records.values(), key=lambda row: row["seed"])
                write_table(ordered, records_path)
                atomic_json_dump({
                    "schema_version": "hb4_standard_teacher_summary_v1", "task": "square",
                    "seed_range": [SEED_START, SEED_START + ROOT_COUNT - 1],
                    "requested_roots": ROOT_COUNT, "processed_records": len(ordered),
                    "valid_for_matching": sum(bool(row["valid_for_matching"]) for row in ordered),
                    "engineering_failures": sum(bool(row["exception_reason"]) for row in ordered),
                    "teacher_checkpoint": str(TEACHER_CHECKPOINT.resolve()),
                    "source_hdf5": str(SOURCE_HDF5.resolve()), "hdf5": str(hdf5_path.resolve()),
                    "elapsed_seconds": time.time() - started,
                }, output_dir / "summary.json")
                output.flush()
    finally:
        env.close()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    return {key: summary[key] for key in ("requested_roots", "processed_records", "valid_for_matching", "engineering_failures")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=ROOT_COUNT)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "data/standard_teacher")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(collect(args.start_index, args.count, args.output_dir, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
