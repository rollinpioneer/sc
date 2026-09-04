"""Check source and v0.1 action replay using the actual Stage 1B reset semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def error_stats(actual, expected):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
    }


def sim_parts(env):
    state = env.env.sim.get_state()
    actuator_state = getattr(state, "act", None)
    if actuator_state is None:
        actuator_state = getattr(env.env.sim.data, "act", None)
    return {
        "flat": np.asarray(state.flatten()).copy(),
        "qpos": np.asarray(state.qpos).copy(),
        "qvel": np.asarray(state.qvel).copy(),
        "act": np.asarray(actuator_state).copy() if actuator_state is not None else np.zeros(0, dtype=np.float64),
        "time": np.asarray([state.time]),
    }


def expected_parts(env, expected_flat, actual_flat):
    env.env.sim.set_state_from_flattened(expected_flat)
    env.env.sim.forward()
    expected = sim_parts(env)
    env.env.sim.set_state_from_flattened(actual_flat)
    env.env.sim.forward()
    return expected


def reset_from_source_initial_state(env, source_group, restore_model):
    # Stage 1B passed only ``states``.  ``source_model`` exists only to
    # quantify the incompatible source-XML path, never to select the v0.1 path.
    payload = {"states": source_group["states"][0]}
    if restore_model == "source_model":
        payload["model"] = text(source_group.attrs["model_file"])
    env.reset()
    env.reset_to(payload)


def replay_rows(env, source_group, recorded_states, actions, state_layout, max_steps, restore_model):
    reset_from_source_initial_state(env, source_group, restore_model)
    rows = []
    limit = min(len(actions), max_steps)
    if state_layout == "source_next_state":
        limit = min(limit, max(len(recorded_states) - 1, 0))
    else:
        limit = min(limit, len(recorded_states))
    for action_t in range(limit):
        env.step(actions[action_t])
        actual = sim_parts(env)
        state_index = action_t + 1 if state_layout == "source_next_state" else action_t
        expected = expected_parts(env, recorded_states[state_index], actual["flat"])
        rows.append(
            {
                "action_t": int(action_t),
                "expected_state_index": int(state_index),
                "flat": error_stats(actual["flat"], expected["flat"]),
                "qpos": error_stats(actual["qpos"], expected["qpos"]),
                "qvel": error_stats(actual["qvel"], expected["qvel"]),
                "act": error_stats(actual["act"], expected["act"]) if len(actual["act"]) else None,
                "time": error_stats(actual["time"], expected["time"]),
            }
        )
    return rows


def summarize(rows):
    output = {"steps": len(rows)}
    for part in ("flat", "qpos", "qvel", "act", "time"):
        part_rows = [row[part] for row in rows if row.get(part) is not None]
        if part_rows:
            output[part] = {
                "median_l2": float(np.median([row["l2"] for row in part_rows])),
                "p95_l2": float(np.quantile([row["l2"] for row in part_rows], 0.95)),
                "max_abs": float(np.max([row["max_abs"] for row in part_rows])),
            }
    output["numerically_exact_all_steps"] = bool(rows) and all(row["flat"]["max_abs"] <= 1e-10 for row in rows)
    return output


def clean_groups(benchmark, task, count):
    groups = []
    for key in benchmark["data"].keys():
        group = benchmark[f"data/{key}"]
        if text(group.attrs.get("variant", "")) == "clean" and text(group.attrs.get("task", "")) == task:
            groups.append(key)
    return sorted(groups, key=lambda key: int(text(benchmark[f"data/{key}"].attrs["base_demo_id"]).split("_")[-1]))[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--restore-model", choices=("none", "source_model"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from stage1.benchmark.simulator_replay import create_shaped_env

    env, _ = create_shaped_env(str(args.source))
    result = {
        "task": args.task,
        "source": str(args.source),
        "benchmark": str(args.benchmark),
        "runtime_reset_policy": "stage1b_states_only" if args.restore_model == "none" else "source_model_for_diagnostic_only",
        "source_demos": [],
        "benchmark_demos": [],
    }
    try:
        with h5py.File(args.source, "r") as source, h5py.File(args.benchmark, "r") as benchmark:
            benchmark_ids = clean_groups(benchmark, args.task, args.num_demos)
            for benchmark_id in benchmark_ids:
                benchmark_group = benchmark[f"data/{benchmark_id}"]
                source_group = source[f"data/{text(benchmark_group.attrs['base_demo_id'])}"]
                source_rows = replay_rows(
                    env,
                    source_group,
                    source_group["states"][:],
                    source_group["actions"][:],
                    "source_next_state",
                    args.max_steps,
                    args.restore_model,
                )
                result["source_demos"].append(
                    {"demo_id": text(benchmark_group.attrs["base_demo_id"]), "summary": summarize(source_rows), "rows": source_rows}
                )
                benchmark_rows = replay_rows(
                    env,
                    source_group,
                    benchmark_group["states"][:],
                    benchmark_group["actions"][:],
                    "post_action",
                    args.max_steps,
                    args.restore_model,
                )
                result["benchmark_demos"].append(
                    {"demo_id": benchmark_id, "summary": summarize(benchmark_rows), "rows": benchmark_rows}
                )
    finally:
        env.close()
    result["summary"] = {
        "source_all_numerically_exact": bool(result["source_demos"]) and all(item["summary"]["numerically_exact_all_steps"] for item in result["source_demos"]),
        "benchmark_all_numerically_exact": bool(result["benchmark_demos"]) and all(item["summary"]["numerically_exact_all_steps"] for item in result["benchmark_demos"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
