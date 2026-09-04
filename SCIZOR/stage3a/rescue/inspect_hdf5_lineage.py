"""Compare frozen benchmark clean rollouts with their source demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return text(value)


def sha_text(value):
    if value is None:
        return None
    return hashlib.sha256(str(text(value)).encode("utf-8")).hexdigest()


def arr_stats(actual, expected):
    n = min(len(actual), len(expected))
    if n <= 0:
        return {"n": 0}
    delta = np.asarray(actual[:n], dtype=np.float64) - np.asarray(expected[:n], dtype=np.float64)
    row_l2 = np.linalg.norm(delta.reshape(n, -1), axis=1)
    return {
        "n": int(n),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "median_row_l2": float(np.median(row_l2)),
        "p95_row_l2": float(np.quantile(row_l2, 0.95)),
    }


def clean_groups(benchmark, task):
    rows = []
    for key in benchmark["data"].keys():
        group = benchmark[f"data/{key}"]
        if text(group.attrs.get("variant", "")) == "clean" and text(group.attrs.get("task", "")) == task:
            rows.append((key, text(group.attrs["base_demo_id"])))
    return sorted(rows, key=lambda row: int(row[1].split("_")[-1]))


def inspect_task(benchmark, source_path, task, max_demos):
    rows = []
    with h5py.File(source_path, "r") as source:
        for benchmark_id, base_demo_id in clean_groups(benchmark, task)[:max_demos]:
            benchmark_group = benchmark[f"data/{benchmark_id}"]
            source_group = source[f"data/{base_demo_id}"]
            benchmark_actions = benchmark_group["actions"][:]
            source_actions = source_group["actions"][:]
            benchmark_states = benchmark_group["states"][:]
            source_states = source_group["states"][:]
            row = {
                "task": task,
                "benchmark_group": benchmark_id,
                "base_demo_id": base_demo_id,
                "benchmark_actions_shape": list(benchmark_actions.shape),
                "source_actions_shape": list(source_actions.shape),
                "benchmark_states_shape": list(benchmark_states.shape),
                "source_states_shape": list(source_states.shape),
                "benchmark_action_dtype": str(benchmark_actions.dtype),
                "source_action_dtype": str(source_actions.dtype),
                "benchmark_state_dtype": str(benchmark_states.dtype),
                "source_state_dtype": str(source_states.dtype),
                "actions_same_index": arr_stats(benchmark_actions, source_actions),
                "states_same_index": arr_stats(benchmark_states, source_states),
                "states_source_plus1": arr_stats(benchmark_states[:-1], source_states[1:]),
                "states_source_minus1": arr_stats(benchmark_states[1:], source_states[:-1]),
                "benchmark_model_sha256": sha_text(benchmark_group.attrs.get("model_file")),
                "source_model_sha256": sha_text(source_group.attrs.get("model_file")),
                "benchmark_source_dataset_attr": text(benchmark_group.attrs.get("source_dataset", "")),
            }
            image_key = "obs/agentview_image"
            if image_key in benchmark_group and image_key in source_group:
                benchmark_images = benchmark_group[image_key][:]
                source_images = source_group[image_key][:]
                row["images_same_index"] = arr_stats(benchmark_images, source_images)
                row["images_source_plus1"] = arr_stats(benchmark_images[:-1], source_images[1:])
            rows.append(row)
        env_args = text(source["data"].attrs.get("env_args", ""))
    return {"task": task, "source": str(source_path), "env_args": json.loads(env_args) if env_args else {}, "rows": rows}


def aggregate(task_result):
    rows = task_result["rows"]

    def median(stat_path):
        values = []
        for row in rows:
            current = row
            for key in stat_path:
                current = current.get(key, {}) if isinstance(current, dict) else {}
            if isinstance(current, (int, float)):
                values.append(float(current))
        return float(np.median(values)) if values else None

    candidates = {
        "same_index": median(("states_same_index", "median_row_l2")),
        "source_plus1": median(("states_source_plus1", "median_row_l2")),
        "source_minus1": median(("states_source_minus1", "median_row_l2")),
    }
    usable = {name: value for name, value in candidates.items() if value is not None}
    best = min(usable, key=usable.get) if usable else None
    return {
        "num_demos": len(rows),
        "median_action_max_abs": median(("actions_same_index", "max_abs")),
        "best_state_relation": best,
        "best_state_median_row_l2": usable.get(best) if best else None,
        "model_file_all_match": bool(rows) and all(row["benchmark_model_sha256"] == row["source_model_sha256"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--can-source", type=Path, required=True)
    parser.add_argument("--square-source", type=Path, required=True)
    parser.add_argument("--max-demos", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.benchmark, "r") as benchmark:
        result = {
            "benchmark": str(args.benchmark),
            "benchmark_root_attrs": {key: text(value) for key, value in benchmark["data"].attrs.items()},
            "tasks": {
                "can": inspect_task(benchmark, args.can_source, "can", args.max_demos),
                "square": inspect_task(benchmark, args.square_source, "square", args.max_demos),
            },
        }
    result["summary"] = {task: aggregate(data) for task, data in result["tasks"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
