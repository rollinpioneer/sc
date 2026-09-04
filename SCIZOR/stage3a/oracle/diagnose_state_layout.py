"""Diagnose the one frozen state/action layout for repaired simulator replay."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from stage3a.data.transition_alignment import TransitionAlignment
from stage3a.oracle.model_xml_compat import reset_to_episode_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-hdf5", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--can-source", type=Path, required=True)
    p.add_argument("--square-source", type=Path, required=True)
    p.add_argument("--output-diagnostics", type=Path, required=True)
    p.add_argument("--output-alignment", type=Path, required=True)
    p.add_argument("--queries-per-cell", type=int, default=2)
    p.add_argument("--median-threshold", type=float, required=True)
    p.add_argument("--p95-threshold", type=float, required=True)
    return p.parse_args()


def text(value):
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def l2(actual, expected):
    return float(np.linalg.norm(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)))


class EnvPool:
    def __init__(self, sources):
        self.sources, self.source_files, self.envs = sources, {}, {}

    def source(self, task):
        if task not in self.source_files:
            self.source_files[task] = h5py.File(self.sources[task], "r")
        return self.source_files[task]

    def env(self, task):
        if task not in self.envs:
            from robomimic.utils import obs_utils
            from robomimic.envs.env_robosuite import EnvRobosuite
            if obs_utils.OBS_KEYS_TO_MODALITIES is None:
                obs_utils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
            env_args = json.loads(text(self.source(task)["data"].attrs["env_args"]))
            kw = dict(env_args["env_kwargs"])
            kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, camera_names=[], reward_shaping=True)
            kw.pop("camera_depths", None)
            self.envs[task] = EnvRobosuite(env_name=str(env_args["env_name"]), render=False, render_offscreen=False, use_image_obs=False, **kw)
        return self.envs[task]

    def close(self):
        for env in self.envs.values():
            try: env.close()
            except Exception: pass
        for source in self.source_files.values(): source.close()


def payload(states, model):
    out = {"states": states}
    if model is not None:
        out["model"] = model
    return out


def episode(benchmark, pool, row):
    group = benchmark[str(row["hdf5_group"])]
    source_group = pool.source(str(row["task"]))[f"data/{row['base_demo_id']}"]
    model = text(group.attrs.get("model_file"))
    model_source = "benchmark_group"
    if model is None:
        model = text(source_group.attrs.get("model_file"))
        model_source = "source_demo" if model is not None else "missing"
    return {"states": group["states"][:], "actions": group["actions"][:], "source_initial_state": source_group["states"][0], "model_file": model, "model_file_source": model_source}


def replay_one(env, data, action_t, alignment, replay_mode):
    states, actions = data["states"], data["actions"]
    pre_idx, next_idx = alignment.state_before_action_index(action_t), alignment.expected_next_state_index(action_t)
    if replay_mode == "direct_state_reset":
        texture_compatibility_patch = reset_to_episode_model(env, states[pre_idx], data["model_file"])
    elif replay_mode == "prefix_replay":
        texture_compatibility_patch = reset_to_episode_model(env, data["source_initial_state"], data["model_file"])
        for k in range(action_t):
            env.step(actions[k])
    else:
        raise ValueError(replay_mode)
    pre_error = l2(env.get_state()["states"], states[pre_idx])
    env.step(actions[action_t])
    return pre_error, l2(env.get_state()["states"], states[next_idx]), texture_compatibility_patch


def select(plan, n):
    unique = {}
    for row in plan: unique.setdefault(str(row["query_id"]), row)
    cells = defaultdict(list)
    for row in unique.values():
        label = "effective" if bool(row.get("is_effective_intervention", False)) else "no_effect"
        if label == "no_effect" and str(row.get("label_status", "")) == "ambiguous": continue
        cells[(str(row["task"]), label)].append(row)
    out = []
    for task in ("can", "square"):
        for label in ("effective", "no_effect"):
            rows = sorted(cells[(task, label)], key=lambda x: (str(x["pair_id"]), int(x["query_t"])))
            if len(rows) < n: raise ValueError(f"insufficient queries for {task}/{label}")
            out.extend(rows[:n])
    return out


def stats(rows, median_threshold, p95_threshold):
    pre, nxt = np.asarray([r["pre_state_l2"] for r in rows]), np.asarray([r["next_state_l2"] for r in rows])
    out = {"queries": len(rows), "pre_state_median_l2": float(np.median(pre)), "pre_state_p95_l2": float(np.quantile(pre, .95)), "next_state_median_l2": float(np.median(nxt)), "next_state_p95_l2": float(np.quantile(nxt, .95))}
    out["passed"] = bool(out["pre_state_median_l2"] < median_threshold and out["pre_state_p95_l2"] < p95_threshold and out["next_state_median_l2"] < median_threshold and out["next_state_p95_l2"] < p95_threshold)
    return out


def main():
    a = parse_args()
    plan = [json.loads(line) for line in a.plan.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = select(plan, a.queries_per_cell)
    benchmark, pool = h5py.File(a.benchmark_hdf5, "r"), EnvPool({"can": a.can_source, "square": a.square_source})
    details, summaries = {}, {}
    try:
        for mode in ("direct_state_reset", "prefix_replay"):
            for layout in ("pre_action", "post_action"):
                name, alignment, rows = f"{mode}+{layout}", TransitionAlignment.from_layout(layout, mode), []
                for query in queries:
                    data, t = episode(benchmark, pool, query), int(query["query_t"])
                    if not alignment.is_valid_action_index(t, num_states=len(data["states"]), num_actions=len(data["actions"]), minimum_future_actions=1):
                        raise ValueError(f"invalid alignment query: {name}/{query['query_id']}")
                    pre, nxt, texture_patch = replay_one(pool.env(str(query["task"])), data, t, alignment, mode)
                    rows.append({"query_id": query["query_id"], "pair_id": query["pair_id"], "task": query["task"], "failure_cell": "effective" if bool(query.get("is_effective_intervention", False)) else "no_effect", "query_t": t, "pre_action_state_index": alignment.state_before_action_index(t), "expected_next_state_index": alignment.expected_next_state_index(t), "model_file_source": data["model_file_source"], "model_file_texture_compatibility_patch": texture_patch, "pre_state_l2": pre, "next_state_l2": nxt})
                details[name], summaries[name] = rows, stats(rows, a.median_threshold, a.p95_threshold)
    finally:
        pool.close(); benchmark.close()
    diag = {"queries_per_cell": a.queries_per_cell, "selected_queries": [{"query_id": x["query_id"], "pair_id": x["pair_id"], "task": x["task"], "query_t": x["query_t"], "failure_cell": "effective" if bool(x.get("is_effective_intervention", False)) else "no_effect"} for x in queries], "median_threshold": a.median_threshold, "p95_threshold": a.p95_threshold, "summaries": summaries, "details": details}
    a.output_diagnostics.parent.mkdir(parents=True, exist_ok=True); a.output_diagnostics.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    direct = [key for key, value in summaries.items() if key.startswith("direct_state_reset+") and value["passed"]]
    prefix = [key for key, value in summaries.items() if key.startswith("prefix_replay+") and value["passed"]]
    choices = direct or prefix
    if not choices:
        result = {"status": "failed", "failure_reason": "neither direct reset nor prefix replay reproduced benchmark transitions", "median_threshold": a.median_threshold, "p95_threshold": a.p95_threshold, "diagnostics": str(a.output_diagnostics)}
    else:
        winner = min(choices, key=lambda key: (summaries[key]["next_state_median_l2"], key))
        mode, layout = winner.split("+", 1); frozen = TransitionAlignment.from_layout(layout, mode); s = summaries[winner]
        result = {"status": "passed", "state_layout": frozen.state_layout, "oracle_replay_mode": frozen.oracle_replay_mode, "state_before_action_offset": frozen.state_before_action_offset, "expected_next_state_offset": frozen.expected_next_state_offset, "image_before_action_offset": frozen.image_before_action_offset, "minimum_query_t": frozen.minimum_query_t, "pre_state_median_l2": s["pre_state_median_l2"], "pre_state_p95_l2": s["pre_state_p95_l2"], "next_state_median_l2": s["next_state_median_l2"], "next_state_p95_l2": s["next_state_p95_l2"], "median_threshold": a.median_threshold, "p95_threshold": a.p95_threshold, "diagnostics": str(a.output_diagnostics)}
    a.output_alignment.parent.mkdir(parents=True, exist_ok=True); a.output_alignment.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
