"""Run frozen 100-frame simulator counterfactuals for proposal query groups."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from stage3a.rescue_v02.common import _staged, _state, env_for_dataset, text
from stage3a.rescue_v02r.outcome_score_long import load_json, score_outcomes


def reset_prefix(env, initial, actions, model):
    env.reset()
    env.reset_to({"states": np.asarray(initial).copy(), "model": model})
    for action in actions:
        env.step(action)
    return _state(env)


def rollout(env, actions, horizon):
    states, rewards, staged, success = [], [], [], []
    for action in np.asarray(actions)[:horizon]:
        _, reward, _, _ = env.step(action)
        states.append(_state(env)); rewards.append(float(reward)); staged.append(_staged(env)); success.append(bool(env.is_success().get("task", False)))
    return {"states_post": np.asarray(states), "rewards": np.asarray(rewards, dtype=np.float32), "staged_rewards": np.asarray(staged, dtype=np.float32), "success": np.asarray(success, dtype=np.bool_)}


def exact(run, group, start):
    maxima = []
    for key in ("states_post", "rewards", "staged_rewards", "success"):
        left, right = np.asarray(run[key]), np.asarray(group[key])[start:start + len(run[key])]
        maxima.append(float(np.max(np.abs(left.astype(float) - right.astype(float)))) if left.shape == right.shape and left.size else (0.0 if left.shape == right.shape else float("inf")))
    return bool(all(value == 0.0 for value in maxima)), float(max(maxima))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--normalizer", type=Path, required=True); p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--part-index", type=int, default=0); p.add_argument("--num-parts", type=int, default=1)
    a = p.parse_args()
    grouped, order = defaultdict(list), []
    for line in a.plans.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        plan = json.loads(line); query = str(plan["query_id"])
        if query not in grouped:
            order.append(query)
        grouped[query].append(plan)
    selected = [query for index, query in enumerate(order) if index % a.num_parts == a.part_index]
    normalizer, spec = load_json(a.normalizer)["normalizer"], load_json(a.spec)
    envs, sources, models, result = {}, {}, {}, []
    with h5py.File(a.benchmark, "r") as h5:
        for query_id in selected:
            plans = sorted(grouped[query_id], key=lambda row: int(row["replacement_rank"]))
            if len(plans) != 4 or len({row["replacement_id"] for row in plans}) != 4:
                raise ValueError(f"invalid plan group {query_id}")
            group = h5[f"data/{plans[0]['perturbed_demo_id']}"]
            task, t = text(group.attrs["task"]), int(plans[0]["query_t"])
            if any(str(plan["task"]) != task or int(plan["query_t"]) != t for plan in plans):
                raise ValueError(f"inconsistent plans for {query_id}")
            actions = np.asarray(group["actions"])
            if task not in envs:
                source = text(group.attrs["source_dataset"])
                envs[task] = (env_for_dataset(source)[0], env_for_dataset(source)[0])
                sources[task] = h5py.File(source, "r")
                models[task] = envs[task][0].env.model.get_xml()
            initial = np.asarray(sources[task][f"data/{text(group.attrs['base_demo_id'])}"]["states"][0]).copy()
            pre_reference = reset_prefix(envs[task][0], initial, actions[:t], models[task])
            horizon = min(int(spec["max_horizon"]), len(actions) - t)
            reference = rollout(envs[task][0], actions[t:], horizon)
            reference_exact, reference_max = exact(reference, group, t)
            for plan in plans:
                pre_replacement = reset_prefix(envs[task][1], initial, actions[:t], models[task])
                replaced = actions[t:].copy(); replaced[0] = np.asarray(plan["replacement_action"], dtype=actions.dtype)
                replacement = rollout(envs[task][1], replaced, horizon)
                row = dict(plan)
                row.update({"branch_pre_state_equal": bool(np.array_equal(pre_reference, pre_replacement)), "branch_pre_state_max_abs": float(np.max(np.abs(pre_reference - pre_replacement))),
                            "reference_exact": reference_exact, "reference_max_abs": reference_max})
                row.update(score_outcomes(task=task, reference_rewards=reference["rewards"], replacement_rewards=replacement["rewards"], reference_staged=reference["staged_rewards"], replacement_staged=replacement["staged_rewards"], reference_success=reference["success"], replacement_success=replacement["success"], normalizer=normalizer, spec=spec))
                result.append(row)
    for pair in envs.values():
        pair[0].close(); pair[1].close()
    for source in sources.values():
        source.close()
    if len(result) != len(selected) * 4:
        raise RuntimeError("candidate oracle lost replacement rows")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(json.dumps(row) for row in result) + "\n", encoding="utf-8")
    summary = {"query_count": len(selected), "row_count": len(result), **{f"{key}_rate": float(np.mean([row[key] for row in result])) if result else 0.0 for key in ("branch_pre_state_equal", "reference_exact", "finite_target")}}
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
