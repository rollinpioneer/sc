"""Twin-prefix feasible replacement oracle for replay-locked v0.2."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from .common import _state, _staged, env_for_dataset, text


def rows_from(path):
    if path.suffix != ".jsonl":
        raise RuntimeError("the simulator runtime has no parquet engine; provide the companion .jsonl plan")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def reset_prefix(env, initial, prefix, model_xml):
    env.reset(); env.reset_to({"states": np.asarray(initial).copy(), "model": model_xml})
    for action in prefix: env.step(action)
    return _state(env)


def continuation(env, actions, horizon):
    states, rewards, staged, success = [], [], [], []
    for action in actions[:horizon]:
        _, reward, _, _ = env.step(action)
        states.append(_state(env)); rewards.append(float(reward)); staged.append(_staged(env)); success.append(bool(env.is_success().get("task", False)))
    return {"states_post": np.asarray(states), "rewards": np.asarray(rewards, np.float32), "staged_rewards": np.asarray(staged), "success": np.asarray(success, np.bool_)}


def mean(values, h):
    return float(np.mean(values[: min(h, len(values))])) if len(values) else np.nan


def exact(reference, stored, t):
    n = len(reference["rewards"])
    fields = (
        (reference["states_post"], np.asarray(stored["states_post"])[t:t + n]),
        (reference["rewards"], np.asarray(stored["rewards"])[t:t + n]),
        (reference["staged_rewards"], np.asarray(stored["staged_rewards"])[t:t + n]),
        (reference["success"], np.asarray(stored["success"])[t:t + n]),
    )
    maxima = [float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape and a.size else np.inf for a, b in fields]
    return bool(all(x == 0.0 for x in maxima)), float(max(maxima))


def one_pair(env_ref, env_repl, source, group, rows, model_xml, horizons):
    t = int(rows[0]["query_t"]); actions = np.asarray(group["actions"]); initial = np.asarray(source[f"data/{text(group.attrs['base_demo_id'])}"]["states"][0]).copy()
    pre_ref = reset_prefix(env_ref, initial, actions[:t], model_xml)
    horizon = min(max(horizons), len(actions) - t)
    reference = continuation(env_ref, actions[t:], horizon)
    ref_exact, ref_max = exact(reference, group, t)
    output = []
    for row in rows:
        pre_repl = reset_prefix(env_repl, initial, actions[:t], model_xml)
        replacement_actions = actions[t:].copy(); replacement_actions[0] = np.asarray(row["replacement_action"], np.float32)
        replacement = continuation(env_repl, replacement_actions, horizon)
        out = {k: row.get(k) for k in ("replacement_id", "query_id", "pair_id", "task", "base_demo_id", "split", "query_t", "query_source", "replacement_rank", "replacement_source", "library_id", "library_base_demo_id", "library_clean_demo_id", "library_t", "state_distance", "action_delta_l2", "state_in_domain", "action_in_domain", "failure_type", "is_effective_intervention")}
        out.update({"branch_pre_state_equal": bool(np.array_equal(pre_ref, pre_repl)), "branch_pre_state_max_abs": float(np.max(np.abs(pre_ref - pre_repl))), "reference_exact_all_horizons": ref_exact, "reference_max_abs": ref_max, "finite_target": bool(np.isfinite(reference["rewards"]).all() and np.isfinite(replacement["rewards"]).all() and np.isfinite(reference["staged_rewards"]).all() and np.isfinite(replacement["staged_rewards"]).all()), "actual_horizon": int(horizon)})
        for h in horizons:
            stage_ref = np.max(reference["staged_rewards"], axis=1) if reference["staged_rewards"].ndim == 2 else reference["staged_rewards"]
            stage_repl = np.max(replacement["staged_rewards"], axis=1) if replacement["staged_rewards"].ndim == 2 else replacement["staged_rewards"]
            out[f"dense_mean_delta_h{h}"] = mean(replacement["rewards"], h) - mean(reference["rewards"], h)
            out[f"stage_mean_delta_h{h}"] = mean(stage_repl, h) - mean(stage_ref, h)
            out[f"success_delta_h{h}"] = mean(replacement["success"].astype(float), h) - mean(reference["success"].astype(float), h)
        output.append(out)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--plans", type=Path, required=True); p.add_argument("--horizons", default="10,20,40")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--part-index", type=int, default=0); p.add_argument("--num-parts", type=int, default=1); a = p.parse_args()
    horizons = [int(x) for x in a.horizons.split(",")]
    plans = rows_from(a.plans); grouped = defaultdict(list); order = []
    for row in plans:
        key = str(row["pair_id"])
        if key not in grouped: order.append(key)
        grouped[key].append(row)
    selected = [key for i, key in enumerate(order) if i % a.num_parts == a.part_index]
    envs, sources, models, output = {}, {}, {}, []
    with h5py.File(a.benchmark, "r") as h5:
        for pair_id in selected:
            rows = grouped[pair_id]; group = h5[f"data/{rows[0]['perturbed_demo_id']}"]; task = text(group.attrs["task"])
            if task not in envs:
                source_path = text(group.attrs["source_dataset"]); envs[task] = (env_for_dataset(source_path)[0], env_for_dataset(source_path)[0]); sources[task] = h5py.File(source_path, "r"); models[task] = envs[task][0].env.model.get_xml()
            output.extend(one_pair(envs[task][0], envs[task][1], sources[task], group, rows, models[task], horizons))
    for a_env, b_env in envs.values(): a_env.close(); b_env.close()
    for source in sources.values(): source.close()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in output) + "\n", encoding="utf-8")
    metrics = {"rows": len(output), "pair_count": len(selected), "branch_pre_state_equal_rate": float(sum(r["branch_pre_state_equal"] for r in output) / len(output)) if output else 0.0, "reference_exact_all_horizons_rate": float(sum(r["reference_exact_all_horizons"] for r in output) / len(output)) if output else 0.0, "finite_target_rate": float(sum(r["finite_target"] for r in output) / len(output)) if output else 0.0}
    a.summary.parent.mkdir(parents=True, exist_ok=True); a.summary.write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
