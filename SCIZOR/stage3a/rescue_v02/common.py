from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def env_for_dataset(dataset_path: str):
    from stage1.benchmark.simulator_replay import create_shaped_env

    return create_shaped_env(str(dataset_path))


def _state(env):
    # EnvRobosuite.get_state() also serializes the full XML on every call.
    # For per-step replay checks the simulator flat state is the canonical
    # numeric state and avoids that expensive serialization.
    return np.asarray(env.env.sim.get_state().flatten()).copy()


def _staged(env):
    fn = getattr(env.env, "staged_rewards", None)
    if fn is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(fn(), dtype=np.float32).reshape(-1).copy()


def replay(env, initial_state, actions, *, render_images=True, height=84, width=84, model_xml=None):
    """Replay actions while explicitly recording pre- and post-action state."""
    env.reset()
    payload = {"states": np.asarray(initial_state).copy()}
    if model_xml is not None:
        payload["model"] = model_xml
    env.reset_to(payload)
    actions = np.asarray(actions).copy()
    pre_states, post_states, images_pre, images_post = [], [], [], []
    rewards, staged, success = [], [], []
    for action in actions:
        pre_states.append(_state(env))
        if render_images:
            images_pre.append(np.asarray(env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")).copy())
        _, reward, _, _ = env.step(action)
        post_states.append(_state(env))
        if render_images:
            images_post.append(np.asarray(env.render(mode="rgb_array", height=height, width=width, camera_name="agentview")).copy())
        rewards.append(float(reward))
        staged.append(_staged(env))
        success.append(bool(env.is_success().get("task", False)))
    out = {
        "actions": actions,
        "states_pre": np.asarray(pre_states),
        "states_post": np.asarray(post_states),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "staged_rewards": np.asarray(staged, dtype=np.float32),
        "success": np.asarray(success, dtype=np.bool_),
    }
    if render_images:
        out["images_pre"] = np.asarray(images_pre, dtype=np.uint8)
        out["images_post"] = np.asarray(images_post, dtype=np.uint8)
    return out


def compare_rollouts(left, right):
    fields = ["states_pre", "states_post", "rewards", "staged_rewards", "success"]
    result = {}
    for field in fields:
        a, b = np.asarray(left[field]), np.asarray(right[field])
        result[field] = {
            "shape_equal": bool(a.shape == b.shape),
            "max_abs": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape and a.size else None,
            "array_equal": bool(np.array_equal(a, b)),
        }
    result["pass"] = all(x["array_equal"] for x in result.values())
    return result


def env_metadata(env, runtime_fingerprint_sha256: str | None = None):
    model_xml = env.env.model.get_xml()
    meta = getattr(env, "env_meta", {}) or {}
    kwargs = meta.get("env_kwargs", {}) if isinstance(meta, dict) else {}
    return {
        "runtime_resolved_model_sha256": sha256_bytes(model_xml.encode("utf-8")),
        "control_freq": int(kwargs.get("control_freq", 20)),
        "simulation_timestep": float(getattr(env.env.sim, "model", env.env.sim).opt.timestep),
        "num_internal_sim_steps_per_action": int(getattr(env.env, "control_freq", kwargs.get("control_freq", 20))),
        "controller_config_json": json.dumps(kwargs.get("controller_configs", {}), sort_keys=True, default=str),
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256 or "",
    }


def perturb_action(action, kind, magnitude, rng, low, high):
    from stage1.benchmark.perturbations import perturb_action as base

    return base(action, kind, magnitude, rng, low, high)
