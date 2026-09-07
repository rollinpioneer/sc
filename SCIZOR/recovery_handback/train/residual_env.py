"""Gymnasium environment for the HB1 privileged residual SAC baseline."""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.adapters.repair_policy import flatten_repair_observation
from recovery_handback.common import stable_seed


def _read_jsonl(path: Path):
    source = path / "roots.jsonl" if path.is_dir() else path
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_payload(row):
    state_path = Path(row["payload_path"])
    stem = state_path.with_suffix("")
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(state_path, allow_pickle=False) as state_file:
        states = np.asarray(state_file["states"], dtype=np.float64)
    return {"states": states, "model": stem.with_suffix(".xml").read_text(encoding="utf-8"),
            "ep_meta": meta.get("ep_meta"), "seed": int(row["seed"])}


class ResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config, assets, task, base_checkpoint, training_roots, *, device="cuda"):
        super().__init__()
        self.config = config
        self.task = task
        self.horizon = int(config["horizon_steps"])
        self.scale = np.asarray(config["repair_training"]["residual_scale"], dtype=np.float32)
        source = assets["tasks"][task]["source_hdf5"]
        self.env = EnvAdapter(source, **load_observation_spec(source))
        self.base = BasePolicyAdapter(base_checkpoint, device=device)
        self.roots = [row for row in _read_jsonl(Path(training_roots)) if not row.get("exception_reason")]
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.prefix_env_steps = 0
        self.transition_steps = 0
        self._episode_index = 0
        self._last_action = np.zeros(7, dtype=np.float32)
        self._t = 0
        self._success_history = []
        self._root_key = ""
        self._obs = None
        self._base_action = None
        sample, _ = self._initial_reset(seed=int(config["repair_training"]["seed"]))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=sample.shape, dtype=np.float32)

    def _initial_reset(self, seed):
        obs, _ = self.env.new_episode(int(seed))
        self.base.start_episode()
        self._t = 0
        self._root_key = f"{self.task}:repair_sac:{seed}:{self._episode_index}"
        self._last_action.fill(0.0)
        self._success_history = []
        self._obs = obs
        self._base_action = self.base.suggest_once(obs, self._t, self._root_key)
        return self._feature(), {"reset_mode": "initial", "prefix_steps": 0}

    def _prefix_reset(self, rng):
        row = self.roots[int(rng.integers(0, len(self.roots)))]
        rollout = np.load(row["rollout_path"], allow_pickle=False)
        max_prefix = min(len(rollout["actions"]) - 1, self.horizon - 1)
        prefix = int(rng.integers(0, max_prefix + 1))
        obs = self.env.reset_canonical(load_payload(row), int(row["seed"]))
        self.base.start_episode()
        self._root_key = row["root_id"]
        self._last_action.fill(0.0)
        for t in range(prefix):
            self.base.suggest_once(obs, t, self._root_key)
            action = np.asarray(rollout["actions"][t], dtype=np.float32)
            obs, _reward, _success, _info = self.env.step(action)
            self._last_action = action.copy()
            self.prefix_env_steps += 1
        self._t = prefix
        history_length = int(self.config["success_consecutive_steps"]) - 1
        self._success_history = [
            bool(value) for value in np.asarray(rollout["success"][:prefix], dtype=bool)[-history_length:]
        ]
        self._obs = obs
        self._base_action = self.base.suggest_once(obs, self._t, self._root_key)
        return self._feature(), {"reset_mode": "prefix", "prefix_steps": prefix, "root_id": row["root_id"]}

    def _feature(self):
        return flatten_repair_observation(
            self._obs, self.env.privileged_features(), self._base_action,
            self.base.memory_snapshot(), self._last_action,
            self.horizon - self._t, self.horizon,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_index += 1
        if seed is not None:
            seed_value = int(seed)
        else:
            role = self.config["roles"]["repair_train"]
            offset = int(self.config.get("square_seed_offset", 0)) if self.task == "square" else 0
            seed_value = int(role["seed_start"]) + int(role["roots_per_task"]) + offset + self._episode_index
        rng = np.random.default_rng(seed_value)
        use_initial = not self.roots or rng.random() < float(self.config["repair_training"]["initial_reset_fraction"])
        return self._initial_reset(seed_value) if use_initial else self._prefix_reset(rng)

    def step(self, residual):
        residual = np.clip(np.asarray(residual, dtype=np.float32), -1.0, 1.0)
        low, high = self.env.action_bounds()
        action = np.clip(self._base_action + self.scale * residual, low, high).astype(np.float32)
        phi_before = float(np.clip(self.env.env.get_reward(), 0.0, 1.0))
        obs_next, _shaped_reward, raw_success, _info = self.env.step(action)
        self.transition_steps += 1
        self._t += 1
        self._success_history.append(bool(raw_success))
        if len(self._success_history) > int(self.config["success_consecutive_steps"]):
            self._success_history.pop(0)
        stable_success = len(self._success_history) == int(self.config["success_consecutive_steps"]) and all(self._success_history)
        deadline = self._t >= self.horizon
        phi_after = 0.0 if (stable_success or deadline) else float(np.clip(self.env.env.get_reward(), 0.0, 1.0))
        gamma = float(self.config["repair_training"]["gamma"])
        reward = (1.0 if stable_success else 0.0) + 0.1 * (gamma * phi_after - phi_before)
        reward -= 0.001 * float(np.mean(np.square(residual)))
        self._obs = obs_next
        self._last_action = action.copy()
        terminated = bool(stable_success or deadline)
        truncated = False
        if not terminated:
            self._base_action = self.base.suggest_once(obs_next, self._t, self._root_key)
            observation = self._feature()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, float(reward), terminated, truncated, {
            "raw_success": bool(raw_success), "stable_success": bool(stable_success),
            "end_reason": "success" if stable_success else ("deadline" if deadline else None),
            "absolute_t": self._t, "prefix_env_steps_total": self.prefix_env_steps,
            "transition_steps_total": self.transition_steps,
        }

    def close(self):
        self.env.close()
        super().close()
