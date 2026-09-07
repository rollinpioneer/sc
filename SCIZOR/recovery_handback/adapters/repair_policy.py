"""Frozen residual repair-policy adapter for HB1 branch execution."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from recovery_handback.common import sha256_file


class _InferenceEnv(gym.Env):
    def __init__(self, observation_shape):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=tuple(observation_shape), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}


def flatten_repair_observation(raw_obs, privileged_state, base_action, base_memory,
                               last_action, remaining_steps, horizon_steps):
    parts = []
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if key in raw_obs:
            parts.append(np.asarray(raw_obs[key], dtype=np.float32).reshape(-1))
    for key in sorted(privileged_state):
        parts.append(np.asarray(privileged_state[key], dtype=np.float32).reshape(-1))
    parts.extend((
        np.asarray(base_action, dtype=np.float32).reshape(-1),
        np.asarray(last_action, dtype=np.float32).reshape(-1),
        np.asarray([remaining_steps / max(1, horizon_steps)], dtype=np.float32),
    ))
    hidden = base_memory.get("_rnn_hidden_state") if base_memory else None
    if hidden is not None:
        tensors = hidden if isinstance(hidden, (tuple, list)) else (hidden,)
        for tensor in tensors:
            if torch.is_tensor(tensor):
                parts.append(tensor.detach().cpu().numpy().astype(np.float32).reshape(-1))
            else:
                parts.append(np.asarray(tensor, dtype=np.float32).reshape(-1))
    counter = float((base_memory or {}).get("_rnn_counter", 0))
    parts.append(np.asarray([counter / 10.0], dtype=np.float32))
    return np.concatenate(parts, dtype=np.float32)


class RepairPolicyAdapter:
    def __init__(self, checkpoint: str | Path, normalizer: str | Path, observation_shape,
                 *, device="cuda", action_mode="residual"):
        self.checkpoint = Path(checkpoint).resolve()
        self.normalizer_path = Path(normalizer).resolve()
        self.action_mode = action_mode
        dummy = DummyVecEnv([lambda: _InferenceEnv(observation_shape)])
        self.normalizer = VecNormalize.load(str(self.normalizer_path), dummy)
        self.normalizer.training = False
        self.normalizer.norm_reward = False
        self.model = SAC.load(str(self.checkpoint), device=device)
        self._last_action = np.zeros(7, dtype=np.float32)
        self._horizon = 400

    def start_episode(self, horizon_steps=400):
        self._last_action = np.zeros(7, dtype=np.float32)
        self._horizon = int(horizon_steps)

    def residual(self, raw_obs, privileged_state, base_action, base_memory,
                 remaining_steps, root_key, absolute_t):
        observation = flatten_repair_observation(
            raw_obs, privileged_state, base_action, base_memory,
            self._last_action, remaining_steps, self._horizon,
        )
        normalized = self.normalizer.normalize_obs(observation[None])[0]
        action, _ = self.model.predict(normalized, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)

    def observe_executed_action(self, action):
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

    def identity(self):
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "normalizer": str(self.normalizer_path),
            "normalizer_sha256": sha256_file(self.normalizer_path),
            "algorithm": "sac_bounded_residual_adapter",
            "action_mode": self.action_mode,
            "uses_privileged_input": True,
        }
