"""Frozen residual repair-policy adapter for HB1 branch execution."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from recovery_handback.adapters.base_policy import BasePolicyAdapter
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


def flatten_teacher_residual_observation(
    raw_obs,
    privileged_state,
    base_action,
    teacher_action,
    base_memory,
    last_action,
    stage_vector,
    remaining_steps,
    horizon_steps,
):
    parts = []
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if key in raw_obs:
            parts.append(np.asarray(raw_obs[key], dtype=np.float32).reshape(-1))
    for key in sorted(privileged_state):
        if not key.startswith("__"):
            parts.append(np.asarray(privileged_state[key], dtype=np.float32).reshape(-1))
    base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
    teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
    parts.extend((
        base_action,
        teacher_action,
        teacher_action - base_action,
        np.asarray(last_action, dtype=np.float32).reshape(-1),
        np.asarray(stage_vector, dtype=np.float32).reshape(-1),
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


class SACResidualRepairPolicy:
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


class DirectRobomimicRepairPolicy:
    action_mode = "direct"
    preserve_base_action = False

    def __init__(self, checkpoint: str | Path, *, device="cuda"):
        self.actor = BasePolicyAdapter(checkpoint, device=device)

    def start_episode(self, horizon_steps=400):
        self.actor.start_episode()

    def residual(self, raw_obs, privileged_state, base_action, base_memory,
                 remaining_steps, root_key, absolute_t):
        return self.actor.suggest_once(
            raw_obs,
            absolute_t=absolute_t,
            root_key=f"{root_key}:direct_repair",
        )

    def observe_executed_action(self, action):
        return None

    def identity(self):
        result = self.actor.actor_identity()
        result.update({
            "algorithm": "robomimic_bc_gmm_direct",
            "action_mode": "direct",
            "uses_privileged_input": True,
        })
        return result


class SACTeacherCenteredRepairPolicy:
    action_mode = "direct"
    preserve_base_action = False
    requires_stage_rewards = True

    def __init__(self, record: dict, *, device="cuda"):
        self.checkpoint = Path(record["checkpoint"]).resolve()
        self.normalizer_path = Path(record["normalizer"]).resolve()
        self.teacher_checkpoint = Path(record["teacher_checkpoint"]).resolve()
        self.teacher = BasePolicyAdapter(self.teacher_checkpoint, device=device)
        self.scale = np.asarray(record["residual_scale"], dtype=np.float32)
        self.action_low = np.asarray(record.get("action_low", [-1.0] * 7), dtype=np.float32)
        self.action_high = np.asarray(record.get("action_high", [1.0] * 7), dtype=np.float32)
        dummy = DummyVecEnv([lambda: _InferenceEnv(record["observation_shape"])])
        self.normalizer = VecNormalize.load(str(self.normalizer_path), dummy)
        self.normalizer.training = False
        self.normalizer.norm_reward = False
        self.model = SAC.load(str(self.checkpoint), device=device)
        self._last_action = np.zeros(7, dtype=np.float32)
        self._horizon = 400

    def start_episode(self, horizon_steps=400):
        self.teacher.start_episode()
        self._last_action.fill(0.0)
        self._horizon = int(horizon_steps)

    def residual(self, raw_obs, privileged_state, base_action, base_memory,
                 remaining_steps, root_key, absolute_t):
        teacher_action = self.teacher.suggest_once(
            raw_obs,
            absolute_t=absolute_t,
            root_key=f"{root_key}:teacher_residual_center",
        )
        stage_vector = privileged_state.get("__stage_rewards__", np.zeros(0, dtype=np.float32))
        observation = flatten_teacher_residual_observation(
            raw_obs,
            privileged_state,
            base_action,
            teacher_action,
            base_memory,
            self._last_action,
            stage_vector,
            remaining_steps,
            self._horizon,
        )
        normalized = self.normalizer.normalize_obs(observation[None])[0]
        residual, _ = self.model.predict(normalized, deterministic=True)
        residual = np.clip(np.asarray(residual, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return np.clip(
            teacher_action + self.scale * residual,
            self.action_low,
            self.action_high,
        ).astype(np.float32)

    def observe_executed_action(self, action):
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

    def identity(self):
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "normalizer": str(self.normalizer_path),
            "normalizer_sha256": sha256_file(self.normalizer_path),
            "teacher_checkpoint": str(self.teacher_checkpoint),
            "teacher_checkpoint_sha256": sha256_file(self.teacher_checkpoint),
            "algorithm": "sac_teacher_centered_residual",
            "action_mode": "direct",
            "uses_privileged_input": True,
        }


def build_repair_policy(record: dict, *, device="cuda"):
    algorithm = str(record["algorithm"])
    if algorithm == "robomimic_bc_gmm_direct":
        return DirectRobomimicRepairPolicy(record["checkpoint"], device=device)
    if algorithm == "sac_teacher_centered_residual":
        return SACTeacherCenteredRepairPolicy(record, device=device)
    if algorithm == "sac_bounded_residual_adapter":
        return SACResidualRepairPolicy(
            record["checkpoint"],
            record["normalizer"],
            record["observation_shape"],
            device=device,
            action_mode=record.get("action_mode", "residual"),
        )
    raise ValueError(f"unknown repair algorithm: {algorithm}")


# Preserve the original import surface for older HB1 utilities.
RepairPolicyAdapter = SACResidualRepairPolicy
