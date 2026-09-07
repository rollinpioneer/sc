"""Teacher-centered privileged residual environment for HB1-R."""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.adapters.repair_policy import flatten_teacher_residual_observation
from recovery_handback.common import read_table
from recovery_handback.train.residual_env import load_payload
from recovery_handback.train.stage_potential import stage_potential


def _decode_list(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


class TeacherResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config,
        assets,
        task,
        base_checkpoint,
        teacher_checkpoint,
        training_roots,
        curriculum_index,
        curriculum_phase,
        *,
        device="cuda",
    ):
        super().__init__()
        if curriculum_phase not in {"easy", "medium", "all"}:
            raise ValueError(f"unknown curriculum phase: {curriculum_phase}")
        self.config = config
        self.task = task
        self.horizon = int(config["horizon_steps"])
        self.settings = config["capability_repair"]["square_teacher_residual"]
        self.scale = np.asarray(self.settings["residual_scale"], dtype=np.float32)
        source = assets["tasks"][task]["source_hdf5"]
        self.env = EnvAdapter(source, **load_observation_spec(source))
        self.base = BasePolicyAdapter(base_checkpoint, device=device)
        self.teacher = BasePolicyAdapter(teacher_checkpoint, device=device)
        roots = read_table(Path(training_roots))
        self.roots = {row["root_id"]: row for row in roots if not row.get("exception_reason")}
        states = read_table(Path(curriculum_index))
        if curriculum_phase != "all":
            states = [row for row in states if bool(row.get(curriculum_phase))]
        self.states = states
        if not self.states:
            raise RuntimeError(f"no curriculum states for phase {curriculum_phase}")
        low, high = self.env.action_bounds()
        self._action_low = low.astype(np.float32)
        self._action_high = high.astype(np.float32)
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
        self._teacher_action = None
        sample, _ = self.reset(seed=int(self.settings["seed"]))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=sample.shape, dtype=np.float32
        )

    def _feature(self):
        return flatten_teacher_residual_observation(
            self._obs,
            self.env.privileged_features(),
            self._base_action,
            self._teacher_action,
            self.base.memory_snapshot(),
            self._last_action,
            self.env.staged_rewards(),
            self.horizon - self._t,
            self.horizon,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_index += 1
        seed_value = int(seed) if seed is not None else int(self.settings["seed"]) + self._episode_index
        rng = np.random.default_rng(seed_value)
        state = self.states[int(rng.integers(0, len(self.states)))]
        root = self.roots[state["root_id"]]
        prefix = int(state["prefix_t"])
        with np.load(root["rollout_path"], allow_pickle=False) as handle:
            actions = np.asarray(handle["actions"], dtype=np.float32)
            successes = np.asarray(handle["success"], dtype=bool)
        obs = self.env.reset_canonical(load_payload(root), int(root["seed"]))
        self.base.start_episode()
        self.teacher.start_episode()
        self._root_key = root["root_id"]
        self._last_action.fill(0.0)
        for t in range(prefix):
            self.base.suggest_once(obs, t, self._root_key)
            action = actions[t]
            obs, _reward, _success, _info = self.env.step(action)
            self._last_action = action.copy()
            self.prefix_env_steps += 1
        self._t = prefix
        history_length = int(self.config["success_consecutive_steps"]) - 1
        self._success_history = [
            bool(value) for value in successes[:prefix][-history_length:]
        ]
        self._obs = obs
        self._base_action = self.base.suggest_once(obs, self._t, self._root_key)
        self._teacher_action = self.teacher.suggest_once(
            obs, self._t, f"{self._root_key}:teacher"
        )
        return self._feature(), {
            "root_id": root["root_id"],
            "prefix_steps": prefix,
            "curriculum_stage_vector": _decode_list(state.get("stage_vector", [])),
        }

    def step(self, residual):
        residual = np.clip(np.asarray(residual, dtype=np.float32), -1.0, 1.0)
        action = np.clip(
            self._teacher_action + self.scale * residual,
            self._action_low,
            self._action_high,
        ).astype(np.float32)
        phi_before = stage_potential(self.env.staged_rewards())
        obs_next, _shaped_reward, raw_success, _info = self.env.step(action)
        self.transition_steps += 1
        self._t += 1
        self._success_history.append(bool(raw_success))
        required = int(self.config["success_consecutive_steps"])
        if len(self._success_history) > required:
            self._success_history.pop(0)
        stable_success = len(self._success_history) == required and all(self._success_history)
        deadline = self._t >= self.horizon
        phi_after = 0.0 if stable_success or deadline else stage_potential(self.env.staged_rewards())
        gamma = float(self.settings["gamma"])
        reward = float(self.settings["success_bonus"]) * float(stable_success)
        reward += float(self.settings["stage_delta_scale"]) * (gamma * phi_after - phi_before)
        reward -= float(self.settings["residual_penalty"]) * float(np.mean(np.square(residual)))
        reward -= float(self.settings["step_penalty"])
        self._obs = obs_next
        self._last_action = action.copy()
        terminated = bool(stable_success or deadline)
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self._base_action = self.base.suggest_once(obs_next, self._t, self._root_key)
            self._teacher_action = self.teacher.suggest_once(
                obs_next, self._t, f"{self._root_key}:teacher"
            )
            observation = self._feature()
        return observation, reward, terminated, False, {
            "raw_success": bool(raw_success),
            "stable_success": bool(stable_success),
            "end_reason": "success" if stable_success else ("deadline" if deadline else None),
            "absolute_t": self._t,
            "prefix_env_steps_total": self.prefix_env_steps,
            "transition_steps_total": self.transition_steps,
        }

    def close(self):
        self.env.close()
        super().close()
