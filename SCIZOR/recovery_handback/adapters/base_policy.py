"""Closed-loop Robomimic action-policy adapter for HB1."""
from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import tensor_utils as TensorUtils

from recovery_handback.common import sha256_file, stable_seed


@contextmanager
def _policy_random_tape(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Explicitly reset every CUDA generator.  This keeps per-step GMM
        # sampling independent of the process' ambient CUDA RNG state.
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _clone_memory(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_memory(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_memory(item) for item in value)
    return value


def _contiguous_observation(value: Any):
    if isinstance(value, dict):
        return {key: _contiguous_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_contiguous_observation(item) for item in value)
    return value


def _sample_gmm(dist, seed: int) -> np.ndarray:
    """Sample a Robomimic GMM from an explicit, device-independent tape."""
    mixture = getattr(dist, "mixture_distribution", None)
    component = getattr(dist, "component_distribution", None)
    base = getattr(component, "base_dist", None)
    if mixture is None or base is None or not hasattr(base, "loc"):
        raise TypeError(f"unsupported GMM distribution: {type(dist).__name__}")
    probs = mixture.probs.detach().cpu().numpy()
    means = base.loc.detach().cpu().numpy()
    scales = base.scale.detach().cpu().numpy()
    if probs.shape[0] != 1 or means.shape[0] != 1 or scales.shape[0] != 1:
        raise ValueError("HB1 action sampling expects a single observation batch")
    probs = np.asarray(probs[0], dtype=np.float64)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    mode = int(rng.choice(len(probs), p=probs))
    return rng.normal(means[0, mode], scales[0, mode]).astype(np.float32)


class BasePolicyAdapter:
    """One action suggestion per environment time, with isolated policy history."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda"):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.policy, self.checkpoint_data = FileUtils.policy_from_checkpoint(
            ckpt_path=str(self.checkpoint), device=self.device, verbose=False
        )
        shape_meta = self.checkpoint_data.get("shape_metadata", {})
        self._obs_keys = tuple(shape_meta.get("all_obs_keys", ()))
        self._calls = 0
        self._total_calls = 0

    def start_episode(self):
        self.policy.start_episode()
        self._calls = 0

    def suggest_once(self, raw_obs: dict, absolute_t: int, root_key: str):
        seed = stable_seed(root_key, int(absolute_t), "base")
        observation = self._prepare_observation(raw_obs)
        with _policy_random_tape(seed), torch.no_grad():
            action = self._suggest_with_frozen_gmm_tape(observation, seed)
            if action is None:
                action = self.policy(observation)
        action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        self._calls += 1
        self._total_calls += 1
        return action

    def _suggest_with_frozen_gmm_tape(self, observation: dict, seed: int):
        """Preserve Robomimic policy state while making GMM sampling explicit."""
        algo = getattr(self.policy, "policy", None)
        algo_name = type(algo).__name__
        if algo_name not in {"BC_GMM", "BC_RNN_GMM"}:
            return None
        prepared = self.policy._prepare_observation(observation, batched=False)
        network = algo.nets["policy"]
        if algo_name == "BC_GMM":
            dist = network.forward_train(obs_dict=prepared, goal_dict=None)
            return _sample_gmm(dist, seed)

        if algo._rnn_hidden_state is None or algo._rnn_counter % algo._rnn_horizon == 0:
            batch_size = list(prepared.values())[0].shape[0]
            algo._rnn_hidden_state = network.get_rnn_init_state(
                batch_size=batch_size, device=algo.device
            )
            if algo._rnn_is_open_loop:
                algo._open_loop_obs = TensorUtils.clone(TensorUtils.detach(prepared))
        obs_to_use = algo._open_loop_obs if algo._rnn_is_open_loop else prepared
        algo._rnn_counter += 1
        dist, algo._rnn_hidden_state = network.forward_train_step(
            obs_to_use, goal_dict=None, rnn_state=algo._rnn_hidden_state
        )
        return _sample_gmm(dist, seed)

    def _prepare_observation(self, raw_obs: dict) -> dict:
        """Convert the runtime HWC/uint8 observation to the policy input layout."""
        keys = self._obs_keys or tuple(raw_obs.keys())
        observation = {}
        for key in keys:
            if key not in raw_obs:
                raise KeyError(f"policy observation {key!r} is missing from runtime observation")
            value = np.ascontiguousarray(raw_obs[key])
            if key not in ObsUtils.OBS_KEYS_TO_MODALITIES:
                raise KeyError(f"policy observation {key!r} has no registered modality")
            # RolloutPolicy only batches and casts; it does not perform the
            # dataset HWC -> CHW / uint8 -> float conversion itself.
            observation[key] = ObsUtils.process_obs(obs=value, obs_key=key)
        return _contiguous_observation(observation)

    def memory_snapshot(self):
        algo = getattr(self.policy, "policy", self.policy)
        snapshot = {}
        for name in ("_rnn_hidden_state", "_rnn_counter", "_open_loop_obs"):
            if hasattr(algo, name):
                snapshot[name] = _clone_memory(getattr(algo, name))
        return snapshot

    def actor_identity(self) -> dict:
        shape_meta = self.checkpoint_data.get("shape_metadata", {})
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "algo_name": self.checkpoint_data.get("algo_name"),
            "action_dim": int(shape_meta.get("ac_dim", -1)),
            "observation_shapes": shape_meta.get("all_shapes", {}),
            "calls": self._calls,
            "total_calls": self._total_calls,
            "device": str(self.device),
            "gmm_random_tape": "numpy_categorical_gaussian_v1",
        }
