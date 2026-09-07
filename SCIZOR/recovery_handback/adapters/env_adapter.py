"""Runtime environment adapter used by the HB1 closed-loop experiment."""
from __future__ import annotations

import json
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robomimic.envs.env_robosuite import EnvRobosuite
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset


@contextmanager
def _seeded(seed: int):
    """Seed reset-time RNGs without leaking reset randomness to the caller."""
    state_py = random.getstate()
    state_np = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(state_py)
        np.random.set_state(state_np)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class EnvAdapter:
    """Small, explicit wrapper around the locked Robomimic Robosuite runtime."""

    def __init__(self, source_hdf5: str | Path, *, camera_keys=None, low_dim_keys=None):
        self.source_hdf5 = Path(source_hdf5).expanduser().resolve()
        self.env_meta = get_env_metadata_from_dataset(str(self.source_hdf5))
        kwargs = deepcopy(self.env_meta["env_kwargs"])
        kwargs["reward_shaping"] = True
        self.control_freq = int(kwargs.get("control_freq", 20))
        self.env_name = str(self.env_meta["env_name"])
        self.camera_keys = list(camera_keys or ["agentview_image", "robot0_eye_in_hand_image"])
        self.low_dim_keys = list(low_dim_keys or [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
        ])

        # EnvRobosuite only knows which values are images after this mapping exists.
        ObsUtils.initialize_obs_utils_with_obs_specs({
            "obs": {
                "low_dim": self.low_dim_keys,
                "rgb": self.camera_keys,
                "depth": [],
                "scan": [],
            }
        })
        self.env = EnvRobosuite(
            env_name=self.env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            postprocess_visual_obs=False,
            **kwargs,
        )
        self._last_obs = None
        self._episode_payload = None

    def new_episode(self, seed: int):
        with _seeded(seed):
            raw_obs = self.env.reset()
        payload = self._canonical_payload(seed)
        self._episode_payload = payload
        self._last_obs = raw_obs
        return raw_obs, deepcopy(payload)

    def reset_canonical(self, payload: dict, seed: int):
        with _seeded(seed):
            raw_obs = self.env.reset()
        live = self.env.get_state()
        expected_state = np.asarray(payload["states"], dtype=np.float64)
        live_state = np.asarray(live["states"], dtype=np.float64)
        state_equal = np.array_equal(live_state, expected_state)
        model_equal = str(live["model"]) == str(payload["model"])
        if not state_equal or not model_equal:
            state_error = (
                float(np.max(np.abs(live_state - expected_state)))
                if live_state.shape == expected_state.shape and live_state.size
                else float("inf")
            )
            raise RuntimeError(
                "seeded reset did not reproduce canonical payload "
                f"(state_max_abs={state_error}, model_equal={model_equal})"
            )
        self._episode_payload = deepcopy(payload)
        self._last_obs = raw_obs
        return raw_obs

    def step(self, action):
        obs_next, reward, _done, info = self.env.step(np.asarray(action, dtype=np.float32))
        success = bool(self.env.is_success().get("task", False))
        self._last_obs = obs_next
        return obs_next, float(reward), success, dict(info or {})

    def physical_state(self):
        return np.asarray(self.env.env.sim.get_state().flatten(), dtype=np.float64).copy()

    def privileged_features(self):
        # The current reset / step observation already contains object state.
        # Calling get_observation() again also re-renders cameras and can change
        # the next image seen by the recurrent base policy.
        obs = self._last_obs
        if obs is None:
            raise RuntimeError("privileged features requested before reset")
        return {
            key: np.asarray(value).copy()
            for key, value in obs.items()
            if key.startswith("object")
        }

    def action_bounds(self):
        low, high = self.env.env.action_spec
        return np.asarray(low, dtype=np.float64).copy(), np.asarray(high, dtype=np.float64).copy()

    def current_observation(self):
        return self._last_obs if self._last_obs is not None else self.env.get_observation()

    def _canonical_payload(self, seed: int) -> dict:
        state = self.env.get_state()
        ep_meta = None
        getter = getattr(self.env.env, "get_ep_meta", None)
        if callable(getter):
            ep_meta = json.dumps(_jsonable(getter()), sort_keys=True)
        return {
            "seed": int(seed),
            "states": np.asarray(state["states"], dtype=np.float64).copy(),
            "model": str(state["model"]),
            "ep_meta": ep_meta,
            "env_name": self.env_name,
            "control_freq": self.control_freq,
            "env_args": _jsonable(self.env_meta),
        }

    def close(self):
        self.env.close()


def load_observation_spec(source_hdf5: str | Path) -> dict:
    """Read only observation names from a source dataset."""
    with h5py.File(source_hdf5, "r") as handle:
        demo_id = sorted(handle["data"].keys())[0]
        keys = sorted(handle[f"data/{demo_id}/obs"].keys())
    rgb = [key for key in keys if key.endswith("_image")]
    low_dim = [key for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos") if key in keys]
    return {"camera_keys": rgb, "low_dim_keys": low_dim}
