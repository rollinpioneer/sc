"""Non-privileged t=80 stop/continue features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
HISTORY_FRAMES = 4
ACTION_DIM = 7
FEATURE_DIM = HISTORY_FRAMES * 9 + HISTORY_FRAMES * ACTION_DIM + 1


def _frame_vector(archive, index: int) -> np.ndarray:
    parts = []
    for key in PROPRIO_KEYS:
        name = f"{index}_{key}"
        if name not in archive:
            raise KeyError(f"handoff history is missing {name}")
        parts.append(np.asarray(archive[name], dtype=np.float32).reshape(-1))
    vector = np.concatenate(parts)
    if vector.shape != (9,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid proprio frame {index}: {vector.shape}")
    return vector


def load_handoff_history(path: Path, horizon: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [4,9] proprio, [4,7] base actions, and [4] absolute times."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(metadata) != HISTORY_FRAMES:
        raise ValueError(f"expected four handoff frames, got {len(metadata)}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        proprio = np.stack([_frame_vector(archive, index) for index in range(HISTORY_FRAMES)])
        actions = np.stack([
            np.asarray(archive[f"{index}_base_action"], dtype=np.float32).reshape(-1)
            for index in range(HISTORY_FRAMES)
        ])
    if actions.shape != (HISTORY_FRAMES, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError(f"invalid base-action history: {actions.shape}")
    times = np.asarray([int(row["absolute_t"]) for row in metadata], dtype=np.int64)
    if times.shape != (HISTORY_FRAMES,) or np.any(np.diff(times) < 0) or times[-1] != 80:
        raise ValueError(f"handoff history is not the frozen t=80 window: {times.tolist()}")
    if np.any(times < 0) or np.any(times >= horizon):
        raise ValueError(f"handoff times outside horizon: {times.tolist()}")
    return proprio, actions, times


def feature_vector(proprio: np.ndarray, actions: np.ndarray, times: np.ndarray, horizon: int = 400) -> np.ndarray:
    proprio = np.asarray(proprio, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    times = np.asarray(times, dtype=np.int64)
    if proprio.shape != (4, 9) or actions.shape != (4, 7) or times.shape != (4,):
        raise ValueError(f"feature shape mismatch: proprio={proprio.shape}, actions={actions.shape}, times={times.shape}")
    if not np.isfinite(proprio).all() or not np.isfinite(actions).all():
        raise ValueError("non-finite stop/continue input")
    current_time = np.asarray([float(times[-1]) / float(horizon)], dtype=np.float32)
    result = np.concatenate([proprio.reshape(-1), actions.reshape(-1), current_time]).astype(np.float32)
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"invalid feature vector: {result.shape}")
    return result


def feature_from_history(path: Path, horizon: int = 400) -> tuple[np.ndarray, dict]:
    proprio, actions, times = load_handoff_history(path, horizon)
    vector = feature_vector(proprio, actions, times, horizon)
    return vector, {
        "history_path": str(Path(path).resolve()),
        "absolute_times": times.tolist(),
        "input_shape": [4, 9, 4, 7, 1],
        "feature_dim": int(vector.size),
        "feature_sha256": hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest(),
    }


def audit_input_schema() -> dict:
    return {
        "schema_version": "hb3p_stop_continue_input_schema_v1",
        "feature_dim": FEATURE_DIM,
        "proprio_keys": list(PROPRIO_KEYS),
        "base_action_history": {"frames": HISTORY_FRAMES, "dim_per_frame": ACTION_DIM, "source": "L60 handoff_history"},
        "time": "current absolute_t divided by horizon",
        "allowed": ["proprio", "base_action_history", "time"],
        "forbidden": ["reward", "success", "object_state", "privileged_state", "future_result", "root_id", "seed", "repair_action", "images"],
    }

