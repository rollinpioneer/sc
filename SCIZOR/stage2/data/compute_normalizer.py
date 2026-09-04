"""Compute train-only normalizers for Stage 2 cached features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALAR_NAMES = [
    "motion_norm", "rotation_norm", "delta_action_norm", "gripper_action", "gripper_delta",
    "state_delta_norm", "image_delta_norm", "interaction_proxy", "relative_position",
]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def scalar_features(actions: np.ndarray, states: np.ndarray, image_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_action = np.zeros_like(actions)
    delta_action[1:] = actions[1:] - actions[:-1]
    state_delta = np.zeros_like(states)
    state_delta[:-1] = states[1:] - states[:-1]
    image_delta = np.zeros_like(image_emb, dtype=np.float32)
    image_delta[:-1] = image_emb[1:].astype(np.float32) - image_emb[:-1].astype(np.float32)
    motion = np.linalg.norm(actions[:, :3], axis=1)
    rotation = np.linalg.norm(actions[:, 3:6], axis=1)
    da_norm = np.linalg.norm(delta_action, axis=1)
    state_norm = np.linalg.norm(state_delta, axis=1)
    image_norm = np.linalg.norm(image_delta, axis=1)
    gripper = actions[:, 6]
    gripper_delta = delta_action[:, 6]
    interaction = (gripper < 0).astype(np.float32) * state_norm
    relative = np.linspace(0.0, 1.0, len(actions), dtype=np.float32)
    scalar = np.stack([motion, rotation, da_norm, gripper, gripper_delta, state_norm, image_norm, interaction, relative], axis=1).astype(np.float32)
    return state_delta, delta_action, scalar


def moments(count: np.ndarray, total: np.ndarray, total_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(total_sq, count, out=np.ones_like(total), where=count > 0) - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))
    mean[count == 0] = 0.0
    std[count == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    opt = args()
    index = pd.read_parquet(opt.feature_index)
    index = index[index["split"] == opt.split]
    if index.empty:
        raise ValueError(f"no demos for split {opt.split}")
    state_dim = int(index["state_pad_dim"].iloc[0])
    state_count = np.zeros(state_dim, dtype=np.float64)
    state_total = np.zeros(state_dim, dtype=np.float64); state_sq = np.zeros(state_dim, dtype=np.float64)
    delta_total = np.zeros(state_dim, dtype=np.float64); delta_sq = np.zeros(state_dim, dtype=np.float64)
    action_total = np.zeros(7, dtype=np.float64); action_sq = np.zeros(7, dtype=np.float64)
    action_delta_total = np.zeros(7, dtype=np.float64); action_delta_sq = np.zeros(7, dtype=np.float64)
    scalar_total = np.zeros(len(SCALAR_NAMES), dtype=np.float64); scalar_sq = np.zeros(len(SCALAR_NAMES), dtype=np.float64)
    total_frames = 0
    for row in index.itertuples(index=False):
        with np.load(row.feature_path) as x:
            states = x["states"].astype(np.float32); actions = x["actions"].astype(np.float32)
            valid = x["state_valid_mask"].astype(bool); image = x["image_emb"]
        state_delta, action_delta, scalars = scalar_features(actions, states, image)
        n = len(actions); total_frames += n
        state_count[valid] += n
        state_total[valid] += states[:, valid].sum(axis=0); state_sq[valid] += (states[:, valid] ** 2).sum(axis=0)
        delta_total[valid] += state_delta[:, valid].sum(axis=0); delta_sq[valid] += (state_delta[:, valid] ** 2).sum(axis=0)
        action_total += actions.sum(axis=0); action_sq += (actions**2).sum(axis=0)
        action_delta_total += action_delta.sum(axis=0); action_delta_sq += (action_delta**2).sum(axis=0)
        scalar_total += scalars.sum(axis=0); scalar_sq += (scalars**2).sum(axis=0)
    state_mean, state_std = moments(state_count, state_total, state_sq)
    state_delta_mean, state_delta_std = moments(state_count, delta_total, delta_sq)
    action_mean, action_std = moments(np.full(7, total_frames), action_total, action_sq)
    action_delta_mean, action_delta_std = moments(np.full(7, total_frames), action_delta_total, action_delta_sq)
    scalar_mean, scalar_std = moments(np.full(len(SCALAR_NAMES), total_frames), scalar_total, scalar_sq)
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(opt.output, state_mean=state_mean, state_std=state_std, state_delta_mean=state_delta_mean, state_delta_std=state_delta_std, action_mean=action_mean, action_std=action_std, action_delta_mean=action_delta_mean, action_delta_std=action_delta_std, scalar_mean=scalar_mean, scalar_std=scalar_std, scalar_names=np.asarray(SCALAR_NAMES))
    summary = {"split": opt.split, "demo_count": int(len(index)), "transition_count": total_frames, "state_pad_dim": state_dim, "scalar_names": SCALAR_NAMES}
    opt.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
