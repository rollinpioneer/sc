"""Compute verifier normalization from train-only pre-action feature caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def moments(total, square, count):
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(square, count, out=np.ones_like(total), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    mean[count == 0], std[count == 0] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-index", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    index = pd.read_parquet(a.feature_index)
    index = index[index.split.eq(a.split)].copy()
    if index.empty:
        raise ValueError(f"no {a.split} features")
    dim = int(index.state_pad_dim.iloc[0])
    state_count = np.zeros(dim); state_sum = np.zeros(dim); state_sq = np.zeros(dim)
    delta_sum = np.zeros(dim); delta_sq = np.zeros(dim)
    action_sum = np.zeros(7); action_sq = np.zeros(7); action_delta_sum = np.zeros(7); action_delta_sq = np.zeros(7)
    scalar_sum = np.zeros(2); scalar_sq = np.zeros(2); frames = 0
    for row in index.itertuples(index=False):
        with np.load(row.feature_path) as x:
            states, mask, actions = x["states_pre"].astype(np.float32), x["state_valid_mask"].astype(bool), x["actions"].astype(np.float32)
        state_delta = np.zeros_like(states); state_delta[1:] = states[1:] - states[:-1]
        action_delta = np.zeros_like(actions); action_delta[1:] = actions[1:] - actions[:-1]
        n = len(actions); frames += n
        state_count[mask] += n
        for values, total, square in ((states, state_sum, state_sq), (state_delta, delta_sum, delta_sq)):
            total[mask] += values[:, mask].sum(0); square[mask] += (values[:, mask] ** 2).sum(0)
        action_sum += actions.sum(0); action_sq += (actions ** 2).sum(0)
        action_delta_sum += action_delta.sum(0); action_delta_sq += (action_delta ** 2).sum(0)
        distances = np.stack([np.linalg.norm(state_delta[:, mask], axis=1), np.linalg.norm(action_delta, axis=1)], axis=1)
        scalar_sum += distances.sum(0); scalar_sq += (distances ** 2).sum(0)
    state_mean, state_std = moments(state_sum, state_sq, state_count)
    delta_mean, delta_std = moments(delta_sum, delta_sq, state_count)
    full_count = np.full(7, frames, dtype=np.float64)
    action_mean, action_std = moments(action_sum, action_sq, full_count)
    action_delta_mean, action_delta_std = moments(action_delta_sum, action_delta_sq, full_count)
    scalar_mean, scalar_std = moments(scalar_sum, scalar_sq, np.full(2, frames, dtype=np.float64))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, state_mean=state_mean, state_std=state_std, state_delta_mean=delta_mean, state_delta_std=delta_std,
             action_mean=action_mean, action_std=action_std, action_delta_mean=action_delta_mean, action_delta_std=action_delta_std,
             state_distance_mean=scalar_mean[0], state_distance_std=scalar_std[0], action_distance_mean=scalar_mean[1], action_distance_std=scalar_std[1])
    summary = {"split": a.split, "feature_groups": int(len(index)), "transition_count": int(frames), "state_pad_dim": dim,
               "padding_policy": "invalid state dimensions use mean=0,std=1", "distance_features": ["adjacent_state_l2", "adjacent_action_l2"]}
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
