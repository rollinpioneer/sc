"""Fixed-horizon responsibility dataset with train-only normalization."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from stage2.data.compute_normalizer import scalar_features

TASK_TO_ID = {"can": 0, "square": 1}
EFFECT_GROUP_TO_ID = {
    "positive": 0,
    "no_effect_control": 1,
    "effective_hard_negative": 2,
    "clean_control": 3,
}


@lru_cache(maxsize=32)
def load_feature_file(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as x:
        return {key: x[key].copy() for key in x.files}


class ResponsibilityDataset(Dataset):
    def __init__(self, chunk_index: str | Path | pd.DataFrame, normalizer: str | Path, max_horizon: int = 40, split: str | None = None):
        self.samples = chunk_index.copy() if isinstance(chunk_index, pd.DataFrame) else pd.read_parquet(chunk_index)
        if split is not None:
            self.samples = self.samples[self.samples.split == split]
        self.samples = self.samples.sort_values("sample_id").reset_index(drop=True)
        self.max_horizon = max_horizon
        with np.load(normalizer) as x:
            self.norm = {key: x[key].copy() for key in x.files if key != "scalar_names"}

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return np.clip((x - mean) / np.maximum(std, 1e-6), -10.0, 10.0).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        row = self.samples.iloc[index]
        sample_type = str(row["sample_type"])
        # Inference samples do not carry a supervision subtype; use the clean
        # control ID solely to keep the collated batch schema valid.
        effect_group_id = EFFECT_GROUP_TO_ID.get(sample_type, EFFECT_GROUP_TO_ID["clean_control"] if sample_type == "inference" else None)
        if effect_group_id is None:
            raise ValueError(f"unknown sample_type={sample_type}")
        feature = load_feature_file(str(row.feature_path))
        start, end = int(row.start_t), int(row.end_t)
        horizon = end - start
        if horizon > self.max_horizon or horizon < 1:
            raise ValueError(f"invalid horizon {horizon}")
        image = feature["image_emb"].astype(np.float32)
        # Stage 3 v0.2 caches the explicitly pre-action-aligned state under
        # ``states_pre``. Preserve the original Stage 2 cache contract while
        # accepting that frozen-equivalent representation for inference.
        states = feature["states"] if "states" in feature else feature["states_pre"]
        states = states.astype(np.float32)
        actions = feature["actions"].astype(np.float32)
        valid_state = feature["state_valid_mask"].astype(bool)
        state_delta, action_delta, scalars = scalar_features(actions, states, image)
        image_delta = np.zeros_like(image, dtype=np.float32)
        image_delta[:-1] = image[1:] - image[:-1]
        chunk_image = image[start:end]
        chunk_image_delta = image_delta[start:end]
        chunk_state = self._normalize(states[start:end], self.norm["state_mean"], self.norm["state_std"])
        chunk_state_delta = self._normalize(state_delta[start:end], self.norm["state_delta_mean"], self.norm["state_delta_std"])
        chunk_state[:, ~valid_state] = 0.0; chunk_state_delta[:, ~valid_state] = 0.0
        norm_action = self._normalize(actions[start:end], self.norm["action_mean"], self.norm["action_std"])
        norm_action_delta = self._normalize(action_delta[start:end], self.norm["action_delta_mean"], self.norm["action_delta_std"])
        norm_scalar = self._normalize(scalars[start:end], self.norm["scalar_mean"], self.norm["scalar_std"])
        action_features = np.concatenate([norm_action, norm_action_delta, norm_scalar], axis=1)
        output = {
            "image": np.zeros((self.max_horizon, 768), dtype=np.float32), "image_delta": np.zeros((self.max_horizon, 768), dtype=np.float32),
            "state": np.zeros((self.max_horizon, states.shape[1]), dtype=np.float32), "state_delta": np.zeros((self.max_horizon, states.shape[1]), dtype=np.float32),
            "action_features": np.zeros((self.max_horizon, 23), dtype=np.float32), "valid_mask": np.zeros(self.max_horizon, dtype=bool),
            "target_distribution": np.zeros(self.max_horizon, dtype=np.float32),
        }
        output["image"][:horizon] = chunk_image; output["image_delta"][:horizon] = chunk_image_delta
        output["state"][:horizon] = chunk_state; output["state_delta"][:horizon] = chunk_state_delta
        output["action_features"][:horizon] = action_features; output["valid_mask"][:horizon] = True
        if int(row.target_effect):
            center = int(row.target_center_t) - start
            target_indices = [t for t in range(int(row.target_region_start_t), int(row.target_region_end_t) + 1) if start <= t < end]
            weights = np.asarray([(.5 if t == int(row.target_center_t) else .25) for t in target_indices], dtype=np.float32)
            output["target_distribution"][[t - start for t in target_indices]] = weights / weights.sum()
        else:
            center = -1
        endpoint = min(end, len(image) - 1)
        context_state = self._normalize((states[endpoint] - states[start])[None], self.norm["state_delta_mean"], self.norm["state_delta_std"])[0]
        context_state[~valid_state] = 0.0
        result = {key: torch.from_numpy(value) for key, value in output.items()}
        result.update({
            "context_image_delta": torch.from_numpy((image[endpoint] - image[start]).astype(np.float32)),
            "context_state_delta": torch.from_numpy(context_state),
            "target_effect": torch.tensor(float(row.target_effect), dtype=torch.float32),
            "target_center_index": torch.tensor(center, dtype=torch.long), "task_id": torch.tensor(TASK_TO_ID[row.task], dtype=torch.long),
            "V_c": torch.tensor(float(row.V_c), dtype=torch.float32), "horizon_ratio": torch.tensor(horizon / self.max_horizon, dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.pair_sample_weight), dtype=torch.float32), "sample_id": row.sample_id,
            "effect_group_id": torch.tensor(effect_group_id, dtype=torch.long),
        })
        return result


def build_pair_balanced_binary_sampler(train_df: pd.DataFrame, *, seed: int) -> tuple[WeightedRandomSampler, dict[str, float]]:
    """Assign exactly 0.5 sampler mass to each binary effect class."""
    target = train_df["target_effect"].to_numpy(dtype=np.int64)
    pair_weight = train_df["pair_sample_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(pair_weight).all():
        raise ValueError("pair_sample_weight contains non-finite values")
    if np.any(pair_weight <= 0):
        raise ValueError("pair_sample_weight must be positive")
    positive, negative = target == 1, target == 0
    if not positive.any() or not negative.any():
        raise ValueError("training split must contain positive and negative samples")
    weights = np.zeros(len(train_df), dtype=np.float64)
    weights[positive] = 0.5 * pair_weight[positive] / pair_weight[positive].sum()
    weights[negative] = 0.5 * pair_weight[negative] / pair_weight[negative].sum()
    positive_mass, negative_mass = float(weights[positive].sum()), float(weights[negative].sum())
    if not np.isclose(positive_mass, 0.5, atol=1e-8):
        raise AssertionError(positive_mass)
    if not np.isclose(negative_mass, 0.5, atol=1e-8):
        raise AssertionError(negative_mass)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_df), replacement=True, generator=generator)
    return sampler, {
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "num_samples": int(len(train_df)),
        "num_positive_rows": int(positive.sum()),
        "num_negative_rows": int(negative.sum()),
    }


def make_dataloader(dataset: ResponsibilityDataset, batch_size: int, train: bool = False, num_workers: int = 0, seed: int = 0) -> DataLoader:
    if not train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sampler, summary = build_pair_balanced_binary_sampler(dataset.samples, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    loader.sampler_summary = summary
    return loader
