"""Model-only inputs for learned long-horizon counterfactual prediction."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TASK_ID = {"can": 0, "square": 1}


@lru_cache(maxsize=128)
def load_feature(path: str) -> dict:
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


class CounterfactualVerifierDataset(Dataset):
    """Never returns outcome labels or proposer metadata as model inputs."""

    def __init__(self, samples: str | Path | pd.DataFrame, feature_index: str | Path | pd.DataFrame, normalizer: str | Path, horizon: int = 100, history: int = 4):
        self.samples = samples.copy() if isinstance(samples, pd.DataFrame) else pd.read_parquet(samples)
        self.samples = self.samples.sort_values("replacement_id").reset_index(drop=True)
        index = feature_index.copy() if isinstance(feature_index, pd.DataFrame) else pd.read_parquet(feature_index)
        self.features = {(str(row.task), str(row.demo_id)): str(row.feature_path) for row in index.itertuples(index=False)}
        with np.load(normalizer) as data:
            self.norm = {key: data[key].copy() for key in data.files}
        self.horizon, self.history = int(horizon), int(history)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def normed(values, mean, std):
        return np.clip((values - mean) / np.maximum(std, 1e-6), -10.0, 10.0).astype(np.float32)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        feature = load_feature(self.features[(str(row.task), str(row.perturbed_demo_id))])
        images = feature["image_emb"].astype(np.float32); states = feature["states_pre"].astype(np.float32)
        mask = feature["state_valid_mask"].astype(bool); actions = feature["actions"].astype(np.float32)
        t, total = int(row.query_t), len(actions)
        if not 0 <= t < total:
            raise ValueError(f"invalid query_t for {row.replacement_id}")
        count = min(self.horizon, total - t)
        reference = np.zeros((self.horizon, 7), np.float32); replacement = np.zeros((self.horizon, 7), np.float32); continuation_mask = np.zeros(self.horizon, bool)
        reference[:count] = actions[t:t + count]; replacement[:count] = reference[:count]; replacement[0] = np.asarray(row.replacement_action, np.float32)
        # ``history`` is the number of context frames, so the corresponding
        # delta window contains at most ``history - 1`` transitions.  The
        # frozen v0.2-R contract therefore uses history_start=max(0, t-3)
        # for the configured history_length=4.
        start = max(0, t - max(self.history - 1, 0))
        if t == 0:
            image_delta = np.zeros(768, np.float32); state_delta = np.zeros(states.shape[1], np.float32)
        else:
            image_delta = (images[start + 1:t + 1] - images[start:t]).mean(0).astype(np.float32)
            state_delta = (states[start + 1:t + 1] - states[start:t]).mean(0).astype(np.float32)
        state = self.normed(states[t], self.norm["state_mean"], self.norm["state_std"])
        state_delta = self.normed(state_delta, self.norm["state_delta_mean"], self.norm["state_delta_std"])
        state[~mask] = 0.0; state_delta[~mask] = 0.0
        reference = self.normed(reference, self.norm["action_mean"], self.norm["action_std"])
        replacement_sequence = self.normed(replacement, self.norm["action_mean"], self.norm["action_std"])
        current_raw = actions[t]
        replacement_raw = np.asarray(row.replacement_action, np.float32)
        current = self.normed(current_raw, self.norm["action_mean"], self.norm["action_std"])
        replacement_action = self.normed(replacement_raw, self.norm["action_mean"], self.norm["action_std"])
        replacement_sequence[0] = replacement_action
        action_delta = self.normed(replacement_raw - current_raw, self.norm["action_delta_mean"], self.norm["action_delta_std"])
        direct = np.concatenate([current, replacement_action, action_delta]).astype(np.float32)
        continuation_mask[:count] = True
        scalar = np.asarray([
            t / max(total - 1, 1),
            (float(row.state_distance) - float(self.norm["state_distance_mean"])) / max(float(self.norm["state_distance_std"]), 1e-6),
            (float(row.action_delta_l2) - float(self.norm["action_distance_mean"])) / max(float(self.norm["action_distance_std"]), 1e-6),
        ], np.float32)
        # Strings identify query groups for ranking and rows for evaluation, not model inputs.
        return {
            "image_t": torch.from_numpy(images[t]), "history_image_delta_mean": torch.from_numpy(image_delta),
            "state_t": torch.from_numpy(state), "history_state_delta_mean": torch.from_numpy(state_delta),
            "reference_actions": torch.from_numpy(reference), "replacement_actions": torch.from_numpy(replacement_sequence),
            "continuation_mask": torch.from_numpy(continuation_mask), "direct_action": torch.from_numpy(direct),
            # Keep the three action vectors addressable individually as part
            # of the frozen dataset contract; ``direct_action`` is only the
            # fixed 21-dimensional concatenation consumed by the model.
            "current_action": torch.from_numpy(current),
            "replacement_action": torch.from_numpy(replacement_action),
            "action_delta": torch.from_numpy(action_delta),
            "state_valid_mask": torch.from_numpy(mask.astype(np.float32)),
            "task_id": torch.tensor(TASK_ID[str(row.task)], dtype=torch.long), "scalars": torch.from_numpy(scalar),
            "relative_position": torch.tensor(float(scalar[0]), dtype=torch.float32),
            "state_distance": torch.tensor(float(scalar[1]), dtype=torch.float32),
            "action_delta_l2": torch.tensor(float(scalar[2]), dtype=torch.float32),
            "target_score": torch.tensor(float(row.counterfactual_improvement_long), dtype=torch.float32),
            "target_positive": torch.tensor(float(row.target_positive), dtype=torch.float32),
            "replacement_id": str(row.replacement_id), "query_group_id": str(row.query_group_id),
        }
