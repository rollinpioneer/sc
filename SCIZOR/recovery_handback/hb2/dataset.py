"""Whitelisted HB2 tensors and root-balanced sampling."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


class _Cache:
    def __init__(self, feature_dir: Path, name: str):
        with np.load(feature_dir / f"{name}.npz", allow_pickle=False) as handle:
            self.arrays = {key: np.asarray(handle[key]) for key in handle.files}
        payload = json.loads((feature_dir / f"{name}.samples.json").read_text(encoding="utf-8"))
        self.sample_ids = list(payload["sample_ids"])
        self.index = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}


def _normalizer(path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path is None or not path.is_file():
        return np.zeros(9, np.float32), np.ones(9, np.float32), np.zeros(7, np.float32), np.ones(7, np.float32)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (np.asarray(payload["proprio_mean"], np.float32), np.asarray(payload["proprio_std"], np.float32),
            np.asarray(payload["action_mean"], np.float32), np.asarray(payload["action_std"], np.float32))


class AnchorDataset(Dataset):
    def __init__(self, rows: list[dict], feature_dir: Path, *, split: str,
                 normalizer_path: Path | None = None):
        self.rows = [row for row in rows if row.get("split") == split and bool(row.get("complete_pair"))]
        self.cache = _Cache(feature_dir, "anchor")
        self.pm, self.ps, self.am, self.ase = _normalizer(normalizer_path)
        self.root_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            if row["example_id"] not in self.cache.index:
                raise KeyError(f"feature missing {row['example_id']}")
            self.root_to_indices[str(row["stat_group_id"])].append(index)
        self.roots = sorted(self.root_to_indices)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        item = self.cache.index[row["example_id"]]
        tensors = {
            "global": self.cache.arrays["global"][item].astype(np.float32),
            "local": self.cache.arrays["local"][item].astype(np.float32),
            "proprio": (self.cache.arrays["proprio"][item] - self.pm) / self.ps,
            "base_actions": (self.cache.arrays["base_actions"][item] - self.am) / self.ase,
            "time": self.cache.arrays["time"][item].astype(np.float32),
            "y0": np.asarray(row["y0"], np.float32),
            "yfull": np.asarray(row["y_full"], np.float32),
            "categories": np.asarray([row[f"category_l{length}"] for length in (5, 20, 80)], np.int64),
            "delta": np.asarray([row[f"gain_autonomy_l{length}"] for length in (5, 20, 80)], np.float32),
            "root_id": str(row["stat_group_id"]),
            "example_id": str(row["example_id"]),
        }
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in tensors.items()}


class HandoffDataset(Dataset):
    def __init__(self, rows: list[dict], feature_dir: Path, *, split: str,
                 normalizer_path: Path | None = None):
        self.rows = [row for row in rows if row.get("split") == split and bool(row.get("eligible"))]
        self.cache = _Cache(feature_dir, "handoff")
        self.pm, self.ps, self.am, self.ase = _normalizer(normalizer_path)
        self.root_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            if row["example_id"] not in self.cache.index:
                raise KeyError(f"feature missing {row['example_id']}")
            self.root_to_indices[str(row["stat_group_id"])].append(index)
        self.roots = sorted(self.root_to_indices)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        item = self.cache.index[row["example_id"]]
        values = {
            "global": self.cache.arrays["global"][item].astype(np.float32),
            "local": self.cache.arrays["local"][item].astype(np.float32),
            "proprio": (self.cache.arrays["proprio"][item] - self.pm) / self.ps,
            "base_actions": (self.cache.arrays["base_actions"][item] - self.am) / self.ase,
            "time": self.cache.arrays["time"][item].astype(np.float32),
            "target": np.asarray(row["handoff_category"], np.int64),
            "root_id": str(row["stat_group_id"]),
            "example_id": str(row["example_id"]),
        }
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in values.items()}


class RootBalancedBatchSampler:
    def __init__(self, dataset: AnchorDataset | HandoffDataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if not self.dataset.roots:
            return
        count = max(1, int(np.ceil(len(self.dataset) / self.batch_size)))
        batches = []
        for _ in range(count):
            roots = rng.choice(self.dataset.roots, size=min(self.batch_size, len(self.dataset.roots)), replace=False)
            indices = []
            for root in roots:
                indices.append(int(rng.choice(self.dataset.root_to_indices[root])))
            while len(indices) < self.batch_size:
                root = str(rng.choice(self.dataset.roots))
                indices.append(int(rng.choice(self.dataset.root_to_indices[root])))
            batches.append(indices)
        return iter(batches)

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.dataset) / self.batch_size)))

