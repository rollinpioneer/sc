"""Balanced query-group sampler retaining all replacements from a query in one batch."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class QueryGroupBatchSampler(Sampler[list[int]]):
    def __init__(self, samples: pd.DataFrame, groups_per_batch: int, seed: int):
        self.samples = samples.reset_index(drop=True)
        self.groups_per_batch = int(groups_per_batch); self.seed = int(seed); self.epoch = 0
        grouped = self.samples.groupby("query_group_id", sort=True).indices
        self.groups = {str(name): list(indices) for name, indices in grouped.items()}
        strata = {}
        for name, indices in self.groups.items():
            part = self.samples.iloc[indices]
            key = (str(part.task.iloc[0]), bool(part.target_positive.any()))
            strata.setdefault(key, []).append(name)
        expected = {(task, positive) for task in ("can", "square") for positive in (False, True)}
        if set(strata) != expected or any(not values for values in strata.values()):
            raise ValueError(f"missing query sampling strata: {set(strata)}")
        self.strata = strata
        self.batches = max(1, math.ceil(len(self.groups) / self.groups_per_batch))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        labels = sorted(self.strata)
        count = self.groups_per_batch // len(labels)
        remainder = self.groups_per_batch - count * len(labels)
        for _ in range(self.batches):
            names = []
            for pos, key in enumerate(labels):
                n = count + (1 if pos < remainder else 0)
                names.extend(rng.choice(self.strata[key], size=n, replace=True).tolist())
            rng.shuffle(names)
            yield [index for name in names for index in self.groups[name]]
