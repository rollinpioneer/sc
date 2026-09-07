"""Task-agnostic potential derived from Robosuite staged rewards."""
from __future__ import annotations

import numpy as np


def stage_potential(stage_values) -> float:
    stage = np.asarray(stage_values, dtype=np.float32).reshape(-1)
    if stage.size == 0:
        return 0.0
    stage = np.clip(stage, 0.0, 1.0)
    weights = np.arange(1, stage.size + 1, dtype=np.float32)
    return float(np.dot(stage, weights) / weights.sum())
