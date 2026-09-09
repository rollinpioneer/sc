"""Small proprio/action-history/time stop-versus-continue model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recovery_handback.common import sha256_file
from recovery_handback.hb3p_stop.features import FEATURE_DIM, feature_vector


class StopContinueMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: tuple[int, int] = (64, 32)):
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden[0])), nn.LayerNorm(int(hidden[0])), nn.GELU(),
            nn.Linear(int(hidden[0]), int(hidden[1])), nn.GELU(),
            nn.Linear(int(hidden[1]), 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def save_normalizer(path: Path, mean: np.ndarray, std: np.ndarray, *, fit_indices: list[int]) -> None:
    payload = {
        "schema_version": "hb3p_stop_continue_normalizer_v1",
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
        "feature_dim": int(len(mean)),
        "fit_indices": [int(index) for index in fit_indices],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_predictor(checkpoint: Path, normalizer: Path, *, device: str = "cpu"):
    if not Path(checkpoint).is_file() or not Path(normalizer).is_file():
        raise FileNotFoundError("stop/continue checkpoint or normalizer is missing")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = StopContinueMLP(int(state["input_dim"]), tuple(state["hidden"])).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    payload = json.loads(Path(normalizer).read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(payload["std"], dtype=np.float32), 1e-6)
    threshold = float(state["threshold"])

    def predict(history: list[dict], absolute_t: int, horizon: int = 400) -> dict:
        if int(absolute_t) != 80 or len(history) < 4:
            raise ValueError("stop/continue model is frozen for one query at t=80 with four frames")
        frames = history[-4:]
        proprio = np.stack([
            np.concatenate([np.asarray(frame["obs"][key], dtype=np.float32).reshape(-1)
                            for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")])
            for frame in frames
        ])
        actions = np.stack([np.asarray(frame["base_action"], dtype=np.float32).reshape(7) for frame in frames])
        times = np.asarray([int(frame["absolute_t"]) for frame in frames], dtype=np.int64)
        values = feature_vector(proprio, actions, times, horizon)
        normalized = torch.from_numpy(((values - mean) / std)[None]).to(device)
        with torch.inference_mode():
            score = float(torch.sigmoid(model(normalized))[0].cpu())
        return {
            "continue_score": score,
            "threshold": threshold,
            "decision": "CONTINUE" if score >= threshold else "STOP",
            "feature_dim": int(values.size),
        }

    return predict
