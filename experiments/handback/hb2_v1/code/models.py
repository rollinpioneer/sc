"""Fixed HB2 prediction model matrix."""
from __future__ import annotations

import torch
from torch import nn


def _fixed_local_identity() -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.stack(torch.meshgrid(
        torch.linspace(-1.0, 1.0, 4),
        torch.linspace(-1.0, 1.0, 4),
        indexing="ij",
    ), dim=-1).reshape(16, 2)
    frequencies = torch.arange(32, dtype=torch.float32)
    frequencies = 2.0 ** (frequencies / 8.0)
    position = torch.cat([
        torch.sin(coordinates[:, axis:axis + 1] * frequencies)
        for axis in (0, 1)
    ] + [
        torch.cos(coordinates[:, axis:axis + 1] * frequencies)
        for axis in (0, 1)
    ], dim=-1)
    camera = torch.zeros(2, 128, dtype=torch.float32)
    camera[0, 0] = 1.0
    camera[1, 0] = -1.0
    return position.reshape(1, 1, 16, 128), camera.reshape(1, 1, 2, 1, 128)


class HB2Predictor(nn.Module):
    def __init__(self, model_id: str, *, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.model_id = model_id
        self.history_length = 1 if model_id == "M5_single" else 4
        self.use_global = model_id == "M2_global"
        self.use_local = model_id in ("M3_local", "M4_paired", "M5_single")
        self.use_visual = self.use_global or self.use_local
        if model_id not in ("M1_proprio", "M2_global", "M3_local", "M4_paired", "M5_single"):
            raise ValueError(f"unsupported neural HB2 model: {model_id}")
        self.proprio_project = nn.Sequential(nn.Linear(16, 128), nn.LayerNorm(128), nn.GELU())
        if self.use_global:
            self.global_project = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.GELU())
        if self.use_local:
            self.patcher = nn.Linear(384, 128)
            self.query = nn.Linear(16, 128)
            position, camera = _fixed_local_identity()
            self.register_buffer("position", position)
            self.register_buffer("camera", camera)
        self.gru = nn.GRU(128, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size + 1, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 14))

    def _frame_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        proprio_action = torch.cat([inputs["proprio"], inputs["base_actions"]], dim=-1)
        streams = []
        streams.append(self.proprio_project(proprio_action))
        if self.use_global:
            streams.append(self.global_project(inputs["global"].reshape(inputs["global"].shape[0], 4, -1)))
        if self.use_local:
            tokens = self.patcher(inputs["local"])
            tokens = tokens + self.position + self.camera
            query = self.query(proprio_action).unsqueeze(2).unsqueeze(3)
            scores = (query * tokens).sum(-1) / (128.0 ** 0.5)
            weights = scores.reshape(scores.shape[0], scores.shape[1], -1).softmax(dim=-1)
            weights = weights.reshape(scores.shape[0], scores.shape[1], 2, 16, 1)
            streams.append((tokens * weights).sum(dim=(2, 3)))
        return torch.stack(streams, dim=0).mean(dim=0)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        frame = self._frame_features(inputs)
        if self.history_length == 1:
            frame = frame[:, -1:, :]
            time = inputs["time"][:, -1:, :]
        else:
            time = inputs["time"]
        sequence, _ = self.gru(frame)
        return self.head(torch.cat([sequence[:, -1], time[:, -1]], dim=-1))


class HandoffPredictor(nn.Module):
    def __init__(self, visual: bool, local: bool = False, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.visual = visual
        self.local = local
        self.proprio_project = nn.Sequential(nn.Linear(16, 128), nn.LayerNorm(128), nn.GELU())
        if visual and not local:
            self.global_project = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.GELU())
        if visual and local:
            self.patcher = nn.Linear(384, 128)
            self.query = nn.Linear(16, 128)
            position, camera = _fixed_local_identity()
            self.register_buffer("position", position)
            self.register_buffer("camera", camera)
        self.gru = nn.GRU(128, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size + 2, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 3))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        proprio_action = torch.cat([inputs["proprio"], inputs["base_actions"]], dim=-1)
        frame = self.proprio_project(proprio_action)
        if self.visual and not self.local:
            frame = (frame + self.global_project(inputs["global"].reshape(inputs["global"].shape[0], 4, -1))) / 2
        elif self.visual:
            tokens = self.patcher(inputs["local"]) + self.position + self.camera
            query = self.query(proprio_action).unsqueeze(2).unsqueeze(3)
            scores = (query * tokens).sum(-1) / (128.0 ** 0.5)
            weights = scores.reshape(scores.shape[0], scores.shape[1], -1).softmax(-1)
            weights = weights.reshape(scores.shape[0], scores.shape[1], 2, 16, 1)
            frame = (frame + (tokens * weights).sum((2, 3))) / 2
        sequence, _ = self.gru(frame)
        return self.head(torch.cat([
            sequence[:, -1],
            inputs["time"][:, -1],
            inputs["helper_elapsed"],
        ], dim=-1))


def build_model(model_id: str, *, head: str = "anchor", hidden_size: int = 128, dropout: float = 0.1) -> nn.Module:
    if head == "handoff":
        if model_id not in ("proprio", "selected_visual_global", "selected_visual_local"):
            raise ValueError(f"unsupported HB2 handoff model: {model_id}")
        return HandoffPredictor(visual=model_id != "proprio", local=model_id == "selected_visual_local",
                                hidden_size=hidden_size, dropout=dropout)
    return HB2Predictor(model_id, hidden_size=hidden_size, dropout=dropout)
