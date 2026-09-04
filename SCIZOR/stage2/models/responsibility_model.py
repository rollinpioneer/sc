"""Gated transition-level responsibility network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class ResponsibilityOutput:
    responsibility_logits: torch.Tensor
    rho: torch.Tensor
    effect_logit: torch.Tensor
    p_effect: torch.Tensor
    context_embedding: torch.Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResponsibilityNet(nn.Module):
    def __init__(self, *, image_dim: int, state_dim: int, action_feature_dim: int, d_model: int = 256, n_heads: int = 4, num_layers: int = 3, dim_feedforward: int = 512, dropout: float = .1, task_embedding_dim: int = 16, image_projection_dim: int = 96, image_delta_projection_dim: int = 64, state_projection_dim: int = 48, state_delta_projection_dim: int = 32, action_projection_dim: int = 64, num_tasks: int = 2, max_horizon: int = 40, model_variant: str = "full"):
        super().__init__()
        if model_variant not in {"full", "action_only"}:
            raise ValueError(f"unknown model_variant={model_variant}")
        self.model_variant, self.max_horizon = model_variant, max_horizon
        self.image_projection_dim, self.image_delta_projection_dim = image_projection_dim, image_delta_projection_dim
        self.state_projection_dim, self.state_delta_projection_dim = state_projection_dim, state_delta_projection_dim
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.image_proj = MLP(image_dim, image_projection_dim)
        self.image_delta_proj = MLP(image_dim, image_delta_projection_dim)
        self.state_proj = MLP(state_dim, state_projection_dim)
        self.state_delta_proj = MLP(state_dim, state_delta_projection_dim)
        self.action_proj = MLP(action_feature_dim, action_projection_dim)
        token_input = image_projection_dim + image_delta_projection_dim + state_projection_dim + state_delta_projection_dim + action_projection_dim + task_embedding_dim
        context_input = image_projection_dim + state_projection_dim + task_embedding_dim + 2
        self.token_fuse, self.context_fuse = MLP(token_input, d_model, d_model), MLP(context_input, d_model, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_horizon + 1, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.responsibility_head = nn.Linear(d_model, 1)
        self.effect_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> ResponsibilityOutput:
        image, image_delta = batch["image"], batch["image_delta"]
        state, state_delta, action = batch["state"], batch["state_delta"], batch["action_features"]
        valid, task_id = batch["valid_mask"].bool(), batch["task_id"].long()
        v_c, horizon = batch["V_c"].float(), batch["horizon_ratio"].float()
        batch_size, length, _ = image.shape
        if length > self.max_horizon:
            raise ValueError(f"length={length} exceeds max_horizon={self.max_horizon}")
        task = self.task_embedding(task_id); task_tokens = task[:, None, :].expand(-1, length, -1)
        if self.model_variant == "full":
            image_token, image_delta_token = self.image_proj(image), self.image_delta_proj(image_delta)
            state_token, state_delta_token = self.state_proj(state), self.state_delta_proj(state_delta)
            context_image, context_state, context_v = self.image_proj(batch["context_image_delta"]), self.state_proj(batch["context_state_delta"]), v_c
        else:
            image_token = image.new_zeros((batch_size, length, self.image_projection_dim))
            image_delta_token = image.new_zeros((batch_size, length, self.image_delta_projection_dim))
            state_token = image.new_zeros((batch_size, length, self.state_projection_dim))
            state_delta_token = image.new_zeros((batch_size, length, self.state_delta_projection_dim))
            context_image = image.new_zeros((batch_size, self.image_projection_dim))
            context_state = image.new_zeros((batch_size, self.state_projection_dim))
            context_v = torch.zeros_like(v_c)
        action_token = self.action_proj(action)
        token = self.token_fuse(torch.cat([image_token, image_delta_token, state_token, state_delta_token, action_token, task_tokens], dim=-1))
        context = self.context_fuse(torch.cat([context_image, context_state, task, context_v[:, None], horizon[:, None]], dim=-1))
        sequence = torch.cat([context[:, None], token], dim=1) + self.position_embedding[:, : length + 1]
        padding = torch.cat([torch.zeros(batch_size, 1, device=valid.device, dtype=torch.bool), ~valid], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        logits = self.responsibility_head(encoded[:, 1:]).squeeze(-1)
        # -1e9 is not representable in AMP fp16; the finite dtype minimum
        # gives the same zero-probability masked-softmax behavior.
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        rho = torch.softmax(logits, dim=-1)
        effect_logit = self.effect_head(encoded[:, 0]).squeeze(-1)
        return ResponsibilityOutput(logits, rho, effect_logit, torch.sigmoid(effect_logit), encoded[:, 0])
