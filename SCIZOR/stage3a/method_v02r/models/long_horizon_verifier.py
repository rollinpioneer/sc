"""Fixed full and action-only long-horizon counterfactual verifier architecture."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


def block(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())


class LongHorizonCounterfactualVerifier(nn.Module):
    def __init__(self, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.image_t = block(768, 128); self.image_delta = block(768, 64)
        self.state_t = block(state_dim, 96); self.state_delta = block(state_dim, 48)
        self.task = nn.Embedding(2, 8); self.gru = nn.GRU(7, 96, batch_first=True)
        self.direct = nn.Sequential(nn.Linear(21, 64), nn.GELU())
        self.full_trunk = nn.Sequential(nn.Linear(128 + 64 + 96 + 48 + 8 + 3 + 96 * 3 + 64, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout))
        self.action_trunk = nn.Sequential(nn.Linear(8 + 3 + 96 * 3 + 64, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout))
        self.score_head = nn.Linear(128, 1); self.positive_head = nn.Linear(128, 1)

    def encode_actions(self, values: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.sum(1).clamp(min=1).to(torch.long).cpu()
        packed = pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, batch: dict[str, Tensor], mode: str = "full") -> dict[str, Tensor]:
        h_ref = self.encode_actions(batch["reference_actions"], batch["continuation_mask"])
        h_replacement = self.encode_actions(batch["replacement_actions"], batch["continuation_mask"])
        action = torch.cat([h_ref, h_replacement, h_replacement - h_ref, self.direct(batch["direct_action"])], 1)
        task = self.task(batch["task_id"])
        if mode == "full":
            context = torch.cat([self.image_t(batch["image_t"]), self.image_delta(batch["history_image_delta_mean"]), self.state_t(batch["state_t"]), self.state_delta(batch["history_state_delta_mean"]), task, batch["scalars"]], 1)
            hidden = self.full_trunk(torch.cat([context, action], 1))
        elif mode == "action_only":
            hidden = self.action_trunk(torch.cat([task, batch["scalars"], action], 1))
        else:
            raise ValueError(f"unknown verifier mode {mode}")
        logit = self.score_head(hidden).squeeze(1)
        return {"pred_score": 0.9 * torch.sigmoid(logit), "pred_positive_logit": self.positive_head(hidden).squeeze(1)}
