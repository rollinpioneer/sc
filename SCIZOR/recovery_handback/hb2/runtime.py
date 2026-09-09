"""Runtime-safe, single-query F/H predictor interfaces."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from recovery_handback.hb2.features import CAMERA_KEYS, ObservationFeaturizer, _proprio
from recovery_handback.hb2.metrics import choose_length, finite_probs
from recovery_handback.hb2.models import build_model


class RuntimePredictor:
    def __init__(self, config: dict | Path, protocol: dict | Path, *, device: str | None = None):
        self.config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else config
        self.protocol = json.loads(Path(protocol).read_text()) if isinstance(protocol, (str, Path)) else protocol
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.featurizer = ObservationFeaturizer(self.config, device=str(self.device))
        selected = self.protocol["selected_visual_model"]
        entry = self.protocol.get("models", {}).get(selected, {}).get("seed0") or self.protocol.get("canonical", {})
        checkpoint = Path(entry["checkpoint"])
        self.model = build_model(selected).to(self.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False); self.model.load_state_dict(state["state_dict"]); self.model.eval()
        self.temperature = float(self.protocol.get("selection", {}).get("canonical", {}).get("temperature", 1.0))

    def _features(self, history_rgb: list[dict[str, np.ndarray]], history_proprio: np.ndarray,
                  history_base_actions: np.ndarray, absolute_t: int) -> dict[str, torch.Tensor]:
        images = np.asarray([[np.asarray(frame[key]) for key in CAMERA_KEYS] for frame in history_rgb], dtype=np.uint8)
        images = np.transpose(images, (0, 1, 4, 2, 3)).reshape(1, 8, 3, images.shape[2], images.shape[3])
        global_, local_ = self.featurizer.encode_images(images)
        return {"global": torch.from_numpy(global_).to(self.device), "local": torch.from_numpy(local_).to(self.device),
                "proprio": torch.from_numpy(np.asarray(history_proprio, np.float32)[None]).to(self.device),
                "base_actions": torch.from_numpy(np.asarray(history_base_actions, np.float32)[None]).to(self.device),
                "time": torch.full((1,4,1), float(absolute_t)/float(self.config['horizon_steps']), device=self.device)}

    def predict_before_takeover(self, history_rgb, history_proprio, history_base_actions, absolute_t):
        started=time.perf_counter(); inputs=self._features(history_rgb,history_proprio,history_base_actions,absolute_t)
        with torch.no_grad(): logits=self.model(inputs).cpu().numpy()
        probs=finite_probs(logits,self.temperature); scalar={key:float(value[0]) for key,value in probs.items()}
        length=choose_length(scalar,float(self.protocol['decision']['primary_lambda']))
        return {**scalar,'selected_length':length,'inference_seconds':time.perf_counter()-started}

    def predict_at_handoff(self, history_rgb, history_proprio, history_base_actions, absolute_t, elapsed_helper_steps):
        started=time.perf_counter(); return {'q_complete':None,'q_strict':None,'input_support':'fixed_length_handoff_only','elapsed_helper_steps':int(elapsed_helper_steps),'absolute_t':int(absolute_t),'inference_seconds':time.perf_counter()-started}

