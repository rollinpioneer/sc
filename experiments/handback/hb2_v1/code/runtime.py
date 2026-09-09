"""Runtime-safe, single-query F/H predictor interfaces."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from recovery_handback.common import sha256_file
from recovery_handback.hb2.features import CAMERA_KEYS, ObservationFeaturizer, _as_rgb
from recovery_handback.hb2.metrics import choose_length, finite_probs
from recovery_handback.hb2.models import build_model


def _load_json(value: dict | str | Path | None) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _normalizer(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        np.asarray(payload["proprio_mean"], np.float32),
        np.asarray(payload["proprio_std"], np.float32),
        np.asarray(payload["action_mean"], np.float32),
        np.asarray(payload["action_std"], np.float32),
    )


def _verified_checkpoint(entry: dict) -> Path:
    checkpoint = Path(entry["checkpoint"])
    expected = entry.get("checkpoint_sha256") or entry.get("sha256")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if expected and sha256_file(checkpoint) != expected:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    return checkpoint


class RuntimePredictor:
    def __init__(
        self,
        config: dict | str | Path,
        protocol: dict | str | Path,
        handoff_protocol: dict | str | Path | None = None,
        *,
        device: str | None = None,
    ):
        self.config = _load_json(config)
        self.protocol = _load_json(protocol)
        self.handoff_protocol = _load_json(handoff_protocol)
        assert self.config is not None and self.protocol is not None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.featurizer = ObservationFeaturizer(self.config, device=str(self.device))

        selected = self.protocol["selected_visual_model"]
        entry = self.protocol["models"][selected]["seed0"]
        checkpoint = _verified_checkpoint(entry)
        self.model = build_model(selected).to(self.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.temperature = float(entry["temperature"])
        self.f_normalizer = _normalizer(self.protocol["normalizer"])

        self.handoff_model = None
        self.handoff_temperature = None
        self.h_normalizer = None
        if self.handoff_protocol:
            h_entry = self.handoff_protocol["models"]["selected_visual"]["seed0"]
            h_checkpoint = _verified_checkpoint(h_entry)
            architecture = str(h_entry["architecture"])
            self.handoff_model = build_model(architecture, head="handoff").to(self.device)
            h_state = torch.load(h_checkpoint, map_location="cpu", weights_only=False)
            self.handoff_model.load_state_dict(h_state["state_dict"])
            self.handoff_model.eval()
            self.handoff_temperature = float(h_entry["temperature"])
            self.h_normalizer = _normalizer(self.handoff_protocol["normalizer"])

    def _features(
        self,
        history_rgb: list[dict[str, np.ndarray]],
        history_proprio: np.ndarray,
        history_base_actions: np.ndarray,
        absolute_t: int,
        *,
        elapsed_helper_steps: int = 0,
        handoff: bool = False,
    ) -> tuple[dict[str, torch.Tensor], float]:
        count = len(history_rgb)
        if count < 1 or count > 4:
            raise ValueError(f"runtime history must contain 1..4 frames, got {count}")
        proprio = np.asarray(history_proprio, np.float32)
        actions = np.asarray(history_base_actions, np.float32)
        if proprio.shape != (count, 9) or actions.shape != (count, 7):
            raise ValueError(f"history shape mismatch: proprio={proprio.shape}, actions={actions.shape}")
        if not np.isfinite(proprio).all() or not np.isfinite(actions).all():
            raise ValueError("runtime low-dimensional input contains non-finite values")

        pad = 4 - count
        rgb = [history_rgb[0]] * pad + list(history_rgb)
        proprio = np.concatenate([np.repeat(proprio[:1], pad, axis=0), proprio], axis=0)
        actions = np.concatenate([np.repeat(actions[:1], pad, axis=0), actions], axis=0)
        original_times = list(range(int(absolute_t) - count + 1, int(absolute_t) + 1))
        absolute_times = [original_times[0]] * pad + original_times

        images = np.asarray(
            [[_as_rgb(frame[key]) for key in CAMERA_KEYS] for frame in rgb],
            dtype=np.uint8,
        )
        image_tensor = np.transpose(images, (0, 1, 4, 2, 3))
        flat = image_tensor.reshape(
            1, 8, 3, image_tensor.shape[-2], image_tensor.shape[-1]
        )
        backbone_started = time.perf_counter()
        global_flat, local_flat = self.featurizer.encode_images(flat)
        backbone_seconds = time.perf_counter() - backbone_started
        global_features = global_flat.reshape(1, 4, 2, 384).astype(np.float16).astype(np.float32)
        local_features = local_flat.reshape(1, 4, 2, 16, 384).astype(np.float16).astype(np.float32)

        normalizer = self.h_normalizer if handoff else self.f_normalizer
        if normalizer is None:
            raise RuntimeError("handoff normalizer is unavailable")
        pm, ps, am, action_std = normalizer
        inputs = {
            "global": torch.from_numpy(global_features.astype(np.float32)).to(self.device),
            "local": torch.from_numpy(local_features.astype(np.float32)).to(self.device),
            "proprio": torch.from_numpy(((proprio - pm) / ps)[None]).to(self.device),
            "base_actions": torch.from_numpy(((actions - am) / action_std)[None]).to(self.device),
            "time": torch.from_numpy(
                (np.asarray(absolute_times, np.float32) / float(self.config["horizon_steps"]))[None, :, None]
            ).to(self.device),
            "helper_elapsed": torch.tensor(
                [[float(elapsed_helper_steps) / float(self.config["horizon_steps"])]],
                dtype=torch.float32,
                device=self.device,
            ),
        }
        return inputs, backbone_seconds

    def predict_before_takeover(
        self, history_rgb, history_proprio, history_base_actions, absolute_t
    ) -> dict:
        started = time.perf_counter()
        inputs, backbone_seconds = self._features(
            history_rgb, history_proprio, history_base_actions, absolute_t
        )
        head_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(inputs).cpu().numpy()
        head_seconds = time.perf_counter() - head_started
        probabilities = finite_probs(logits, self.temperature)
        scalar = {key: float(value[0]) for key, value in probabilities.items()}
        length = choose_length(
            scalar,
            float(self.protocol["decision"]["primary_lambda"]),
            denominator=float(self.protocol["decision"]["cost_denominator"]),
            tolerance=float(self.protocol["decision"]["tie_tolerance"]),
        )
        return {
            **scalar,
            "selected_length": length,
            "backbone_seconds": backbone_seconds,
            "prediction_head_seconds": head_seconds,
            "inference_seconds": time.perf_counter() - started,
        }

    def predict_at_handoff(
        self,
        history_rgb,
        history_proprio,
        history_base_actions,
        absolute_t,
        elapsed_helper_steps,
    ) -> dict:
        if self.handoff_model is None or self.handoff_temperature is None:
            raise RuntimeError("frozen handoff protocol was not loaded")
        started = time.perf_counter()
        inputs, backbone_seconds = self._features(
            history_rgb,
            history_proprio,
            history_base_actions,
            absolute_t,
            elapsed_helper_steps=int(elapsed_helper_steps),
            handoff=True,
        )
        head_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.handoff_model(inputs) / self.handoff_temperature
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        head_seconds = time.perf_counter() - head_started
        return {
            "q_complete": float(probabilities[1] + probabilities[2]),
            "q_strict": float(probabilities[2]),
            "input_support": self.handoff_protocol["input_support"],
            "elapsed_helper_steps": int(elapsed_helper_steps),
            "absolute_t": int(absolute_t),
            "backbone_seconds": backbone_seconds,
            "prediction_head_seconds": head_seconds,
            "inference_seconds": time.perf_counter() - started,
        }
