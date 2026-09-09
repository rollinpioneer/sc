"""Frozen M0/M1 predictors and common online history handling."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from recovery_handback.common import sha256_file
from recovery_handback.hb2.features import CAMERA_KEYS, PROPRIO_KEYS
from recovery_handback.hb2.metrics import choose_length, finite_probs
from recovery_handback.hb2.models import build_model


def proprio_vector(obs: dict) -> np.ndarray:
    value = np.concatenate([
        np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in PROPRIO_KEYS
    ])
    if value.shape != (9,) or not np.isfinite(value).all():
        raise ValueError(f"invalid proprio vector: {value.shape}")
    return value


def history_arrays(history: list[dict], horizon: int = 400) -> dict[str, np.ndarray]:
    if not 1 <= len(history) <= 4:
        raise ValueError("history must contain 1..4 frames")
    frames = [history[0]] * (4 - len(history)) + list(history)
    rgb = np.asarray([
        [np.asarray(frame["obs"][key], dtype=np.uint8) for key in CAMERA_KEYS]
        for frame in frames
    ], dtype=np.uint8)
    proprio = np.stack([proprio_vector(frame["obs"]) for frame in frames]).astype(np.float32)
    actions = np.stack([np.asarray(frame["base_action"], np.float32).reshape(7) for frame in frames])
    times = np.asarray([int(frame["absolute_t"]) for frame in frames], np.int64)
    return {"rgb": rgb, "proprio": proprio, "base_actions": actions, "absolute_times": times}


class M0Predictor:
    def __init__(self, protocol: dict):
        entry = protocol["models"]["M0_time"]
        checkpoint = Path(entry["checkpoint"])
        if sha256_file(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError("M0 checkpoint hash mismatch")
        self.payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.temperature = float(entry["temperature"])
        self.decision = protocol["decision"]

    def predict(self, history: list[dict], absolute_t: int) -> dict:
        del history
        started = time.perf_counter()
        entry = self.payload["by_time"].get(str(int(absolute_t)), self.payload["overall"])
        logits = np.asarray([[
            np.log(entry["p0"] / (1 - entry["p0"])),
            *[np.log(max(value, 1e-7)) for group in entry["categories"] for value in group],
            np.log(entry["pfull"] / (1 - entry["pfull"])),
        ]], np.float32)
        values = finite_probs(logits, self.temperature)
        result = {key: float(value[0]) for key, value in values.items()}
        result["selected_length"] = choose_length(
            result,
            float(self.decision["primary_lambda"]),
            denominator=float(self.decision["cost_denominator"]),
            tolerance=float(self.decision["tie_tolerance"]),
        )
        result.update(backbone_seconds=0.0, prediction_head_seconds=time.perf_counter() - started)
        result["inference_seconds"] = result["prediction_head_seconds"]
        return result


class M1Predictor:
    def __init__(self, protocol: dict, *, device: str = "cuda"):
        entry = protocol["models"]["M1_proprio"]
        checkpoint = Path(entry["checkpoint"])
        if sha256_file(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError("M1 checkpoint hash mismatch")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = build_model("M1_proprio").to(self.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.temperature = float(entry["temperature"])
        self.decision = protocol["decision"]
        normalizer = json.loads(Path(protocol["normalizer"]).read_text(encoding="utf-8"))
        self.pm = np.asarray(normalizer["proprio_mean"], np.float32)
        self.ps = np.asarray(normalizer["proprio_std"], np.float32)
        self.am = np.asarray(normalizer["action_mean"], np.float32)
        self.ass = np.asarray(normalizer["action_std"], np.float32)

    def predict(self, history: list[dict], absolute_t: int) -> dict:
        started = time.perf_counter()
        arrays = history_arrays(history)
        if int(arrays["absolute_times"][-1]) != int(absolute_t):
            raise ValueError("history absolute time mismatch")
        inputs = {
            "proprio": torch.from_numpy(((arrays["proprio"] - self.pm) / self.ps)[None]).to(self.device),
            "base_actions": torch.from_numpy(((arrays["base_actions"] - self.am) / self.ass)[None]).to(self.device),
            "time": torch.from_numpy((arrays["absolute_times"].astype(np.float32) / 400.0)[None, :, None]).to(self.device),
        }
        head_started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(inputs).cpu().numpy()
        head_seconds = time.perf_counter() - head_started
        values = finite_probs(logits, self.temperature)
        result = {key: float(value[0]) for key, value in values.items()}
        result["selected_length"] = choose_length(
            result,
            float(self.decision["primary_lambda"]),
            denominator=float(self.decision["cost_denominator"]),
            tolerance=float(self.decision["tie_tolerance"]),
        )
        result.update(backbone_seconds=0.0, prediction_head_seconds=head_seconds, inference_seconds=time.perf_counter() - started)
        return result
