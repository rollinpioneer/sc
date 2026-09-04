"""Single frozen 100-frame / 3-frame-persistent outcome definition."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_json(path): return json.loads(Path(path).read_text())


def stage_delta_trace(replacement, reference):
    repl, ref = np.asarray(replacement, float), np.asarray(reference, float)
    if repl.shape != ref.shape: raise ValueError(f"staged reward shape mismatch: {repl.shape}/{ref.shape}")
    if repl.ndim == 1: return repl - ref
    if repl.ndim == 2: return np.max(repl - ref, axis=1) if repl.shape[1] else np.zeros(len(repl))
    raise ValueError(f"unsupported staged reward ndim: {repl.ndim}")


def persistent_score(trace, window):
    x = np.asarray(trace, float).reshape(-1)
    if not len(x): return float("nan"), -1
    width = min(max(int(window), 1), len(x)); rolled = np.convolve(x, np.ones(width) / width, mode="valid"); index = int(np.argmax(rolled))
    return float(rolled[index]), index


def score_outcomes(*, task, reference_rewards, replacement_rewards, reference_staged, replacement_staged, reference_success, replacement_success, normalizer, spec):
    max_horizon, window = int(spec["max_horizon"]), int(spec["persistence_window"])
    rr, rp = np.asarray(reference_rewards, float)[:max_horizon], np.asarray(replacement_rewards, float)[:max_horizon]
    sr, sp = np.asarray(reference_staged)[:max_horizon], np.asarray(replacement_staged)[:max_horizon]
    yr, yp = np.asarray(reference_success, float)[:max_horizon], np.asarray(replacement_success, float)[:max_horizon]
    n = min(len(rr), len(rp), len(sr), len(sp), len(yr), len(yp))
    rr, rp, sr, sp, yr, yp = rr[:n], rp[:n], sr[:n], sp[:n], yr[:n], yp[:n]
    scale = normalizer[task]; dense = rp - rr; stage = stage_delta_trace(sp, sr); success = yp - yr
    weights = spec["component_weights"]; clip = float(spec["component_clip"])
    component = weights["dense"] * np.clip(dense / float(scale["dense_scale"]), -clip, clip) + weights["stage"] * np.clip(stage / float(scale["stage_scale"]), -clip, clip) + weights["success"] * success
    value, index = persistent_score(component, window)
    out = {"counterfactual_improvement_long": value, "peak_window_start": index, "actual_horizon": n, "finite_target": bool(np.isfinite(component).all())}
    for h in spec["diagnostic_horizons"]:
        x, ix = persistent_score(component[: int(h)], window); out[f"counterfactual_improvement_h{h}"] = x; out[f"peak_window_start_h{h}"] = ix
    return out
