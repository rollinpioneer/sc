"""Root-equal probability and one-shot decision helpers."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


def finite_probs(logits: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]:
    logits = np.asarray(logits, np.float64) / float(temperature)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    result = {"p0": sigmoid(logits[:, 0]), "p_full": sigmoid(logits[:, 13])}
    for length, start in ((5, 1), (20, 5), (80, 9)):
        values = logits[:, start:start + 4]
        values = np.exp(values - values.max(axis=1, keepdims=True))
        values /= values.sum(axis=1, keepdims=True)
        result[f"p_sys_{length}"] = 1.0 - values[:, 0]
        result[f"p_genuine_{length}"] = values[:, 3]
        result[f"p_helper_{length}"] = values[:, 1]
    return result


def root_mean(rows: Iterable[dict], value_key: str) -> float | None:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None and math.isfinite(float(value)):
            grouped[str(row["stat_group_id"])].append(float(value))
    return float(np.mean([np.mean(values) for values in grouped.values()])) if grouped else None


def root_equal_mean(values: Iterable[float], group_ids: Iterable[str]) -> float | None:
    grouped = defaultdict(list)
    for group_id, value in zip(group_ids, values):
        if math.isfinite(float(value)):
            grouped[str(group_id)].append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()])) if grouped else None


def choose_length(probabilities: dict[str, float], lambda_value: float, costs: dict[int, float] | None = None,
                  denominator: float = 400.0, tolerance: float = 1e-8) -> int:
    costs = costs or {0: 0.0, 5: 5.0, 20: 20.0, 80: 80.0}
    scored = []
    for length in (0, 5, 20, 80):
        probability = float(probabilities["p0"] if length == 0 else probabilities[f"p_genuine_{length}"])
        scored.append((probability - lambda_value * float(costs[length]) / denominator, length))
    best_score = max(score for score, _ in scored)
    return min(length for score, length in scored if best_score - score <= tolerance)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def nll_binary(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return float(-np.mean(np.asarray(y, float) * np.log(p) + (1 - np.asarray(y, float)) * np.log1p(-p)))


def root_probability_metrics(rows: list[dict], truth_key: str, probability_key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["stat_group_id"])].append((float(row[truth_key]), float(row[probability_key])))
    values = []
    for pairs in groups.values():
        y = np.asarray([pair[0] for pair in pairs]); p = np.asarray([pair[1] for pair in pairs])
        values.append({"brier": brier(y, p), "nll": nll_binary(y, p), "n": len(pairs)})
    result = {"roots": len(values), "anchors": sum(value["n"] for value in values)}
    for key in ("brier", "nll"):
        result[key] = float(np.mean([value[key] for value in values])) if values else None
    y_all = np.asarray([pair[0] for pairs in groups.values() for pair in pairs])
    p_all = np.asarray([pair[1] for pairs in groups.values() for pair in pairs])
    result["positive_anchors"] = int(y_all.sum())
    result["positive_roots"] = int(sum(any(pair[0] for pair in pairs) for pairs in groups.values()))
    result["negative_anchors"] = int(len(y_all) - y_all.sum())
    result["negative_roots"] = int(sum(any(not bool(pair[0]) for pair in pairs) for pairs in groups.values()))
    result["prevalence_anchor"] = float(y_all.mean()) if len(y_all) else None
    edges = np.linspace(0.0, 1.0, 11)
    bins = []
    for index in range(10):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (p_all >= lower) & ((p_all < upper) if index < 9 else (p_all <= upper))
        bins.append({
            "lower": lower,
            "upper": upper,
            "count": int(mask.sum()),
            "mean_probability": float(p_all[mask].mean()) if mask.any() else None,
            "observed_rate": float(y_all[mask].mean()) if mask.any() else None,
        })
    result["reliability_bins"] = bins
    if len(np.unique(y_all)) < 2:
        result["auroc"] = None; result["average_precision"] = None; result["ranking_not_estimable"] = True
    else:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            result["auroc"] = float(roc_auc_score(y_all, p_all))
            result["average_precision"] = float(average_precision_score(y_all, p_all))
            result["ranking_not_estimable"] = False
        except ImportError:
            result["auroc"] = None; result["average_precision"] = None; result["ranking_not_estimable"] = "sklearn_unavailable"
    return result
