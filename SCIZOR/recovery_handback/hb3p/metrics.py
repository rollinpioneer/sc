"""Ranking and complete-episode metrics for HB3-P."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Optional

import numpy as np


def binary_ranking(
    y: Iterable[int], score: Iterable[float], weights: Optional[Iterable[float]] = None
) -> dict:
    raw_y = np.asarray(list(y))
    if raw_y.ndim != 1 or not np.isin(raw_y, [0, 1]).all():
        raise ValueError("truth values must be binary")
    y_array = raw_y.astype(np.int64)
    scores = np.asarray(list(score), dtype=np.float64)
    weight = (
        np.ones(len(y_array), np.float64)
        if weights is None
        else np.asarray(list(weights), np.float64)
    )
    if scores.shape != y_array.shape or weight.shape != y_array.shape:
        raise ValueError("ranking arrays must have identical one-dimensional shape")
    if not np.isfinite(scores).all() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("invalid ranking score or weight")
    keep = weight > 0
    y_array, scores, weight = y_array[keep], scores[keep], weight[keep]
    positive = float(weight[y_array == 1].sum())
    negative = float(weight[y_array == 0].sum())
    if positive == 0 or negative == 0:
        return {
            "auroc": None,
            "average_precision": None,
            "ranking_not_estimable": "single_class_or_empty",
        }
    order = np.argsort(scores, kind="stable")
    y_array, scores, weight = y_array[order], scores[order], weight[order]
    bounds = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1, len(y_array)]
    groups = []
    cumulative_negative = 0.0
    numerator = 0.0
    for start, end in zip(bounds[:-1], bounds[1:]):
        group_positive = float(weight[start:end][y_array[start:end] == 1].sum())
        group_negative = float(weight[start:end][y_array[start:end] == 0].sum())
        numerator += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        groups.append((group_positive, group_negative))
    true_positive = false_positive = average_precision = 0.0
    for group_positive, group_negative in reversed(groups):
        true_positive += group_positive
        false_positive += group_negative
        average_precision += (
            group_positive / positive * true_positive / (true_positive + false_positive)
        )
    return {
        "auroc": numerator / (positive * negative),
        "average_precision": average_precision,
        "ranking_not_estimable": False,
    }


def equal_root_weights(group_ids: Iterable[str]) -> np.ndarray:
    ids = list(group_ids)
    counts = Counter(ids)
    return np.asarray([1.0 / counts[group_id] for group_id in ids], np.float64)


def episode_value(row: dict, penalty: float = 0.25, denominator: float = 400.0) -> float:
    if not row.get("engineering_ok", False):
        raise ValueError("engineering failure is not a task failure")
    helped = int(row["takeover_count"]) > 0
    autonomous = (
        bool(row["genuine_handoff_success"])
        if helped
        else bool(row["system_success"])
    )
    cost = int(row["helper_steps_actual"])
    if not 0 <= cost <= 80:
        raise ValueError("helper budget violation")
    return float(autonomous) - float(penalty) * cost / float(denominator)


def paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    repeats: int = 2000,
    seed: int = 20260909,
) -> dict:
    if set(left) != set(right) or not left:
        raise ValueError("all compared methods must cover the same nonempty root set")
    ids = sorted(left)
    delta = np.asarray([left[group_id] - right[group_id] for group_id in ids], np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite paired outcome")
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(ids), size=(repeats, len(ids)))].mean(axis=1)
    return {
        "root_count": len(ids),
        "point_difference": float(delta.mean()),
        "ci95_percentile": np.quantile(samples, [0.025, 0.975]).tolist(),
        "repeats": int(repeats),
        "seed": int(seed),
        "unit": "stat_group_id",
    }


def interference_counts(rows: list[dict], baseline: dict[str, bool]) -> dict:
    ids = [str(row["stat_group_id"]) for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(baseline):
        raise ValueError("one complete episode per root is required")
    failed = sum(
        bool(baseline[group_id]) and not bool(row["system_success"])
        for group_id, row in zip(ids, rows)
    )
    eligible = sum(bool(value) for value in baseline.values())
    touched = sum(
        bool(baseline[group_id]) and int(row["takeover_count"]) > 0
        for group_id, row in zip(ids, rows)
    )
    return {
        "count": int(failed),
        "all_baseline_success_roots": int(eligible),
        "helped_baseline_success_roots": int(touched),
        "unconditional_rate": failed / eligible if eligible else None,
        "conditional_rate": failed / touched if touched else None,
    }


def binary_nll(y: Iterable[int], score: Iterable[float], weights=None) -> float:
    y_array = np.asarray(list(y), np.float64)
    score_array = np.clip(np.asarray(list(score), np.float64), 1e-7, 1 - 1e-7)
    weight = np.ones(len(y_array)) if weights is None else np.asarray(weights, np.float64)
    losses = -(y_array * np.log(score_array) + (1 - y_array) * np.log1p(-score_array))
    return float(np.average(losses, weights=weight))


def binary_brier(y: Iterable[int], score: Iterable[float], weights=None) -> float:
    errors = (np.asarray(list(score), np.float64) - np.asarray(list(y), np.float64)) ** 2
    weight = np.ones(len(errors)) if weights is None else np.asarray(weights, np.float64)
    return float(np.average(errors, weights=weight))
