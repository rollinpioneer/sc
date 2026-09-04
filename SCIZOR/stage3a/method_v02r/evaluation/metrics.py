"""Dependency-light binary ranking metrics used by frozen Stage 3 evaluation."""
from __future__ import annotations

import numpy as np


def binary(labels, scores):
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    mask = np.isfinite(s); y, s = y[mask], s[mask]
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return {"auroc": None, "auprc": None, "n": int(len(y)), "positive": pos, "prevalence": None}
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]: j += 1
        ranks[order[i:j]] = (i + j + 1) / 2; i = j
    auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    ordered = y[np.argsort(-s, kind="mergesort")]
    ap = float(((np.cumsum(ordered) / np.arange(1, len(ordered) + 1)) * ordered).sum() / pos)
    return {"auroc": auc, "auprc": ap, "n": int(len(y)), "positive": pos, "prevalence": float(pos / len(y))}


def threshold_metrics(labels, scores, threshold):
    y, s = np.asarray(labels, bool), np.asarray(scores, float); pred = s >= float(threshold)
    positives, negatives = y.sum(), (~y).sum()
    return {"threshold": float(threshold), "effective_recall": float(pred[y].mean()) if positives else None, "no_effect_far": float(pred[~y].mean()) if negatives else None, "predicted_positive_count": int(pred.sum())}
