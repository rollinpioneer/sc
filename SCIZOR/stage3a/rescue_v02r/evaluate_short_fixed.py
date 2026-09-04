"""Diagnostic-only correction for the v0.2 short-horizon field-name mismatch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HORIZONS, H_WEIGHTS = (10, 20, 40), (0.2, 0.3, 0.5)


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def delta(row, name, horizon):
    for key in (f"{name}_delta_h{horizon}", f"{name}_mean_delta_h{horizon}"):
        if key in row:
            return float(row[key])
    return float("nan")


def metrics(labels, scores):
    y, s = np.asarray(labels, int), np.asarray(scores, float)
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return {"auroc": None, "auprc": None, "n": int(len(y)), "positive": int(pos.sum())}
    order = np.argsort(s, kind="mergesort"); rank = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]: j += 1
        rank[order[i:j]] = (i + j + 1) / 2; i = j
    auc = float((rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
    yy = y[np.argsort(-s, kind="mergesort")]; ap = float((np.cumsum(yy) / np.arange(1, len(yy) + 1) * yy).sum() / pos.sum())
    return {"auroc": auc, "auprc": ap, "n": int(len(y)), "positive": int(pos.sum())}


def normalizer(data):
    out = {}
    for task in ("can", "square"):
        out[task] = {}
        train = [r for r in data if r.get("split") == "train" and r.get("task") == task and r.get("finite_target", True)]
        for h in HORIZONS:
            out[task][str(h)] = {}
            for kind in ("dense", "stage"):
                x = np.abs(np.asarray([delta(r, kind, h) for r in train], float)); x = x[np.isfinite(x) & (x > 1e-12)]
                out[task][str(h)][f"{kind}_scale"] = float(np.quantile(x, .9)) if len(x) else 1.0
    return out


def score(row, norm):
    value = 0.0
    for h, hw in zip(HORIZONS, H_WEIGHTS):
        n = norm[row["task"]][str(h)]
        value += hw * (0.4 * np.clip(delta(row, "dense", h) / n["dense_scale"], -1, 1) + 0.5 * np.clip(delta(row, "stage", h) / n["stage_scale"], -1, 1) + 0.1 * float(row[f"success_delta_h{h}"]))
    return float(value)


def evaluate(data, norm, split):
    part = [r for r in data if r["split"] == split]
    return {"metrics": metrics([bool(r["is_effective_intervention"]) for r in part], [score(r, norm) for r in part]), "score_count": len(part)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--paired-clean", type=Path, required=True); p.add_argument("--feasible", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    paired, feasible = rows(a.paired_clean), rows(a.feasible)
    pn, fn = normalizer(paired), normalizer(feasible)
    output = {"status": "DIAGNOSTIC_ONLY_NOT_FINAL_GATE", "paired_normalizer": pn, "feasible_normalizer": fn, "paired_validation": evaluate(paired, pn, "validation"), "primary_feasible_validation": evaluate([r for r in feasible if int(r["replacement_rank"]) == 0], fn, "validation")}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(output, indent=2)); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
