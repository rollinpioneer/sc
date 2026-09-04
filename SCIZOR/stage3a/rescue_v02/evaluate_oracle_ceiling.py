"""Evaluate frozen v0.2 paired-clean and feasible-oracle ceilings on validation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HORIZONS, H_WEIGHTS = (10, 20, 40), (0.2, 0.3, 0.5)


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def auc_ap(labels, scores):
    y, s = np.asarray(labels, np.int64), np.asarray(scores, float)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if not n_pos or not n_neg: return {"auroc": None, "auprc": None, "n": len(y), "positive": n_pos}
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0; i = j
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    desc = np.argsort(-s, kind="mergesort"); yy = y[desc]; tp = np.cumsum(yy); ap = float((tp / np.arange(1, len(y) + 1) * yy).sum() / n_pos)
    return {"auroc": auc, "auprc": ap, "n": int(len(y)), "positive": n_pos}


def scales(rows):
    out = {}
    for task in ("can", "square"):
        out[task] = {}
        subset = [r for r in rows if r.get("split") == "train" and r.get("task") == task and r.get("finite_target", True) and int(r.get("actual_horizon", 0)) >= 10]
        for h in HORIZONS:
            out[task][str(h)] = {}
            for key in ("dense_mean_delta", "stage_mean_delta"):
                values = np.asarray([float(r.get(f"{key}_h{h}", np.nan)) for r in subset], float); values = values[np.isfinite(values)]
                out[task][str(h)][key] = float(max(np.quantile(np.abs(values), .90), 1e-6)) if len(values) else 1e-6
    return out


def score(row, normalizer):
    total = 0.0
    for h, hw in zip(HORIZONS, H_WEIGHTS):
        sc = normalizer[row["task"]][str(h)]
        dense = np.clip(float(row[f"dense_delta_h{h}"] if f"dense_delta_h{h}" in row else row[f"dense_mean_delta_h{h}"]) / sc["dense_mean_delta"], -1, 1)
        stage = np.clip(float(row[f"stage_delta_h{h}"] if f"stage_delta_h{h}" in row else row[f"stage_mean_delta_h{h}"]) / sc["stage_mean_delta"], -1, 1)
        success = float(row[f"success_delta_h{h}"])
        total += hw * (0.4 * dense + 0.5 * stage + 0.1 * success)
    return float(total)


def engineering(rows, paired=False):
    keys = ["branch_pre_state_equal", "reference_exact_all_horizons", "finite_target"]
    if paired: keys.append("paired_clean_exact_all_horizons")
    return {f"{key}_rate": float(sum(bool(r.get(key, False)) for r in rows) / len(rows)) if rows else 0.0 for key in keys}


def distribution(rows):
    return {name: {"n": len(values), "median": float(np.median(values)) if values else None, "mean": float(np.mean(values)) if values else None} for name, values in (("effective", [r["score"] for r in rows if r["is_effective_intervention"]]), ("no_effect", [r["score"] for r in rows if not r["is_effective_intervention"]]))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--paired-clean", type=Path, required=True); p.add_argument("--feasible", type=Path, required=True); p.add_argument("--split", default="validation"); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    if a.split != "validation": raise ValueError("ceiling gate is validation-only")
    paired, feasible = read_rows(a.paired_clean), read_rows(a.feasible)
    for rows, name in ((paired, "paired"), (feasible, "feasible")):
        if any(r.get("split") not in {"train", "validation"} for r in rows): raise ValueError(f"{name}: forbidden split")
    paired_norm, feasible_norm = scales(paired), scales(feasible)
    for r in paired: r["score"] = score(r, paired_norm)
    for r in feasible: r["score"] = score(r, feasible_norm)
    pval = [r for r in paired if r["split"] == a.split]
    fval = [r for r in feasible if r["split"] == a.split]
    primary = [r for r in fval if int(r["replacement_rank"]) == 0]
    by_pair = defaultdict(list)
    for r in fval: by_pair[r["pair_id"]].append(r)
    best = [max(rs, key=lambda r: r["score"]) for rs in by_pair.values()]
    def metric(rows): return auc_ap([int(bool(r["is_effective_intervention"])) for r in rows], [r["score"] for r in rows])
    paired_metrics, primary_metrics, best_metrics = metric(pval), metric(primary), metric(best)
    p_eng, f_eng = engineering(paired, paired=True), engineering(feasible)
    engineering_pass = (p_eng["branch_pre_state_equal_rate"] == 1.0 and p_eng["reference_exact_all_horizons_rate"] >= .999 and p_eng["paired_clean_exact_all_horizons_rate"] >= .999 and p_eng["finite_target_rate"] >= .99 and f_eng["branch_pre_state_equal_rate"] == 1.0 and f_eng["reference_exact_all_horizons_rate"] >= .999 and f_eng["finite_target_rate"] >= .99)
    method_pass = bool(paired_metrics["auroc"] is not None and paired_metrics["auroc"] >= .70 and primary_metrics["auroc"] is not None and primary_metrics["auroc"] >= .70)
    result = {"split": a.split, "paired_clean": {"metrics": paired_metrics, "engineering": p_eng, "normalizer": paired_norm, "score_distribution": distribution(pval)}, "primary_feasible": {"metrics": primary_metrics, "score_distribution": distribution(primary)}, "best_of_4_feasible": {"metrics": best_metrics, "score_distribution": distribution(best)}, "feasible_engineering": f_eng, "feasible_normalizer": feasible_norm, "validation_pair_count": len(pval), "validation_feasible_rows": len(fval), "engineering_pass": engineering_pass, "method_pass": method_pass, "oracle_ceiling_pass": bool(engineering_pass and method_pass), "frozen_thresholds": {"validation_paired_clean_auroc": .70, "validation_primary_feasible_auroc": .70}}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
