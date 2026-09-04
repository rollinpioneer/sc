from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run_paired_clean_oracle import binary_metrics


def main():
    p = argparse.ArgumentParser(); p.add_argument("--inputs", nargs="+", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--summary", type=Path, required=True); a = p.parse_args()
    rows = []
    for path in a.inputs:
        rows.extend(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    ids = [str(r["pair_id"]) for r in rows]
    if len(ids) != len(set(ids)): raise ValueError("duplicate pair_id across paired-clean partitions")
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    engineering = {f"{key}_rate": float(sum(bool(r[key]) for r in rows) / len(rows)) if rows else 0.0 for key in ("branch_pre_state_equal", "reference_exact_all_horizons", "paired_clean_exact_all_horizons", "finite_target")}
    labels, scores = [int(bool(r["is_effective_intervention"])) for r in rows], [float(r["counterfactual_improvement"]) for r in rows]
    metrics = {}
    if len(set(labels)) == 2:
        overall = binary_metrics(labels, scores); metrics = {"overall_auroc": overall["auroc"], "overall_auprc": overall["auprc"]}
        for task in ("can", "square"):
            sub = [r for r in rows if r["task"] == task]
            if len({bool(r["is_effective_intervention"]) for r in sub}) == 2:
                got = binary_metrics([int(bool(r["is_effective_intervention"])) for r in sub], [float(r["counterfactual_improvement"]) for r in sub])
                metrics[f"{task}_auroc"] = got["auroc"]; metrics[f"{task}_auprc"] = got["auprc"]; metrics[f"{task}_n"] = len(sub)
    failure_counts = {}
    for r in rows: failure_counts[r["failure_type"]] = failure_counts.get(r["failure_type"], 0) + 1
    result = {"engineering": engineering, "metrics": metrics, "failure_type_counts": failure_counts, "effective_count": sum(labels), "no_effect_count": len(rows) - sum(labels), "pair_count": len(rows)}
    a.summary.parent.mkdir(parents=True, exist_ok=True); a.summary.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
