"""Summarize Stage 2 iteration validation/test results and fixed decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["transition_f1", "responsibility_region_iou", "mean_abs_localization_delay", "top1_within_1", "top5_hit", "recovery_retention", "innocent_downstream_retention", "expert_retention", "rare_retention", "no_effect_false_attribution_rate", "delete_rate"]


def overall(report: dict, method: str) -> dict:
    return report["methods"][method]["overall"]


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-test-metrics", type=Path, required=True)
    parser.add_argument("--candidate-selection", type=Path, required=True)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path, required=True)
    parser.add_argument("--canonical-seed", type=Path, required=True)
    parser.add_argument("--go-no-go", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--seed-summary", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_test_metrics.read_text())
    selection = json.loads(args.candidate_selection.read_text())
    valid = json.loads(args.validation_metrics.read_text())
    test = json.loads(args.test_metrics.read_text())
    canonical = json.loads(args.canonical_seed.read_text())
    decision = json.loads(args.go_no_go.read_text())

    rows = []
    history_stats = {}
    for seed in (0, 1, 2):
        run_name = f"full_seed_{seed}"
        method = f"responsibility_iter1_seed{seed}"
        selected = selection["runs"][run_name]
        v, t = overall(valid, method), overall(test, method)
        candidate_data = json.loads((args.runs_root / run_name / "candidate_checkpoints.json").read_text())
        epoch = int(candidate_data[selected["candidate_name"]]["epoch"])
        rows.append({
            "seed": str(seed), "selected_candidate": selected["candidate_name"], "selected_epoch": epoch,
            "validation_iou": v.get("responsibility_region_iou"), "validation_top1": v.get("top1_within_1"), "validation_no_effect_far": v.get("no_effect_false_attribution_rate"),
            "test_iou": t.get("responsibility_region_iou"), "test_top1": t.get("top1_within_1"), "test_mean_abs_delay": t.get("mean_abs_localization_delay"),
            "test_recovery_retention": t.get("recovery_retention"), "test_no_effect_far": t.get("no_effect_false_attribution_rate"),
        })
        history = pd.read_csv(args.runs_root / run_name / "history.csv")
        history_stats[method] = {"max_validation_gate_gap": float(history["validation_gate_gap"].max()), "max_validation_effect_balanced_accuracy": float(history["validation_effect_balanced_accuracy"].max())}
    numeric_columns = [c for c in rows[0] if c not in ("seed", "selected_candidate")]
    seed_frame = pd.DataFrame(rows)
    mean_row = {"seed": "full_mean", "selected_candidate": "NA", **{c: pd.to_numeric(seed_frame[c], errors="coerce").mean() for c in numeric_columns}}
    std_row = {"seed": "full_std", "selected_candidate": "NA", **{c: pd.to_numeric(seed_frame[c], errors="coerce").std(ddof=0) for c in numeric_columns}}
    seed_frame = pd.concat([seed_frame, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    args.seed_summary.parent.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(args.seed_summary, index=False)

    canonical_method = canonical["canonical_method"]
    ordering = [
        ("canonical_responsibility_iter1", canonical_method),
        ("responsibility_iter1_seed0", "responsibility_iter1_seed0"),
        ("responsibility_iter1_seed1", "responsibility_iter1_seed1"),
        ("responsibility_iter1_seed2", "responsibility_iter1_seed2"),
        ("action_only_iter1_seed0", "action_only_iter1_seed0"),
        ("old_canonical_responsibility", "responsibility_seed1"),
        ("future_discount", "future_discount"),
        ("original_scizor", "original_scizor"),
        ("uniform", "uniform"),
    ]
    board_rows = []
    for display, method in ordering:
        item = overall(test if method in test["methods"] else base, method)
        board_rows.append({"method": display, **{("test_delete_rate" if key == "delete_rate" else key): item.get(key) for key in METRICS}})
    pd.DataFrame(board_rows).to_csv(args.leaderboard, index=False)

    base_methods = {name: overall(base, name) for name in ("original_scizor", "uniform", "future_discount")}
    base_iou = max(float(x.get("responsibility_region_iou") or -np.inf) for x in base_methods.values())
    base_top1 = max(float(x.get("top1_within_1") or -np.inf) for x in base_methods.values())
    base_delay = min(float(x.get("mean_abs_localization_delay") or np.inf) for x in base_methods.values())
    base_recovery = max(float(x.get("recovery_retention") or -np.inf) for x in base_methods.values())
    base_far = min(float(x.get("no_effect_false_attribution_rate") or np.inf) for x in base_methods.values())
    canon, action = overall(test, canonical_method), overall(test, "action_only_iter1_seed0")
    full_test = [overall(test, f"responsibility_iter1_seed{i}") for i in (0, 1, 2)]
    table = ["| method | " + " | ".join(METRICS) + " |", "| " + " | ".join(["---"] * (len(METRICS) + 1)) + " |"]
    for row in board_rows:
        table.append("| " + " | ".join(fmt(row.get("test_delete_rate" if c == "delete_rate" else c)) for c in ["method", *METRICS]) + " |")
    sampler = [json.loads((args.runs_root / f"full_seed_{i}" / "run_config.json").read_text())["sampler"] for i in (0, 1, 2)]
    sampler_ok = all(abs(float(x["positive_mass"]) - .5) < 1e-8 and abs(float(x["negative_mass"]) - .5) < 1e-8 for x in sampler)
    full_iou_count = sum(float(x.get("responsibility_region_iou") or -np.inf) > base_iou for x in full_test)
    full_top1_count = sum(float(x.get("top1_within_1") or -np.inf) > base_top1 for x in full_test)
    report = f"""# Stage 2 Iteration 1 Report

## Fixed protocol

- Sampler positive/negative mass is `0.5 / 0.5`: **{sampler_ok}**.
- Validation deletion budget is fixed at `9616`; test thresholds were not retuned.
- Canonical method: `{canonical_method}` (full seed {canonical['canonical_seed']}).

## Validation effect supervision

The maximum validation gate gap / effect balanced accuracy observed in the three full runs was: {json.dumps(history_stats, sort_keys=True)}. Selected candidates were: {json.dumps({k: v['candidate_name'] for k, v in selection['runs'].items()}, sort_keys=True)}.

## Test answers

1. Sampler mass is exactly balanced in all full runs: **{sampler_ok}**.
2. Full-seed validation gate/effect metrics are recorded above; checkpoint selection remains validation-only.
3. Selected trackers: full seed 0=`{selection['runs']['full_seed_0']['candidate_name']}`, full seed 1=`{selection['runs']['full_seed_1']['candidate_name']}`, full seed 2=`{selection['runs']['full_seed_2']['candidate_name']}`, action-only=`{selection['runs']['action_only_seed_0']['candidate_name']}`.
4. Canonical test IoU/top-1/delay: `{canon.get('responsibility_region_iou')}` / `{canon.get('top1_within_1')}` / `{canon.get('mean_abs_localization_delay')}`; strongest baseline values are `{base_iou}` / `{base_top1}` / `{base_delay}`. Full seeds above baseline: IoU {full_iou_count}/3, top-1 {full_top1_count}/3.
5. Canonical no-effect FAR is `{canon.get('no_effect_false_attribution_rate')}`, fixed limit is `{base_far + 0.03}`; the FAR gate **{'passes' if (canon.get('no_effect_false_attribution_rate') or np.inf) <= base_far + 0.03 else 'fails'}**.
6. Canonical recovery retention is `{canon.get('recovery_retention')}`, baseline maximum is `{base_recovery}`, tolerance floor is `{base_recovery - 0.10}`.
7. Canonical versus action-only test IoU/top-1/FAR: `{canon.get('responsibility_region_iou')}` / `{canon.get('top1_within_1')}` / `{canon.get('no_effect_false_attribution_rate')}` vs `{action.get('responsibility_region_iou')}` / `{action.get('top1_within_1')}` / `{action.get('no_effect_false_attribution_rate')}`.
8. Final fixed decision: **{decision['decision']}**. Failed rules: `{decision.get('failed_rules', [])}`.

## Test leaderboard

{chr(10).join(table)}

All learned test metrics use operating points frozen on validation; no second Stage 2 tuning is performed.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "canonical_method": canonical_method, "seed_rows": len(seed_frame), "leaderboard_rows": len(board_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
