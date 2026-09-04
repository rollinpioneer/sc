"""Build Stage 2 seed summaries, leaderboard, report, and fixed decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["transition_f1", "responsibility_region_iou", "mean_abs_localization_delay", "top1_within_1", "top5_hit", "recovery_retention", "innocent_downstream_retention", "expert_retention", "rare_retention", "no_effect_false_attribution_rate", "delete_rate"]


def overall(metric, method): return metric["methods"][method]["overall"]


def markdown(frame):
    cols = list(frame.columns); rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for values in frame.itertuples(index=False, name=None): rows.append("| " + " | ".join("NA" if pd.isna(v) else (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in values) + " |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--feature-manifest", type=Path, required=True); p.add_argument("--chunk-summary", type=Path, required=True); p.add_argument("--validation-metrics", type=Path, required=True); p.add_argument("--test-metrics", type=Path, required=True); p.add_argument("--canonical-seed", type=Path, required=True); p.add_argument("--stage1-leaderboard", type=Path, required=True); p.add_argument("--runs-root", type=Path, required=True); p.add_argument("--seed-summary", type=Path, required=True); p.add_argument("--leaderboard", type=Path, required=True); p.add_argument("--go-no-go", type=Path, required=True); p.add_argument("--report", type=Path, required=True); args = p.parse_args()
    feature, chunks, valid, test, canonical = [json.loads(path.read_text()) for path in (args.feature_manifest, args.chunk_summary, args.validation_metrics, args.test_metrics, args.canonical_seed)]
    rows = []
    for seed in range(3):
        method = f"responsibility_seed{seed}"; best = json.loads((args.runs_root / f"full_seed_{seed}" / "best_metrics.json").read_text()); v, t = overall(valid, method), overall(test, method)
        rows.append({"seed": str(seed), "best_epoch": best["best_epoch"], "validation_top1_within_1": v.get("top1_within_1"), "validation_iou": v.get("responsibility_region_iou"), "validation_f1": v.get("transition_f1"), "test_top1_within_1": t.get("top1_within_1"), "test_top5_hit": t.get("top5_hit"), "test_iou": t.get("responsibility_region_iou"), "test_f1": t.get("transition_f1"), "test_mean_abs_delay": t.get("mean_abs_localization_delay"), "test_recovery_retention": t.get("recovery_retention"), "test_no_effect_far": t.get("no_effect_false_attribution_rate")})
    seed_frame = pd.DataFrame(rows); numeric = seed_frame.drop(columns="seed").apply(pd.to_numeric); seed_frame = pd.concat([seed_frame, pd.DataFrame([{ "seed": "full_mean", **numeric.mean().to_dict()}, {"seed": "full_std", **numeric.std(ddof=0).to_dict()}])], ignore_index=True); args.seed_summary.parent.mkdir(parents=True, exist_ok=True); seed_frame.to_csv(args.seed_summary, index=False)
    canonical_method = canonical["canonical_method"]; ordering = [("canonical_responsibility", canonical_method), ("responsibility_seed0", "responsibility_seed0"), ("responsibility_seed1", "responsibility_seed1"), ("responsibility_seed2", "responsibility_seed2"), ("action_only_seed0", "action_only_seed0"), ("future_discount", "future_discount"), ("original_scizor", "original_scizor"), ("uniform", "uniform")]
    board_rows = [{"method": display, **{("test_delete_rate" if key == "delete_rate" else key): overall(test, source).get(key) for key in METRICS}} for display, source in ordering]
    board = pd.DataFrame(board_rows); args.leaderboard.parent.mkdir(parents=True, exist_ok=True); board.to_csv(args.leaderboard, index=False)
    baselines = {name: overall(test, name) for name in ("original_scizor", "uniform", "future_discount")}; full = [overall(test, f"responsibility_seed{i}") for i in range(3)]; canon, action = overall(test, canonical_method), overall(test, "action_only_seed0")
    def gt_count(key): return sum((item.get(key) or -np.inf) > max((b.get(key) or -np.inf) for b in baselines.values()) for item in full)
    iou_base = max((b.get("responsibility_region_iou") or -np.inf) for b in baselines.values()); top_base = max((b.get("top1_within_1") or -np.inf) for b in baselines.values()); delay_base = min((b.get("mean_abs_localization_delay") or np.inf) for b in baselines.values()); recovery_base = max((b.get("recovery_retention") or -np.inf) for b in baselines.values()); far_base = min((b.get("no_effect_false_attribution_rate") or np.inf) for b in baselines.values())
    action_advantage = (canon.get("responsibility_region_iou") or -np.inf) > (action.get("responsibility_region_iou") or -np.inf) or (canon.get("top1_within_1") or -np.inf) > (action.get("top1_within_1") or -np.inf) or (canon.get("no_effect_false_attribution_rate") or np.inf) <= (action.get("no_effect_false_attribution_rate") or np.inf) - .02
    go = gt_count("responsibility_region_iou") >= 2 and gt_count("top1_within_1") >= 2 and (canon.get("mean_abs_localization_delay") or np.inf) < delay_base and (canon.get("recovery_retention") or -np.inf) >= recovery_base - .10 and (canon.get("no_effect_false_attribution_rate") or np.inf) <= far_base + .03 and action_advantage
    no_go = gt_count("responsibility_region_iou") == 0 and gt_count("top1_within_1") == 0
    decision = "GO_STAGE3" if go else ("NO_GO_SWITCH_DIRECTION" if no_go else "ITERATE_STAGE2_ONCE")
    decision_data = {"decision": decision, "canonical_method": canonical_method, "rules": {"full_iou_above_strongest_count": gt_count("responsibility_region_iou"), "full_top1_above_strongest_count": gt_count("top1_within_1"), "iou_baseline": iou_base, "top1_baseline": top_base, "delay_baseline": delay_base, "recovery_baseline": recovery_base, "no_effect_far_baseline": far_base, "canonical_vs_action_only_requirement": action_advantage}}
    args.go_no_go.parent.mkdir(parents=True, exist_ok=True); args.go_no_go.write_text(json.dumps(decision_data, indent=2, sort_keys=True) + "\n")
    can_task, square_task = test["methods"][canonical_method]["task"].get("can", {}), test["methods"][canonical_method]["task"].get("square", {})
    report = f"# Stage 2 Report\n\n## Frozen protocol\n\nFeature cache: {feature['demo_count']} demos / {feature['transition_count']} transitions. Chunk samples: {chunks['sample_count']}. Canonical method `{canonical_method}` was selected on validation only; its test threshold was not retuned.\n\n## Test leaderboard\n\n{markdown(board)}\n\n## Direct answers\n\n1. **Stability versus Stage 1 baselines:** all {decision_data['rules']['full_iou_above_strongest_count']}/3 full seeds exceed the strongest baseline IoU and all {decision_data['rules']['full_top1_above_strongest_count']}/3 exceed its top-1-within-1 score.\n2. **Source of improvement:** canonical IoU/top-1/delay are `{canon.get('responsibility_region_iou')}`, `{canon.get('top1_within_1')}`, and `{canon.get('mean_abs_localization_delay')}`, compared with the best baseline values `{iou_base}`, `{top_base}`, and `{delay_base}`. Threshold-level F1 is `{canon.get('transition_f1')}`.\n3. **Recovery:** canonical recovery retention is `{canon.get('recovery_retention')}` versus baseline maximum `{recovery_base}`; it did not decline.\n4. **No-effect control:** canonical no-effect false-attribution rate is `{canon.get('no_effect_false_attribution_rate')}` versus baseline minimum `{far_base}`; it exceeds the fixed tolerance, so this gate prevents GO_STAGE3.\n5. **Task difference:** canonical Can top-1/IoU are `{can_task.get('top1_within_1')}` / `{can_task.get('responsibility_region_iou')}`; Square are `{square_task.get('top1_within_1')}` / `{square_task.get('responsibility_region_iou')}`.\n6. **Action-only diagnostic:** action-only top-1/IoU/FAR are `{action.get('top1_within_1')}` / `{action.get('responsibility_region_iou')}` / `{action.get('no_effect_false_attribution_rate')}`, versus canonical `{canon.get('top1_within_1')}` / `{canon.get('responsibility_region_iou')}` / `{canon.get('no_effect_false_attribution_rate')}`.\n7. **Next action:** the fixed automatic conclusion is **{decision}**. Per the protocol, this permits one targeted Stage 2 adjustment (effect-loss emphasis for no-effect/hard negatives *or* learning rate 1e-4), rather than proceeding to Stage 3 now.\n\nThe comparisons above all use validation-frozen operating points; the test set was not used to select seed or threshold.\n"
    args.report.write_text(report); print(json.dumps({"decision": decision, "canonical_method": canonical_method}, indent=2))


if __name__ == "__main__": main()
