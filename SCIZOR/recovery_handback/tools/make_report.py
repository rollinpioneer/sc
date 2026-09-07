"""Generate the fixed-order HB1 evidence report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from recovery_handback.common import read_table


def _pct(value) -> str:
    try:
        numeric = float(value)
        return f"{100 * numeric:.1f}%" if math.isfinite(numeric) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _pct_ci(value, low, high) -> str:
    point = _pct(value)
    if point == "n/a" or _pct(low) == "n/a" or _pct(high) == "n/a":
        return point
    return f"{point} [{_pct(low)}, {_pct(high)}]"


def _number(value, digits: int = 1) -> str:
    try:
        numeric = float(value)
        return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads((args.assets_dir / "assets.json").read_text(encoding="utf-8"))
    coverage = json.loads((args.metrics_dir / "coverage.json").read_text(encoding="utf-8"))
    decision = json.loads((args.metrics_dir / "hb1_decision.json").read_text(encoding="utf-8"))
    curves = pd.read_csv(args.metrics_dir / "curves_by_task.csv")
    paired = pd.DataFrame(read_table(args.metrics_dir / "paired_results.parquet"))
    opportunities = pd.DataFrame(read_table(args.metrics_dir / "handoff_opportunities.parquet"))
    lines = [
        "# HB1 Report", "",
        "## 1. Fixed policy pair and task scope", "",
        f"Protocol: `{config['schema']}`. Tasks: {', '.join(config['tasks'])}. Horizon: {config['horizon_steps']} steps at the recorded runtime control frequency.", "",
    ]
    for task in config["tasks"]:
        pair_path = args.assets_dir / f"policy_pair_{task}.json"
        if pair_path.is_file():
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            base = pair["base"]
            repair = pair["repair"]
            if repair is None:
                lines.append(
                    f"- {task}: base `{base.get('algorithm')}` checkpoint `{base.get('checkpoint_sha256')}`; "
                    f"task status `{pair.get('status', 'NEED_BASE_POLICY')}`; no repair policy was trained or claimed."
                )
            else:
                selection = pair.get("qualification", {}).get("selection", {})
                repair_cost = selection.get("repair_evaluation_cost", {})
                training_summary_path = args.metrics_dir.parent / f"repair/{task}/train/training_summary.json"
                training_note = ""
                if training_summary_path.is_file():
                    training = json.loads(training_summary_path.read_text(encoding="utf-8"))
                    if training.get("resumed_from_checkpoint"):
                        training_note = (
                            f" resumed from `{training['resumed_from_checkpoint']}` after an external interruption;"
                            f" replay buffer resumed: `{training.get('replay_buffer_resumed')}`."
                        )
                lines.append(
                    f"- {task}: base `{base.get('algorithm')}` checkpoint `{base.get('checkpoint_sha256')}`; "
                    f"repair `{repair.get('algorithm')}` checkpoint `{repair.get('checkpoint_sha256')}`; "
                    f"privileged repair input: `{repair.get('uses_privileged_input')}`; "
                    f"repair training steps `{repair.get('training_steps')}`; qualification environment steps "
                    f"`{int(repair_cost.get('prefix_env_steps', 0)) + int(repair_cost.get('continuation_env_steps', 0))}`."
                    f"{training_note}"
                )
        else:
            lines.append(f"- {task}: policy pair is missing.")
    lines.extend(["", "Old scorer/responsibility checkpoints listed during asset resolution were not used as action policies.", "",
                  "## 2. Data roles", "",
                  "BC source is demonstration supervision. `base_val` selects only the base checkpoint; `repair_train` and `repair_val` train and select only the repairer; `pilot` is limited to engineering checks; `probe` is the exploratory paired feasibility set.", "",
                  "## 3. Coverage and natural scenarios", "",
                  f"Authoritative anchors: {coverage['authoritative_anchors']}; complete paired anchors: {coverage['complete_anchors']} ({_pct(coverage['coverage_fraction'])}); missing branches: {len(coverage['missing_keys'])}.", ""])
    if len(paired):
        roots = paired.drop_duplicates("anchor_id").groupby("task")["stat_group_id"].nunique().to_dict()
        anchor_times = paired.drop_duplicates("anchor_id").groupby(["task", "anchor_t"]).size().to_dict()
        lines.append(f"Independent root groups by task: `{json.dumps(roots, sort_keys=True)}`. Anchor-time counts: `{json.dumps({str(k): int(v) for k, v in anchor_times.items()}, sort_keys=True)}`.")
        baseline = paired[
            (paired["branch_name"] == "none")
            & paired["complete_pair"].fillna(False).astype(bool)
        ]
        baseline_counts = {}
        for task in config["tasks"]:
            task_baseline = baseline[baseline["task"] == task]
            baseline_counts[task] = {
                "successful": int(task_baseline["system_success"].fillna(False).astype(bool).sum()),
                "complete": int(len(task_baseline)),
            }
        lines.append(
            "No-help successful/complete paired anchors by task: "
            f"`{json.dumps(baseline_counts, sort_keys=True)}`. Missing branches are not counted as failures."
        )
    lines.extend(["", "## 4. Paired outcomes", ""])
    for task in config["tasks"]:
        rows = curves[curves["task"] == task]
        lines.append(f"### {task.capitalize()}")
        lines.append("")
        lines.append("| Branch | System success | Paired difference | Raw handoff | Genuine handoff | Helper-period success | Genuine rescue roots | Mean helper actions |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['branch_name']} | {_pct_ci(row.get('system_success_rate_root'), row.get('system_success_ci95_low'), row.get('system_success_ci95_high'))} | "
                f"{_pct_ci(row.get('paired_success_rate_difference_root'), row.get('paired_difference_ci95_low'), row.get('paired_difference_ci95_high'))} | "
                f"{_pct(row.get('handoff_success_raw_fraction'))} | "
                f"{_pct(row.get('genuine_handoff_success_fraction'))} | "
                f"{_pct(row.get('success_seen_under_helper_fraction'))} | "
                f"{int(row.get('genuine_rescue_roots', 0) or 0)} | "
                f"{_number(row.get('mean_helper_steps_actual'))} |"
            )
        lines.append("")
    lines.extend(["## 5. Helper completion versus handoff", "",
                  "`system_success`, `handoff_success_raw`, and `genuine_handoff_success` are retained separately. Genuine handoff requires no success under helper control, at least the configured autonomous delay, stable success, and zero repair calls after handoff.", ""])
    for task in config["tasks"]:
        finite = curves[(curves["task"] == task) & curves["branch_name"].isin(["l5", "l20", "l80"])]
        if finite.empty:
            continue
        waits = finite["first_success_wait_after_handoff_median"].dropna().to_dict()
        unnecessary = finite["mean_helper_steps_on_baseline_success"].dropna().to_dict()
        reentry = int(finite["repair_calls_after_handoff_total"].fillna(0).sum())
        lines.append(
            f"- {task}: helper-after-handoff calls `{reentry}`; median first-success waits by finite branch "
            f"`{json.dumps({str(finite.loc[index, 'branch_name']): float(value) for index, value in waits.items()}, sort_keys=True)}`; "
            f"mean helper actions on no-help-success anchors `{json.dumps({str(finite.loc[index, 'branch_name']): float(value) for index, value in unnecessary.items()}, sort_keys=True)}`."
        )
    lines.extend(["",
                  "## 6. Post-hoc shortest length and cost", ""])
    if len(opportunities):
        counts = opportunities.groupby(["task", opportunities["diagnostic_shortest"].astype(str)]).size().to_dict()
        saving = opportunities["conditional_helper_cost_saving"].dropna()
        lines.append(f"Shortest diagnostic categories: `{json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)}`. Conditional mean helper-action saving: `{float(saving.mean()) if len(saving) else None}` over `{len(saving)}` eligible anchors.")
    lines.extend(["", "## 7. Task decisions", "", f"Overall status: **{decision['status']}**.", ""])
    for task, result in decision["tasks"].items():
        lines.append(f"- {task}: `{result['status']}`; valid roots {result['valid_roots']}, failed roots {result['baseline_failed_roots']}, genuinely rescued roots {result['genuine_rescued_roots']}, control roots {result['control_roots']}.")
    lines.extend(["", "## 8. Blocking reason classification", "",
                  "Any hold is reported by its concrete task-level status: engineering coverage, base/repair capability, natural-failure count, or lack of local handoff evidence for this fixed pair and grid.", "",
                  "## 9. Limitations", "",
                  "One base seed and one repair seed per task; privileged repair observations; fixed 5/20/80 duration grid; current Can/Square task scope; exploratory probe rather than a final blind test; no new visual handoff model was trained.", "",
                  "## 10. HB-2 handoff", "",
                  f"Decision: `{decision['status']}`. Any later visual model must be compared against no-help and the strongest fixed-duration baseline, split by root scenario, and evaluated on new scenes not relabeled as a blind reuse of HB1 probe.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
