"""Generate the capability-repair and formal handback report for HB1-R."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from recovery_handback.tools.make_hb1_repair_decision import build_decision


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(summary: dict | None) -> dict:
    return (summary or {}).get("selection", {})


def _checkpoint(summary: dict | None) -> str:
    selected = (summary or {}).get("selected_repair") or (summary or {}).get("selected_candidate") or {}
    return str(selected.get("checkpoint", "not available"))


def _rate(successes: int, roots: int) -> str:
    if roots <= 0:
        return "not run"
    return f"{successes}/{roots} ({100.0 * successes / roots:.1f}%)"


def _formal_table(root: Path) -> list[str]:
    path = root / "metrics/curves_by_task.csv"
    if not path.is_file():
        return ["Formal paired curves have not been generated."]
    frame = pd.read_csv(path)
    lines = [
        "| Task | Branch | System success | Genuine rescue roots | Helper-completed anchors |",
        "|---|---|---:|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        rate = row.get("system_success_rate_root")
        rate_text = "n/a" if pd.isna(rate) else f"{100.0 * float(rate):.1f}%"
        lines.append(
            f"| {row.get('task')} | {row.get('branch_name')} | {rate_text} | "
            f"{int(row.get('genuine_rescue_roots', 0) or 0)} | "
            f"{int(row.get('helper_completed_anchors', 0) or 0)} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--old-hb1-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config) or {}
    root = args.repair_root.resolve()
    old_decision = _read(args.old_hb1_root / "metrics/hb1_decision.json") or {}
    direct = _read(root / "square/teacher/evaluation/qualification/summary.json")
    residual = _read(root / "square/repair_rl/qualification/summary.json")
    can_lowdim = _read(root / "can/diagnostic/evaluation/summary.json")
    can_visual = _read(root / "can/visual/evaluation/summary.json")
    distill = _read(root / "can/distill/rollouts/summary.json")
    formal = _read(root / "metrics/hb1_decision.json")
    coverage = _read(root / "metrics/coverage.json") or {}
    decision = build_decision(root)

    direct_selection = _selected(direct)
    residual_selection = _selected(residual)
    lowdim_selection = _selected(can_lowdim)
    visual_selection = _selected(can_visual)
    square_final_rescues = max(
        int(direct_selection.get("base_failed_repair_succeeded_roots", 0) or 0),
        int(residual_selection.get("base_failed_repair_succeeded_roots", 0) or 0),
    )
    formal_tasks = (formal or {}).get("tasks", {})
    completed_tasks = [
        task for task, item in formal_tasks.items()
        if int(item.get("complete_anchors", 0) or 0) > 0
    ]

    lines = [
        "# HB1-R Capability Repair Report",
        "",
        "## Scope",
        "",
        f"Protocol: `{config.get('schema', 'unknown')}`. This report separates policy capability repair from the formal 0/5/20/80/full handback experiment.",
        "",
        "## Preserved HB1 Result",
        "",
        f"The preserved HB1 decision was `{old_decision.get('status', 'unknown')}`. It is historical evidence and was not overwritten.",
        "",
        "## Square Capability Repair",
        "",
        f"- Direct privileged teacher status: `{(direct or {}).get('status', 'NOT_RUN')}`.",
        f"- Direct teacher checkpoint: `{_checkpoint(direct)}`.",
        f"- Direct teacher qualification: {_rate(int(direct_selection.get('repair_successes', 0) or 0), int(direct_selection.get('anchors', 0) or 0))}; rescued independent roots: `{int(direct_selection.get('base_failed_repair_succeeded_roots', 0) or 0)}`.",
        f"- Teacher-centered residual curriculum executed: `{'yes' if residual else 'no'}`.",
        f"- Residual qualification status: `{(residual or {}).get('status', 'NOT_RUN')}`; rescued independent roots: `{int(residual_selection.get('base_failed_repair_succeeded_roots', 0) or 0)}`.",
        f"- Final Square rescued independent roots: `{square_final_rescues}`.",
        f"- Frozen Square pair: `{(root / 'assets/policy_pair_square.json').is_file()}`.",
        "",
        "The Square teacher and any teacher-centered residual repairer use privileged object-state input and are feasibility instruments, not deployable visual repair policies.",
        "",
        "## Can Capability Diagnosis",
        "",
        f"- Low-dimensional privileged teacher status: `{(can_lowdim or {}).get('status', 'NOT_RUN')}`.",
        f"- Low-dimensional base_val result: {_rate(int(lowdim_selection.get('successes', 0) or 0), int(lowdim_selection.get('roots', 0) or 0))}.",
        f"- Successful runtime teacher rollouts: `{int((distill or {}).get('successes', 0) or 0)}` of `{int((distill or {}).get('roots', 0) or 0)}` attempted.",
        f"- Visual student status: `{(can_visual or {}).get('status', 'NOT_RUN')}`.",
        f"- Visual student base_val result: {_rate(int(visual_selection.get('successes', 0) or 0), int(visual_selection.get('roots', 0) or 0))}.",
        f"- Frozen Can base policy: `{(root / 'assets/base_policy_can.json').is_file()}`.",
        "",
        "The Can low-dimensional teacher uses privileged object state. The visual student excludes object state and uses only the configured cameras and proprioception.",
        "",
        "## Formal HB1 Paired Experiment",
        "",
        f"Tasks with completed formal paired anchors: `{completed_tasks}`.",
        f"Coverage: `{int(coverage.get('complete_anchors', 0) or 0)}` complete anchors of `{int(coverage.get('authoritative_anchors', 0) or 0)}`; fraction `{float(coverage.get('coverage_fraction', 0.0) or 0.0):.3f}`.",
        "",
    ]
    lines.extend(_formal_table(root))
    lines.extend([
        "",
        "System success, genuine handoff rescue, and helper-completed outcomes are reported separately. A full-helper success is not counted as genuine handoff evidence.",
        "",
        "## Decision",
        "",
        f"Final HB1-R status: **{decision['status']}**.",
        "",
        f"- Square: `{decision['tasks']['square']['status']}`.",
        f"- Can: `{decision['tasks']['can']['status']}`.",
        f"- Tasks cleared for HB-2: `{decision['ready_tasks']}`.",
        "",
        "HB-2 should start only for tasks meeting the formal independent-root, natural-failure, genuine-rescue, control-root, and complete-coverage gates.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
