"""Generate the ordered, scope-limited HB3-P research report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_file


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _cases(episodes_path: Path) -> list[dict]:
    if not episodes_path.is_file():
        return []
    rows = read_table(episodes_path)
    by_method = {}
    for row in rows:
        by_method[(str(row["method_id"]), str(row["stat_group_id"]))] = row
    roots = sorted({str(row["stat_group_id"]) for row in rows})
    predicates = (
        ("effective_help", lambda n, m, s: not n["system_success"] and m["genuine_handoff_success"]),
        ("baseline_interference", lambda n, m, s: n["system_success"] and not m["system_success"]),
        ("state_time_choice_difference", lambda n, m, s: m["selected_length"] != s["selected_length"] or m["takeover_t"] != s["takeover_t"]),
        ("waited_to_later_candidate", lambda n, m, s: m["takeover_t"] in (80, 160)),
    )
    output, used = [], set()
    for category, predicate in predicates:
        for root in roots:
            if root in used:
                continue
            none = by_method[("NONE", root)]
            m1 = by_method[("M1_GRID", root)]
            scheduled = by_method[("S_STAR", root)]
            if predicate(none, m1, scheduled):
                output.append({
                    "category": category, "stat_group_id": root,
                    "root_id": m1["root_id"], "M1_takeover_t": m1["takeover_t"],
                    "M1_length": m1["selected_length"],
                })
                used.add(root)
                break
    return output[:4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    config = _json(args.config)
    protocol = _json(root / "config/frozen_protocol.json")
    development = _json(root / "config/protocol.pre_freeze.json")
    coverage = _json(root / "metrics/test/coverage.json", {})
    comparisons = _json(root / "metrics/test/paired_comparisons.json", {})
    decision = _json(root / "metrics/hb3p_decision.json", {})
    pilot = _json(root / "metrics/pilot/summary.json", {})
    ranking = _json(root / "metrics/clarification/ranking_metrics.json", {})
    methods = _csv(root / "metrics/test/methods.csv")
    interference = {row["method_id"]: row for row in _csv(root / "metrics/test/interference.csv")}
    costs = {row["method_id"]: row for row in _csv(root / "metrics/test/costs.csv")}
    method_by_id = {row["method_id"]: row for row in methods}
    if (
        not coverage.get("complete")
        or int(coverage.get("preregistered_roots", -1)) != 80
        or int(coverage.get("complete_logical_records", -1)) != 480
        or coverage.get("missing_records")
        or coverage.get("engineering_failures")
    ):
        raise RuntimeError("refusing to generate a VERIFIED report from incomplete test coverage")
    expected_methods = set(protocol["methods"])
    if set(method_by_id) != expected_methods or set(interference) != expected_methods or set(costs) != expected_methods:
        raise RuntimeError("report inputs do not cover the complete frozen method set")
    primary = comparisons["M1_GRID_minus_S_STAR"]["utility"]
    cases = _cases(root / "metrics/test/episodes.parquet")
    atomic_json_dump({"schema_version": "hb3p_report_cases_v1", "cases": cases}, root / "report/development_cases.json")

    base = protocol["models"]
    pair = _json(root / "assets/policy_pair_square.json")
    lines = [
        "# HB3-P Report",
        "",
        "## Material Passport",
        "",
        "- Verification Status: VERIFIED",
        "- Experiment: HB3-P-v1 Square",
        f"- Source commit: `{protocol['source_ref']}`",
        f"- Frozen code commit: `{protocol['code_commit']}`",
        f"- Result status: **{decision['hb3p_status']}**",
        f"- Frozen protocol SHA256: `{sha256_file(root / 'config/frozen_protocol.json')}`",
        "",
        "## Fixed Pair And Scope",
        "",
        f"- Base checkpoint SHA256: `{pair['base']['checkpoint_sha256']}`",
        f"- Repair checkpoint SHA256: `{pair['repair']['checkpoint_sha256']}`",
        f"- Semantic pair ID: `{protocol['semantic_pair_id']}`",
        "- Scope: fixed candidate times 20/80/160, at most one takeover, fixed 5/20/80-step exit, permanent handback.",
        "- Preserved HB2 status: `HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN`.",
        "- No action policy, teacher, DINO backbone, F model, or H model was retrained.",
        "",
        "## Episode-Level Change From HB2",
        "",
        "HB2 averaged decisions over separately supplied legal anchors. HB3-P executes one complete trajectory from t=0 per method and root; after the first nonzero decision it never queries F again. Therefore HB2 anchor-level utilities are historical context and are not subtracted from these episode-level results.",
        "",
        "## Development And Frozen Rules",
        "",
        f"- Development roots / legal anchors / complete branches: 40 / 119 / 595.",
        f"- Selected fixed schedule from old validation: `{development['selected_fixed_schedule']['candidate_id']}` with validation U={_fmt(development['selected_fixed_schedule']['validation_root_mean_utility'])}.",
        f"- Compiled M0 schedule: `{json.dumps(protocol['m0_compiled_schedule'], sort_keys=True)}`.",
        f"- Method aliases: `{json.dumps(protocol['method_aliases'], sort_keys=True)}`.",
        f"- Primary method / comparator: `{protocol['primary_method']}` / `{protocol['primary_comparator']}`.",
        "",
        "## Pilot And Online Semantics",
        "",
        f"- Pilot roots / complete records: {pilot.get('pilot_roots')} / {pilot.get('complete_records')}.",
        f"- All input, baseline, fixed-branch control-semantic, and selection parity checks passed: {pilot.get('all_checks_passed')}.",
        f"- Pilot repair calls after handback: {pilot.get('repair_calls_after_handoff')}.",
        f"- Mean pilot episode wall time: {_fmt(pilot.get('mean_episode_wall_seconds'))} seconds.",
        f"- Mean live M4 inference+IPC time per pilot episode: {_fmt(pilot.get('mean_m4_inference_wall_seconds'))} seconds.",
        "",
        "## New Test Coverage",
        "",
        f"- Preregistered independent roots: {coverage.get('preregistered_roots')} (seeds 500000-500079).",
        f"- Complete logical root-method records: {coverage.get('complete_logical_records')} / {coverage.get('expected_logical_records')}.",
        f"- Actual unique rollouts including baseline: {coverage.get('actual_unique_rollouts')}.",
        f"- Total executed environment steps across unique rollouts: {coverage.get('total_env_steps_actual')}.",
        f"- Missing records / engineering failures: {len(coverage.get('missing_records', []))} / {len(coverage.get('engineering_failures', []))}.",
        "",
        "## Complete-Trajectory Results",
        "",
        "| Method | System success | Autonomous completion | U (lambda=0.25) | Mean helper steps | Genuine rescued roots |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in protocol["methods"]:
        row = method_by_id[method]
        lines.append(
            f"| {method} | {_fmt(row['system_success_rate'])} ({row['system_success_count']}/80) | "
            f"{_fmt(row['autonomous_completion_rate'])} ({row['autonomous_completion_count']}/80) | "
            f"{_fmt(row['primary_utility_U_lambda_0p25'])} | {_fmt(row['mean_helper_steps_actual'])} | "
            f"{row['genuine_rescued_roots']} |"
        )
    lines.extend([
        "",
        "## Paired Comparisons",
        "",
        f"- Primary M1_GRID - S_STAR delta U: {_fmt(primary['point_difference'], 6)}; 95% paired-root percentile CI [{_fmt(primary['ci95_percentile'][0], 6)}, {_fmt(primary['ci95_percentile'][1], 6)}], n={primary['root_count']}, 2000 resamples, seed 20260909.",
    ])
    for key in ("M1_GRID_minus_NONE", "M1_GRID_minus_M0_GRID", "M4_GRID_minus_S_STAR", "M4_GRID_minus_M1_GRID", "S_STAR_minus_NONE"):
        item = comparisons[key]["utility"]
        lines.append(
            f"- Exploratory {key}: delta U={_fmt(item['point_difference'], 6)}, "
            f"CI=[{_fmt(item['ci95_percentile'][0], 6)}, {_fmt(item['ci95_percentile'][1], 6)}]."
        )
    lines.extend([
        "",
        "## Interference And Cost",
        "",
        "| Method | Interference / all baseline-success | Interference / helped baseline-success | Mean takeover | Mean queries | Repair calls after handback |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in protocol["methods"]:
        irow = interference[method]
        crow = costs[method]
        lines.append(
            f"| {method} | {irow['count']}/{irow['all_baseline_success_roots']} ({_fmt(irow['all_baseline_success_rate'])}) | "
            f"{irow['count']}/{irow['helped_baseline_success_roots']} ({_fmt(irow['helped_baseline_success_rate'])}) | "
            f"{_fmt(crow['mean_takeover_count'])} | {_fmt(crow['mean_query_count'])} | {crow['repair_calls_after_handoff_total']} |"
        )
    lines.extend([
        "",
        "## Predictor Inputs And Latency",
        "",
        "- F received only the two current camera streams, 9-D proprioception, the current base suggestion history, and normalized absolute time. Labels, reward, privileged object state, future actions, root seed encodings, and future images were excluded.",
        "- The base policy was advanced exactly once on every executed environment step, including helper-controlled steps. M4 used a synchronous local file queue; its backbone, head, IPC, and total wall times are reported in `metrics/test/costs.csv`.",
        "- Simulation waits for inference and therefore does not claim a real-robot 20 Hz latency result.",
        "",
        "## HB2 Ranking Supplement",
        "",
        f"- Ranking supplement artifact: `metrics/clarification/ranking_metrics.json` ({len(ranking.get('models', ranking)) if isinstance(ranking, dict) else 'available'} top-level entries).",
        "- AUROC/AP were recomputed from existing predictions only. This did not change HB2 model selection, temperatures, thresholds, utility, or the preserved HB2 status.",
        "- HB2 local/history/paired gains remain validation-set single-seed point estimates, not independent significance results.",
        "",
        "## Layered Decision",
        "",
        f"- HB3-P status: **{decision['hb3p_status']}**.",
        f"- State-entry gain: {decision['state_entry_gain']}.",
        f"- Time-schedule gain: {decision['time_schedule_gain']}.",
        f"- Visual increment: {decision['visual_increment']} (pre-registered secondary comparison only).",
        f"- Baseline preservation evidence: {decision['baseline_preservation_evidence']}.",
        "",
        "## Not Tested",
        "",
        "Learned H-controlled exit, repeated takeover, arbitrary query times, Can, full-help capability on the new roots, policy improvement, distribution shift, and real-robot execution were not tested. The result does not reclassify HB2 visual evidence or claim complete HB-3.",
        "",
        "## Next Allowed Scope",
        "",
        "Any continuation must use a new frozen protocol. If mechanism value is supported, the next bounded question is stop-versus-continue at a fixed entry after 5/20 helper steps; current H probabilities are not a validated stopping controller.",
        "",
        "## Selected Cases",
        "",
    ])
    if cases:
        for case in cases:
            lines.append(f"- `{case['category']}`: `{case['root_id']}`, M1 t={case['M1_takeover_t']}, L={case['M1_length']}.")
    else:
        lines.append("- No case matched the four predeclared descriptive categories; no extra roots were collected.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": decision["hb3p_status"], "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
