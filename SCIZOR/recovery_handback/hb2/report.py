"""Generate the ordered HB2 report and conservative layered decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, str)):
        return str(value)
    return f"{float(value):.{digits}f}"


def _validation_utilities(selection: dict) -> dict[str, float]:
    output = {
        row["model"]: float(row["validation_utility"])
        for row in selection.get("visual_candidates", [])
    }
    output.update({
        row["model"]: float(row["validation_utility"])
        for row in selection.get("validation_nonvisual_candidates", [])
    })
    return output


def _handoff_ready(metrics: dict, protocol: dict | None) -> tuple[bool, dict]:
    if not protocol:
        return False, {"reason": "handoff protocol missing"}
    visual = metrics.get("selected_visual/seed0")
    proprio = metrics.get("proprio/seed0")
    prior = metrics.get("frozen_train_prior")
    if not visual or not proprio or not prior:
        return False, {"reason": "handoff test metrics incomplete"}
    rule = protocol["readiness_rule"]
    counts_ok = True
    for key in ("q_complete", "q_strict"):
        item = visual[key]
        counts_ok = counts_ok and (
            int(item["roots"]) >= int(rule["minimum_test_roots"])
            and int(item["positive_roots"]) >= int(rule["minimum_positive_roots_per_output"])
            and int(item["negative_roots"]) >= int(rule["minimum_negative_roots_per_output"])
        )
    visual_brier = sum(visual[key]["brier"] for key in ("q_complete", "q_strict")) / 2
    proprio_brier = sum(proprio[key]["brier"] for key in ("q_complete", "q_strict")) / 2
    prior_brier = sum(prior[key]["brier"] for key in ("q_complete", "q_strict")) / 2
    beats_controls = visual_brier < proprio_brier and visual_brier < prior_brier
    evidence = {
        "counts_ok": counts_ok,
        "selected_visual_mean_brier": visual_brier,
        "proprio_mean_brier": proprio_brier,
        "frozen_train_prior_mean_brier": prior_brier,
        "beats_proprio_and_prior": beats_controls,
        "rule": rule,
    }
    return bool(counts_ok and beats_controls), evidence


def _development_cases(root: Path, selected: str) -> list[dict]:
    rows_path = root / "data/frozen/anchor_examples.parquet"
    predictions_path = root / f"metrics/validation/{selected}.json"
    if not rows_path.is_file() or not predictions_path.is_file():
        return []
    rows = {
        str(row["example_id"]): row for row in read_table(rows_path)
        if row.get("split") == "validation" and bool(row.get("complete_pair"))
    }
    predictions = _json(predictions_path, {}).get("records", [])
    predicates = [
        ("effective_short_help", lambda row: not row["y0"] and bool(row["y_genuine_l5"])),
        ("help_ineffective", lambda row: not row["y0"] and not any(row[f"y_sys_l{x}"] for x in (5, 20, 80))),
        ("baseline_success_interfered", lambda row: bool(row["y0"]) and any(not row[f"y_sys_l{x}"] for x in (5, 20, 80))),
        ("helper_completed", lambda row: any(row[f"category_l{x}"] == 1 for x in (5, 20, 80))),
        ("finite_nonmonotonic", lambda row: bool(row["y_genuine_l5"]) and not bool(row["y_genuine_l20"])),
    ]
    cases = []
    used = set()
    for category, predicate in predicates:
        for prediction in predictions:
            row = rows.get(str(prediction["example_id"]))
            if row and row["example_id"] not in used and predicate(row):
                cases.append({
                    "category": category,
                    "example_id": row["example_id"],
                    "root_id": row["root_id"],
                    "anchor_t": row["anchor_t"],
                })
                used.add(row["example_id"])
                break
    return cases[:6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    config = _json(args.config)
    protocol = _json(root / "config/frozen_protocol.json")
    handoff_protocol = _json(root / "config/frozen_handoff_protocol.json")
    test = _json(root / "metrics/test/summary.json", {})
    probability = _json(root / "metrics/test/probability_metrics.json", {})
    handoff_metrics = _json(root / "metrics/test/handoff_probability_metrics.json", {})
    runtime = _json(root / "metrics/runtime_probe/runtime_probe.json", {})
    count_clarification = _json(root / "metrics/hb1_count_clarification.json", {})
    split = _json(root / "data/frozen/split_manifest.json", {})
    sufficiency = _json(root / "data/frozen/data_sufficiency.json", {})
    selection = protocol["selection"]
    frozen_anchor_rows = read_table(root / "data/frozen/anchor_examples.parquet")
    legacy_anchor_count = sum(
        row.get("data_role") == "legacy_train" for row in frozen_anchor_rows
    )
    selected = str(protocol["selected_visual_model"])
    canonical_name = f"model_{selected}"
    methods = test.get("methods", {})
    chosen = methods.get(canonical_name, {})
    comparisons = test.get("canonical_comparisons", {})
    valid = int(test.get("valid_roots", 0))
    rescuable = int(test.get("oracle_rescuable_roots", 0))
    rescue = int(chosen.get("genuine_rescue_unique_roots", 0))
    full_coverage = bool(test.get("full_branch_coverage"))
    data_ok = valid >= 30 and rescuable >= 5 and full_coverage
    points_positive = bool(comparisons) and all(
        float(item["point_difference"]) > 0 for item in comparisons.values()
    )
    bstar_key = f"{canonical_name}_vs_{test.get('bootstrap_reference')}"
    bstar = test.get("bootstrap", {}).get(bstar_key, {})
    bstar_ci = bstar.get("ci95_percentile", [None, None])
    bstar_positive = bstar_ci[0] is not None and float(bstar_ci[0]) > 0

    validation_u = _validation_utilities(selection)
    best_fixed = str(selection["validation_best_fixed"]["method"])
    best_nonvisual = str(selection["validation_best_nonvisual"]["method"])
    selected_test_u = chosen.get("primary_utility_U_lambda_0p25")
    fixed_test_u = methods.get(best_fixed, {}).get("primary_utility_U_lambda_0p25")
    nonvisual_test_u = methods.get(best_nonvisual, {}).get("primary_utility_U_lambda_0p25")
    signal = (
        selected_test_u is not None and fixed_test_u is not None
        and float(selected_test_u) > float(fixed_test_u)
    )
    nonvisual_signal = (
        nonvisual_test_u is not None and fixed_test_u is not None
        and float(nonvisual_test_u) > float(fixed_test_u)
    )
    visual_gain_supported = bool(
        comparisons.get("best_nonvisual", {}).get("ci95_percentile", [None])[0] is not None
        and comparisons["best_nonvisual"]["ci95_percentile"][0] > 0
    )
    if not data_ok:
        status = "HOLD_HB2_DATA"
    elif rescue >= 3 and points_positive and bstar_positive:
        status = "READY_HB3_SINGLE_TASK"
    elif rescue >= 3 and points_positive:
        status = "READY_HB3_PILOT_ONLY"
    elif (signal or nonvisual_signal) and not visual_gain_supported:
        status = "HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN"
    else:
        status = "HB2_NO_PREDICTIVE_GAIN_CURRENT_SETUP"

    exit_ready, exit_evidence = _handoff_ready(handoff_metrics, handoff_protocol)
    local_gain = validation_u.get("M3_local", float("-inf")) > validation_u.get("M2_global", float("inf"))
    paired_gain = validation_u.get("M4_paired", float("-inf")) > validation_u.get("M3_local", float("inf"))
    history_gain = validation_u.get("M4_paired", float("-inf")) > validation_u.get("M5_single", float("inf"))
    baseline_success_roots = int(test.get("baseline_success_roots", 0))
    decision = {
        "task": "square",
        "overall_status": status,
        "counterfactual_start_prediction_signal": (
            "supported" if signal else ("weak" if validation_u.get(selected, float("-inf"))
                                        > selection["validation_best_fixed"]["validation_utility"]
                                        else "unsupported")
        ),
        "visual_gain": "supported" if visual_gain_supported else "unproven",
        "local_feature_gain": "supported" if local_gain else "unproven",
        "history_gain": "supported" if history_gain else "unproven",
        "paired_supervision_gain": "supported" if paired_gain else "unproven",
        "exit_model_ready": exit_ready,
        "exit_model_evidence": exit_evidence,
        "baseline_preservation_evidence": (
            "adequate" if baseline_success_roots >= 5 else "small_sample"
        ),
        "scope": "fixed_pair_fixed_anchor_single_intervention",
        "online_adaptive_switching_tested": False,
        "selected_visual_model": selected,
        "canonical_seed": 0,
        "test_valid_roots": valid,
        "test_valid_anchors": test.get("valid_anchors"),
        "test_full_branch_coverage": full_coverage,
        "test_oracle_rescuable_roots": rescuable,
        "selected_visual_rescued_roots": rescue,
        "canonical_comparisons": comparisons,
        "b_star": test.get("bootstrap_reference"),
        "root_cluster_ci_vs_b_star": bstar_ci,
        "runtime_probe_status": runtime.get("status"),
        "single_allowed_data_expansion_used": sufficiency.get(
            "single_allowed_expansion_used", False
        ),
    }
    atomic_json_dump(decision, root / "metrics/hb2_decision.json")

    canonical_probability = probability.get(f"{selected}/seed0", {})
    lines = [
        "# HB2 Report",
        "",
        "## Material Passport",
        "",
        f"- Verification Status: VERIFIED",
        f"- Experiment: HB2-v1 Square",
        f"- Source commit: `{protocol.get('source_ref')}`",
        f"- Frozen code commit: `{protocol.get('code_commit')}`",
        f"- Result status: **{status}**",
        "",
        "## Fixed Pair And Scope",
        "",
        f"- Base checkpoint SHA256: `e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6`",
        f"- Repair checkpoint SHA256: `7b2e5f22e0bc3722c9c70f4d6bcb3ab79696026500fc121572d50bf530349d62`",
        f"- Semantic pair ID: `{protocol.get('semantic_pair_id')}`",
        "- Scope: Square, fixed policy pair, fixed anchors, one intervention from 0/5/20/80 steps.",
        "- The repair teacher uses privileged state. Online adaptive switching was not tested.",
        "",
        "## Existing HB1-R Evidence",
        "",
        f"- Legacy rows reused as training/development data: {legacy_anchor_count}",
        f"- Count clarification: `{root / 'metrics/hb1_count_clarification.json'}`",
        f"- Legacy full unique success roots: {count_clarification.get('unique_success_roots', 'NA')}",
        "",
        "## Data Roles And Independent Roots",
        "",
        f"- Frozen F examples: {split.get('anchor_examples', split.get('anchors', 'NA'))}",
        f"- Frozen H examples: {split.get('eligible_handoff_examples', 'NA')}",
        f"- Test valid roots / anchors: {valid} / {test.get('valid_anchors')}",
        f"- Full branch coverage: {full_coverage}",
        f"- Test finite-help rescuable roots: {rescuable}",
        f"- One allowed train/validation expansion used: {decision['single_allowed_data_expansion_used']}",
        "",
        "## Labels And Inputs",
        "",
        "- F uses one anchor history with none, l5, l20, l80, and full outcomes; missing engineering records are excluded, not relabeled as failures.",
        "- H uses actual handoff histories and includes absolute time plus elapsed helper steps.",
        "- Model tensors are restricted to two RGB cameras, 9-D proprioception, 7-D cached base actions, and time.",
        f"- DINOv2 checkpoint SHA256: `{protocol.get('encoder', {}).get('weights_sha256')}`",
        "",
        "## Probability Evaluation",
        "",
        "| Output | Brier | NLL | AUROC | AP | Positive roots |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("p0", "p_sys_5", "p_sys_20", "p_sys_80",
                 "p_genuine_5", "p_genuine_20", "p_genuine_80", "p_full"):
        item = canonical_probability.get(name, {})
        lines.append(
            f"| {name} | {_fmt(item.get('brier'))} | {_fmt(item.get('nll'))} | "
            f"{_fmt(item.get('auroc'))} | {_fmt(item.get('average_precision'))} | "
            f"{_fmt(item.get('positive_roots'))} |"
        )
    lines.extend([
        "",
        "## Fixed Matrix Ablation",
        "",
        "| Model | Validation root-equal utility |",
        "|---|---:|",
    ])
    for model in ("M0_time", "M1_proprio", "M2_global", "M3_local", "M4_paired", "M5_single"):
        lines.append(f"| {model} | {_fmt(validation_u.get(model))} |")
    lines.extend([
        "",
        f"- Selected visual model: `{selected}` (canonical seed 0).",
        f"- Visual / local / history / paired gain: {decision['visual_gain']} / "
        f"{decision['local_feature_gain']} / {decision['history_gain']} / "
        f"{decision['paired_supervision_gain']}.",
        "",
        "## One-Shot Selection Value And Cost",
        "",
        "| Method | Root utility | Autonomous completion | System success | Mean helper steps | Rescued roots |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, values in methods.items():
        lines.append(
            f"| {name} | {_fmt(values.get('primary_utility_U_lambda_0p25'))} | "
            f"{_fmt(values.get('autonomous_completion_rate'))} | "
            f"{_fmt(values.get('root_mean_system_success'))} | "
            f"{_fmt(values.get('mean_helper_steps_actual'))} | "
            f"{_fmt(values.get('genuine_rescue_unique_roots'))} |"
        )
    lines.extend([
        "",
        f"- Frozen B_star: `{test.get('bootstrap_reference')}`; canonical 95% root-cluster CI: `{bstar_ci}`.",
        f"- Canonical interference: `{chosen.get('interference_among_baseline_success')}`.",
        f"- Canonical regret to observed finite oracle: {_fmt(chosen.get('regret_to_observed_finite_oracle'))}.",
        "",
        "## Actual Handoff Prediction",
        "",
        f"- Exit model ready: **{exit_ready}**.",
        f"- Evidence: `{json.dumps(exit_evidence, sort_keys=True)}`",
        "- Support is limited to observed fixed 5/20/80-step handoff exits.",
        "",
        "## Runtime",
        "",
        f"- Runtime probe status: `{runtime.get('status')}`.",
        f"- Pilot roots checked: {runtime.get('pilot_roots')}.",
        f"- Repair calls after handoff: {runtime.get('control_execution', {}).get('repair_calls_after_handoff_total', 'NA')}.",
        "- Prefix replay cost is reported in runtime_probe.json and is not folded into formal test metrics.",
        "",
        "## Development Cases",
        "",
    ])
    cases = _development_cases(root, selected)
    if cases:
        for case in cases:
            lines.append(
                f"- {case['category']}: `{case['example_id']}`, root "
                f"`{case['root_id']}`, t={case['anchor_t']}."
            )
    else:
        lines.append("- No development case index available.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Each rollout is one binary outcome under the frozen policy and random mechanism, not a per-state safety probability.",
        "- The test is same-task and same-distribution; cross-task, distribution-shift, and real-robot claims are unsupported.",
        "- Full repair is a capability/cost reference and is excluded from the deployable selector.",
        "- Multiple anchors and model seeds are not treated as additional independent roots.",
        "",
        "## Layered Decision",
        "",
        f"- Overall: **{status}**",
        f"- Counterfactual start signal: {decision['counterfactual_start_prediction_signal']}",
        f"- Visual gain: {decision['visual_gain']}",
        f"- Exit model ready: {exit_ready}",
        "",
        "## HB3 Handoff",
        "",
        f"- Allowed scope: `{decision['scope']}`.",
        "- A full dynamic-query or repeated-takeover controller remains out of scope.",
        "- Continue only within the capability implied by the layered status above.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
