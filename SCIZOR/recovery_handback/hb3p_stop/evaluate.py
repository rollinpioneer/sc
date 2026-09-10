"""Evaluate paired root-level pilot outcomes and non-inferiority diagnostics."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.hb3p.metrics import paired_bootstrap
from recovery_handback.hb3p_stop.io import atomic_json_dump, read_jsonl, write_csv


METHODS = ("NONE", "FIXED_L60", "FIXED_L80", "LEARNED_STOP_CONTINUE")


def _fourfold(left: dict[str, bool], right: dict[str, bool]) -> dict:
    counts = Counter((left[root], right[root]) for root in left)
    return {"both_success": counts[(True, True)], "left_only": counts[(True, False)], "right_only": counts[(False, True)], "both_failure": counts[(False, False)]}


def evaluate(episode_path: Path, coverage_path: Path, protocol: dict, output_dir: Path) -> dict:
    rows = read_jsonl(Path(episode_path))
    coverage = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    if not coverage.get("complete"):
        raise RuntimeError("cannot evaluate incomplete coverage")
    grouped = defaultdict(dict)
    for row in rows:
        grouped[str(row["method_id"])][str(row["stat_group_id"])] = row
    if set(grouped) != set(METHODS) or any(len(grouped[method]) != int(protocol["new_test_roots"]) for method in METHODS):
        raise RuntimeError("method/root coverage mismatch")
    summaries, method_rows = [], []
    for method in METHODS:
        values = list(grouped[method].values())
        summary = {
            "method_id": method, "roots": len(values),
            "system_success_count": int(sum(bool(row["system_success"]) for row in values)),
            "system_success_rate": float(np.mean([bool(row["system_success"]) for row in values])),
            "autonomous_completion_count": int(sum(bool(row["autonomous_completion"]) for row in values)),
            "autonomous_completion_rate": float(np.mean([bool(row["autonomous_completion"]) for row in values])),
            "mean_utility": float(np.mean([float(row["utility"]) for row in values])),
            "mean_helper_steps": float(np.mean([int(row["helper_steps_actual"]) for row in values])),
            "mean_inference_seconds": float(np.mean([float(row.get("inference_wall_seconds", 0.0)) for row in values])),
            "takeover_count": int(sum(int(row["takeover_count"]) for row in values)),
        }
        if method == "LEARNED_STOP_CONTINUE":
            summary["stop_decisions"] = int(sum(row.get("stop_continue_decision") == "STOP" for row in values))
            summary["continue_decisions"] = int(sum(row.get("stop_continue_decision") == "CONTINUE" for row in values))
        summaries.append(summary)
    comparisons = {}
    for right in ("FIXED_L60", "FIXED_L80", "NONE"):
        left = "LEARNED_STOP_CONTINUE"
        name = f"{left}_minus_{right}"
        utility = paired_bootstrap(
            {root: float(grouped[left][root]["utility"]) for root in grouped[left]},
            {root: float(grouped[right][root]["utility"]) for root in grouped[right]},
            int(protocol["statistics"]["bootstrap_repeats"]), int(protocol["statistics"]["seed"]),
        )
        system = paired_bootstrap(
            {root: float(bool(grouped[left][root]["system_success"])) for root in grouped[left]},
            {root: float(bool(grouped[right][root]["system_success"])) for root in grouped[right]},
            int(protocol["statistics"]["bootstrap_repeats"]), int(protocol["statistics"]["seed"]),
        )
        autonomous = paired_bootstrap(
            {root: float(bool(grouped[left][root]["autonomous_completion"])) for root in grouped[left]},
            {root: float(bool(grouped[right][root]["autonomous_completion"])) for root in grouped[right]},
            int(protocol["statistics"]["bootstrap_repeats"]), int(protocol["statistics"]["seed"]),
        )
        cost = paired_bootstrap(
            {root: float(grouped[left][root]["helper_steps_actual"]) for root in grouped[left]},
            {root: float(grouped[right][root]["helper_steps_actual"]) for root in grouped[right]},
            int(protocol["statistics"]["bootstrap_repeats"]), int(protocol["statistics"]["seed"]),
        )
        comparisons[name] = {"utility": utility, "system_success": system, "autonomous_completion": autonomous, "helper_steps": cost,
                             "system_fourfold": _fourfold({root: bool(grouped[left][root]["system_success"]) for root in grouped[left]}, {root: bool(grouped[right][root]["system_success"]) for root in grouped[right]}),
                             "autonomous_fourfold": _fourfold({root: bool(grouped[left][root]["autonomous_completion"]) for root in grouped[left]}, {root: bool(grouped[right][root]["autonomous_completion"]) for root in grouped[right]})}
    comparator = str(protocol["noninferiority_comparator"])
    primary = comparisons[f"LEARNED_STOP_CONTINUE_minus_{comparator}"]
    margin = float(protocol["noninferiority_margin_absolute"])
    pilot = bool(protocol.get("pilot", True))
    formal_claim_allowed = bool(protocol.get("formal_claim_allowed", not pilot))
    lower_system = float(primary["system_success"]["ci95_percentile"][0])
    lower_autonomous = float(primary["autonomous_completion"]["ci95_percentile"][0])
    margin = float(protocol["noninferiority_margin_absolute"])
    system_pass = bool(lower_system >= -margin)
    autonomous_pass = bool(lower_autonomous >= -margin)
    if pilot:
        status = "PILOT_ONLY"
        reason = "frozen protocol is explicitly a pilot; no formal non-inferiority claim is allowed"
    elif formal_claim_allowed and system_pass and autonomous_pass:
        status = "FORMAL_NONINFERIORITY_PASS"
        reason = "the preregistered root-level lower confidence bounds meet the absolute margin"
    else:
        status = "FORMAL_NONINFERIORITY_NOT_SHOWN"
        reason = "at least one preregistered root-level lower confidence bound is below the absolute margin"
    decision = {
        "schema_version": "hb3p_stop_continue_decision_v1",
        "status": status,
        "formal_claim_allowed": formal_claim_allowed,
        "pilot": pilot,
        "reason": reason,
        "noninferiority_comparator": comparator,
        "noninferiority_margin_absolute": margin,
        "learned_vs_comparator_system_success_lower_ci": lower_system,
        "learned_vs_comparator_autonomous_lower_ci": lower_autonomous,
        "system_success_noninferiority_diagnostic": system_pass,
        "autonomous_noninferiority_diagnostic": autonomous_pass,
        "interpretation": (
            "diagnostic only; do not report as a proof"
            if pilot else "formal root-level comparison under the frozen protocol"
        ),
    }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({"schema_version": "hb3p_stop_continue_summary_v1", "methods": summaries, "comparisons": comparisons}, output_dir / "summary.json")
    atomic_json_dump(decision, output_dir / "decision.json")
    rows_out = []
    for method in METHODS:
        row = next(item for item in summaries if item["method_id"] == method)
        rows_out.append(row)
    write_csv(rows_out, output_dir / "methods.csv")
    return {"methods": summaries, "comparisons": comparisons, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(args.episodes, args.coverage, protocol, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
