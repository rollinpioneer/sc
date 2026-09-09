"""Freeze validation-only HB2 model and comparator selection."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump
from recovery_handback.hb2.metrics import choose_length


def _root_equal(rows: list[dict], values: list[float]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values):
        grouped[str(row["stat_group_id"])].append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()])) if grouped else float("nan")


def _chosen_utility(records: list[dict], lambda_value: float, *, system_objective: bool = False) -> float:
    values = []
    for row in records:
        if system_objective:
            scores = [(float(row["p0"]), 0)] + [
                (float(row[f"p_sys_{length}"]) - lambda_value * length / 400.0, length)
                for length in (5, 20, 80)
            ]
            best = max(score for score, _ in scores)
            length = min(length for score, length in scores if best - score <= 1e-8)
        else:
            length = choose_length(row, lambda_value)
        if length == 0:
            outcome, cost = int(row["y0"]), 0
        else:
            truth = f"y_sys_{length}" if system_objective else f"y_genuine_{length}"
            outcome = int(row[truth])
            cost = int(row.get(f"helper_steps_l{length}", length))
        values.append(outcome - lambda_value * cost / 400.0)
    return _root_equal(records, values)


def _fixed_utility(records: list[dict], length: int, lambda_value: float) -> float:
    values = []
    for row in records:
        if length == 0:
            values.append(float(row["y0"]))
        else:
            cost = float(row.get(f"helper_steps_l{length}", length))
            values.append(float(row[f"y_genuine_{length}"]) - lambda_value * cost / 400.0)
    return _root_equal(records, values)


def _risk_utility(records: list[dict], length: int, threshold: float, lambda_value: float) -> float:
    values = []
    for row in records:
        selected = length if 1.0 - float(row["p0"]) >= threshold else 0
        if selected == 0:
            values.append(float(row["y0"]))
        else:
            cost = float(row.get(f"helper_steps_l{selected}", selected))
            values.append(float(row[f"y_genuine_{selected}"]) - lambda_value * cost / 400.0)
    return _root_equal(records, values)


def _load_seed0(validation_dir: Path) -> dict[str, dict]:
    payloads = {}
    for path in sorted(validation_dir.glob("M*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("seed", "seed0") != "seed0":
            continue
        model = payload.get("model")
        if model and model not in payloads:
            payloads[str(model)] = payload
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    primary_lambda = float(config["hb2"]["decision"]["primary_lambda"])
    payloads = _load_seed0(args.validation_dir)
    visual_ids = ("M2_global", "M3_local", "M4_paired", "M5_single")
    candidates = []
    for model in visual_ids:
        payload = payloads.get(model)
        if not payload:
            continue
        metrics = payload["metrics"]
        candidates.append({
            "model": model,
            "validation_utility": _chosen_utility(payload["records"], primary_lambda),
            "validation_system_objective_utility": _chosen_utility(payload["records"], primary_lambda, system_objective=True),
            "checkpoint": metrics["checkpoint"], "checkpoint_sha256": metrics["checkpoint_sha256"],
            "temperature": float(metrics["temperature"]),
            "parameters": int(metrics.get("parameters") or 10**18),
            "validation_nll": float(metrics["root_equal_five_head_nll_calibrated"]),
            "history_length": int(metrics.get("history_length", 4)),
        })
    if not candidates:
        raise SystemExit("no visual validation predictions found")
    candidates.sort(key=lambda item: (
        -np.nan_to_num(item["validation_utility"], nan=-np.inf),
        item["parameters"], item["validation_nll"], item["history_length"], item["model"],
    ))
    selected = candidates[0]

    reference_payload = payloads.get("M1_proprio") or payloads[selected["model"]]
    reference_records = reference_payload["records"]
    fixed_candidates = [{
        "method": "fixed_none" if length == 0 else f"fixed_l{length}",
        "length": length,
        "validation_utility": _fixed_utility(reference_records, length, primary_lambda),
    } for length in (0, 5, 20, 80)]
    fixed_candidates.sort(key=lambda item: (-item["validation_utility"], item["length"]))
    best_fixed = fixed_candidates[0]

    nonvisual_candidates = []
    for model in ("M0_time", "M1_proprio"):
        payload = payloads.get(model)
        if payload:
            nonvisual_candidates.append({
                "method": f"model_{model}", "model": model,
                "validation_utility": _chosen_utility(payload["records"], primary_lambda),
            })
    nonvisual_candidates.sort(key=lambda item: (-item["validation_utility"], item["model"]))
    best_nonvisual = nonvisual_candidates[0]

    risk_candidates = []
    for model in ("M0_time", "M1_proprio"):
        payload = payloads.get(model)
        if not payload:
            continue
        risk_candidates.append({
            "method": "risk_fixed", "mode": "always_none", "p0_model": model,
            "length": 0, "threshold": None,
            "validation_utility": _fixed_utility(payload["records"], 0, primary_lambda),
        })
        for length in (5, 20, 80):
            for threshold in config["hb2"]["decision"]["risk_thresholds"]:
                risk_candidates.append({
                    "method": "risk_fixed", "mode": "threshold", "p0_model": model, "length": length,
                    "threshold": float(threshold),
                    "validation_utility": _risk_utility(payload["records"], length, float(threshold), primary_lambda),
                })
    risk_candidates.sort(key=lambda item: (
        -item["validation_utility"], item["length"],
        -1.0 if item["threshold"] is None else item["threshold"], item["p0_model"],
    ))
    best_risk = risk_candidates[0]

    comparator_candidates = [
        {"category": "fixed", **best_fixed},
        {"category": "nonvisual", **best_nonvisual},
        {"category": "risk", **best_risk},
    ]
    comparator_candidates.sort(key=lambda item: (-item["validation_utility"], item["category"], item["method"]))
    best_comparator = comparator_candidates[0]
    output = {
        "schema_version": "hb2_selected_protocol_v2", "task": "square",
        "selected_visual_model": selected["model"], "canonical_seed": 0,
        "primary_lambda": primary_lambda,
        "lambda_curve": config["hb2"]["decision"]["lambda_curve"],
        "lengths": config["hb2"]["decision"]["lengths"],
        "include_full_in_selector": False,
        "visual_candidates": candidates, "canonical": selected,
        "validation_fixed_candidates": fixed_candidates, "validation_best_fixed": best_fixed,
        "validation_nonvisual_candidates": nonvisual_candidates,
        "validation_best_nonvisual": best_nonvisual,
        "validation_risk_candidates": risk_candidates, "validation_best_risk": best_risk,
        "validation_comparator_candidates": comparator_candidates,
        "validation_best_comparator": best_comparator["method"],
        "validation_best_comparator_spec": best_comparator,
        "validation_frozen": True,
    }
    atomic_json_dump(output, args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
