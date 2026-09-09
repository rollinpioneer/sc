"""Select the validation visual model and freeze the one-shot decision specification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump
from recovery_handback.hb2.metrics import choose_length


def _utility(records: list[dict], lambda_value: float) -> float:
    values = []
    for row in records:
        length = choose_length(row, lambda_value)
        if length == 0: y = int(row["y0"]); cost = 0
        else: y = int(row[f"y_genuine_{length}"]); cost = int(row.get(f"helper_steps_l{length}", length))
        values.append(y - lambda_value * cost / 400.0)
    return float(np.mean(values)) if values else float("nan")


def _fixed_utility(records: list[dict], length: int, lambda_value: float) -> float:
    values = []
    for row in records:
        if length == 0:
            values.append(float(row["y0"]))
        else:
            cost = float(row.get(f"helper_steps_l{length}", length))
            values.append(float(row[f"y_genuine_{length}"]) - lambda_value * cost / 400.0)
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = []
    for path in sorted(args.validation_dir.glob("M*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") not in ("M2_global", "M3_local", "M4_paired", "M5_single"):
            continue
        records = payload["records"]
        utility = _utility(records, float(config["hb2"]["decision"]["primary_lambda"]))
        candidates.append({"model": payload["model"], "validation_utility": utility,
                           "checkpoint": payload["metrics"].get("checkpoint"), "temperature": payload["metrics"].get("temperature", 1.0)})
    if not candidates: raise SystemExit("no visual validation predictions found")
    candidates.sort(key=lambda item: (-np.nan_to_num(item["validation_utility"], nan=-np.inf), item["model"]))
    selected = candidates[0]
    reference_records = json.loads((args.validation_dir / "M1_proprio.json").read_text(encoding="utf-8"))["records"] if (args.validation_dir / "M1_proprio.json").is_file() else json.loads((args.validation_dir / f"{selected['model']}.json").read_text(encoding="utf-8"))["records"]
    baseline_scores = {f"fixed_l{length}" if length else "fixed_none": _fixed_utility(reference_records, length, float(config["hb2"]["decision"]["primary_lambda"])) for length in (0, 5, 20, 80)}
    validation_best_comparator = max(baseline_scores, key=lambda key: (baseline_scores[key], -int(key.removeprefix("fixed_l") or 0)))
    payload = {
        "schema_version": "hb2_selected_protocol_v1",
        "task": "square", "selected_visual_model": selected["model"],
        "canonical_seed": 0, "primary_lambda": config["hb2"]["decision"]["primary_lambda"],
        "lambda_curve": config["hb2"]["decision"]["lambda_curve"],
        "lengths": config["hb2"]["decision"]["lengths"], "include_full_in_selector": False,
        "visual_candidates": candidates, "canonical": selected,
        "validation_baselines": baseline_scores, "validation_best_comparator": validation_best_comparator,
        "nonvisual_comparators": ["M0_time", "M1_proprio"],
        "risk_thresholds": config["hb2"]["decision"]["risk_thresholds"],
        "validation_frozen": True,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
