"""Evaluate frozen, label-free HB2 predictions on new root groups."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, write_table
from recovery_handback.hb2.metrics import choose_length, root_probability_metrics


FINITE_LENGTHS = (5, 20, 80)


def _load_prediction_files(path: Path) -> tuple[dict[tuple[str, str], list[dict]], dict[tuple[str, str], list[dict]]]:
    anchors, handoffs = {}, {}
    for file in sorted(path.glob("*.json")):
        if file.name == "summary.json":
            continue
        payload = json.loads(file.read_text(encoding="utf-8"))
        key = (str(payload["model"]), str(payload["seed"]))
        target = handoffs if payload.get("head") == "handoff" else anchors
        if key in target:
            raise ValueError(f"duplicate prediction payload for {key}")
        target[key] = list(payload.get("records", []))
    return anchors, handoffs


def _root_mean(values: list[dict], key: str) -> float | None:
    grouped = defaultdict(list)
    for row in values:
        grouped[str(row["stat_group_id"])].append(float(row[key]))
    if not grouped:
        return None
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def _model_length(prediction: dict, lambda_value: float, decision: dict, *, system: bool = False) -> int:
    if not system:
        return choose_length(
            prediction,
            lambda_value,
            denominator=float(decision["cost_denominator"]),
            tolerance=float(decision["tie_tolerance"]),
        )
    scores = [(float(prediction["p0"]), 0)]
    for length in FINITE_LENGTHS:
        scores.append((
            float(prediction[f"p_sys_{length}"])
            - lambda_value * length / float(decision["cost_denominator"]),
            length,
        ))
    best = max(score for score, _ in scores)
    return min(
        length for score, length in scores
        if best - score <= float(decision["tie_tolerance"])
    )


def _oracle_length(row: dict, lambda_value: float, denominator: float) -> int:
    scores = [(float(row["y0"]), 0)]
    for length in FINITE_LENGTHS:
        scores.append((
            float(row[f"y_genuine_l{length}"])
            - lambda_value * float(row[f"helper_steps_l{length}"]) / denominator,
            length,
        ))
    best = max(score for score, _ in scores)
    return min(length for score, length in scores if best - score <= 1e-12)


def _selected_length(
    row: dict,
    prediction: dict,
    spec: dict,
    lambda_value: float,
    decision: dict,
) -> int:
    kind = spec["kind"]
    if kind == "fixed":
        return int(spec["length"])
    if kind == "oracle":
        return _oracle_length(row, lambda_value, float(decision["cost_denominator"]))
    if kind == "model":
        return _model_length(prediction, lambda_value, decision, system=bool(spec.get("system_objective")))
    if kind == "risk":
        if spec.get("mode") == "always_none" or int(spec["length"]) == 0:
            return 0
        return int(spec["length"]) if 1.0 - float(prediction["p0"]) >= float(spec["threshold"]) else 0
    raise ValueError(f"unknown method kind: {kind}")


def _method_rows(
    rows: list[dict],
    predictions: list[dict],
    spec: dict,
    lambda_value: float,
    decision: dict,
) -> list[dict]:
    prediction_by_id = {str(row["example_id"]): row for row in predictions}
    output = []
    for row in rows:
        prediction = prediction_by_id.get(str(row["example_id"]))
        if prediction is None:
            raise KeyError(f"prediction missing for {row['example_id']}")
        length = _selected_length(row, prediction, spec, lambda_value, decision)
        if length == 0:
            autonomous = system = int(row["y0"])
            cost = changed = repair_after = helper_completed = 0
        else:
            autonomous = int(row[f"y_genuine_l{length}"])
            system = int(row[f"y_sys_l{length}"])
            cost = int(row[f"helper_steps_l{length}"])
            changed = int(row.get(f"changed_action_steps_l{length}", cost))
            repair_after = int(row.get(f"repair_calls_after_handoff_l{length}", 0))
            helper_completed = int(
                row.get(f"helper_completed_l{length}", int(row[f"category_l{length}"]) == 1)
            )
        longer_rescue = int(
            length in FINITE_LENGTHS
            and not autonomous
            and any(
                candidate > length and bool(row[f"y_genuine_l{candidate}"])
                for candidate in FINITE_LENGTHS
            )
        )
        full_rescue = int(length in FINITE_LENGTHS and not system and bool(row["y_full"]))
        output.append({
            "example_id": str(row["example_id"]),
            "stat_group_id": str(row["stat_group_id"]),
            "root_id": str(row["root_id"]),
            "anchor_t": int(row["anchor_t"]),
            "selected_length": length,
            "y_autonomous": autonomous,
            "y_system": system,
            "y0": int(row["y0"]),
            "helper_steps_actual": cost,
            "changed_action_steps": changed,
            "repair_calls_after_handoff": repair_after,
            "helper_completed": helper_completed,
            "utility": autonomous - lambda_value * cost / float(decision["cost_denominator"]),
            "interference": int(bool(row["y0"]) and length > 0 and not bool(system)),
            "interference_eligible": int(bool(row["y0"]) and length > 0),
            "rescue": int(not bool(row["y0"]) and bool(autonomous)),
            "premature_vs_longer_finite": longer_rescue,
            "premature_vs_full": full_rescue,
        })
    return output


def _aggregate(values: list[dict], oracle: list[dict] | None = None) -> dict:
    valid_roots = len({row["stat_group_id"] for row in values})
    interference_count = int(sum(row["interference"] for row in values))
    interference_denominator = int(sum(row["interference_eligible"] for row in values))
    result = {
        "valid_roots": valid_roots,
        "valid_anchors": len(values),
        "system_success_rate_anchor": float(np.mean([row["y_system"] for row in values])) if values else None,
        "root_mean_system_success": _root_mean(values, "y_system"),
        "autonomous_completion_rate": _root_mean(values, "y_autonomous"),
        "genuine_rescue_anchors": int(sum(row["rescue"] for row in values)),
        "genuine_rescue_unique_roots": len({
            row["stat_group_id"] for row in values if row["rescue"]
        }),
        "helper_completed_anchors": int(sum(row["helper_completed"] for row in values)),
        "interference_among_baseline_success": {
            "count": interference_count,
            "denominator": interference_denominator,
            "rate": (interference_count / interference_denominator
                     if interference_denominator else None),
        },
        "mean_helper_steps_actual": _root_mean(values, "helper_steps_actual"),
        "mean_changed_action_steps": _root_mean(values, "changed_action_steps"),
        "helped_fraction": _root_mean([
            {**row, "helped": int(row["selected_length"] > 0)} for row in values
        ], "helped"),
        "primary_utility_U_lambda_0p25": _root_mean(values, "utility"),
        "repair_calls_after_handoff": {
            "total": int(sum(row["repair_calls_after_handoff"] for row in values)),
            "root_mean": _root_mean(values, "repair_calls_after_handoff"),
        },
        "premature_return_diagnostic": {
            "selected_finite_failure_with_longer_finite_genuine": int(
                sum(row["premature_vs_longer_finite"] for row in values)
            ),
            "selected_finite_failure_with_full_success": int(
                sum(row["premature_vs_full"] for row in values)
            ),
        },
    }
    if oracle is not None:
        oracle_by_id = {row["example_id"]: row for row in oracle}
        regrets = [{
            "stat_group_id": row["stat_group_id"],
            "regret": oracle_by_id[row["example_id"]]["utility"] - row["utility"],
        } for row in values]
        result["regret_to_observed_finite_oracle"] = _root_mean(regrets, "regret")
    return result


def _method_root_utilities(values: list[dict]) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in values:
        grouped[str(row["stat_group_id"])].append(float(row["utility"]))
    return {root: float(np.mean(group)) for root, group in grouped.items()}


def _bootstrap(
    method_values: dict[str, list[dict]], reference: str, repeats: int, seed: int
) -> dict:
    roots = sorted(_method_root_utilities(method_values[reference]))
    root_values = {name: _method_root_utilities(values) for name, values in method_values.items()}
    if any(set(values) != set(roots) for values in root_values.values()):
        raise ValueError("bootstrap methods do not share the same root groups")
    rng = np.random.default_rng(seed)
    differences = {name: [] for name in method_values if name != reference}
    for _ in range(repeats):
        sampled = rng.choice(roots, size=len(roots), replace=True)
        reference_mean = float(np.mean([root_values[reference][root] for root in sampled]))
        for name, values in root_values.items():
            if name != reference:
                differences[name].append(
                    float(np.mean([values[root] for root in sampled])) - reference_mean
                )
    result = {}
    reference_point = float(np.mean(list(root_values[reference].values())))
    for name, samples in differences.items():
        array = np.asarray(samples, np.float64)
        point = float(np.mean(list(root_values[name].values()))) - reference_point
        result[f"{name}_vs_{reference}"] = {
            "point_difference": point,
            "bootstrap_mean_difference": float(array.mean()),
            "ci95_percentile": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
            "unit": "stat_group_id",
            "repeats": repeats,
        }
    return result


def _probability_metrics(
    rows: list[dict], predictions: dict[tuple[str, str], list[dict]]
) -> dict:
    truth_by_id = {str(row["example_id"]): row for row in rows}
    targets = {
        "p0": "y0", "p_full": "y_full",
        "p_sys_5": "y_sys_l5", "p_sys_20": "y_sys_l20", "p_sys_80": "y_sys_l80",
        "p_genuine_5": "y_genuine_l5", "p_genuine_20": "y_genuine_l20",
        "p_genuine_80": "y_genuine_l80",
    }
    output = {}
    for (model, seed), records in predictions.items():
        joined = []
        for prediction in records:
            truth = truth_by_id[str(prediction["example_id"])]
            joined.append({
                **prediction,
                **{truth_key: int(truth[truth_key]) for truth_key in targets.values()},
            })
        output[f"{model}/{seed}"] = {
            probability: root_probability_metrics(joined, truth, probability)
            for probability, truth in targets.items()
        }
    return output


def _handoff_metrics(
    rows: list[dict], predictions: dict[tuple[str, str], list[dict]], protocol: dict | None
) -> dict:
    truth_by_id = {str(row["example_id"]): row for row in rows}
    output = {}
    for (model, seed), records in predictions.items():
        joined = []
        for prediction in records:
            truth = truth_by_id[str(prediction["example_id"])]
            joined.append({
                **prediction,
                "y_complete": int(truth["y_complete_after_handoff"]),
                "y_strict": int(truth["y_strict_genuine"]),
            })
        output[f"{model}/{seed}"] = {
            "eligible_rows": len(joined),
            "valid_roots": len({row["stat_group_id"] for row in joined}),
            "q_complete": root_probability_metrics(joined, "y_complete", "q_complete"),
            "q_strict": root_probability_metrics(joined, "y_strict", "q_strict"),
        }
    if protocol and rows:
        priors = protocol["training_root_equal_priors"]
        prior_rows = [{
            "stat_group_id": str(row["stat_group_id"]),
            "y_complete": int(row["y_complete_after_handoff"]),
            "y_strict": int(row["y_strict_genuine"]),
            "q_complete": float(priors["q_complete"]),
            "q_strict": float(priors["q_strict"]),
        } for row in rows]
        output["frozen_train_prior"] = {
            "eligible_rows": len(prior_rows),
            "valid_roots": len({row["stat_group_id"] for row in prior_rows}),
            "q_complete": root_probability_metrics(prior_rows, "y_complete", "q_complete"),
            "q_strict": root_probability_metrics(prior_rows, "y_strict", "q_strict"),
        }
    return output


def _per_anchor_time(name: str, values: list[dict]) -> list[dict]:
    output = []
    for anchor_t in sorted({row["anchor_t"] for row in values}):
        subset = [row for row in values if row["anchor_t"] == anchor_t]
        output.append({
            "method": name,
            "anchor_t": anchor_t,
            "anchors": len(subset),
            "roots": len({row["stat_group_id"] for row in subset}),
            "root_mean_system_success": _root_mean(subset, "y_system"),
            "autonomous_completion_rate": _root_mean(subset, "y_autonomous"),
            "mean_helper_steps_actual": _root_mean(subset, "helper_steps_actual"),
            "utility": _root_mean(subset, "utility"),
        })
    return output


def _full_reference(rows: list[dict]) -> dict:
    values = [{
        "stat_group_id": str(row["stat_group_id"]),
        "system": int(row["y_full"]),
        "cost": int(row["helper_steps_full"]),
        "changed": int(row.get("changed_action_steps_full", row["helper_steps_full"])),
    } for row in rows]
    return {
        "system_success_rate_anchor": float(np.mean([row["system"] for row in values])),
        "root_mean_system_success": _root_mean(values, "system"),
        "mean_helper_steps_actual": _root_mean(values, "cost"),
        "mean_changed_action_steps": _root_mean(values, "changed"),
        "unique_success_roots": len({row["stat_group_id"] for row in values if row["system"]}),
        "selector_candidate": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--handoff-protocol", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    decision = protocol["decision"]
    lambda_value = float(decision["primary_lambda"])
    all_test_rows = [
        row for row in read_table(args.dataset / "anchor_examples.parquet")
        if row.get("split") == "test"
    ]
    rows = [
        row for row in all_test_rows
        if row.get("split") == "test" and bool(row.get("complete_pair"))
    ]
    anchor_predictions, handoff_predictions = _load_prediction_files(args.predictions)
    selected = str(protocol["selected_visual_model"])
    canonical_key = (selected, "seed0")
    if canonical_key not in anchor_predictions:
        raise SystemExit("canonical selected prediction missing")

    method_specs = {
        "fixed_none": {"kind": "fixed", "length": 0},
        "fixed_l5": {"kind": "fixed", "length": 5},
        "fixed_l20": {"kind": "fixed", "length": 20},
        "fixed_l80": {"kind": "fixed", "length": 80},
        "observed_finite_oracle": {"kind": "oracle"},
    }
    selection = protocol["selection"]
    risk = selection["validation_best_risk"]
    method_specs["risk_fixed"] = {"kind": "risk", **risk}
    for model_id in ("M0_time", "M1_proprio"):
        if (model_id, "seed0") in anchor_predictions:
            method_specs[f"model_{model_id}"] = {
                "kind": "model", "model": model_id, "seed": "seed0"
            }
    for seed_name in sorted(protocol["models"][selected]):
        suffix = "" if seed_name == "seed0" else f"_{seed_name}"
        method_specs[f"model_{selected}{suffix}"] = {
            "kind": "model", "model": selected, "seed": seed_name
        }
    method_specs[f"model_{selected}_system_objective"] = {
        "kind": "model", "model": selected, "seed": "seed0", "system_objective": True
    }

    def predictions_for(spec: dict) -> list[dict]:
        if spec["kind"] in ("fixed", "oracle"):
            return anchor_predictions[canonical_key]
        if spec["kind"] == "risk":
            return anchor_predictions[(str(spec["p0_model"]), "seed0")]
        return anchor_predictions[(str(spec["model"]), str(spec["seed"]))]

    method_values = {
        name: _method_rows(rows, predictions_for(spec), spec, lambda_value, decision)
        for name, spec in method_specs.items()
    }
    oracle_values = method_values["observed_finite_oracle"]
    aggregates = {
        name: _aggregate(values, None if name == "observed_finite_oracle" else oracle_values)
        for name, values in method_values.items()
    }
    reference = str(selection["validation_best_comparator"])
    if reference not in method_values:
        raise ValueError(f"frozen B_star is unavailable: {reference}")
    bootstrap = _bootstrap(
        method_values,
        reference,
        int(protocol["statistics"]["bootstrap_repeats"]),
        int(protocol["statistics"]["seed"]),
    )
    canonical_method = f"model_{selected}"
    comparison_methods = {
        "best_fixed": str(selection["validation_best_fixed"]["method"]),
        "best_nonvisual": str(selection["validation_best_nonvisual"]["method"]),
        "risk": "risk_fixed",
    }
    canonical_comparisons = {}
    for label, baseline in comparison_methods.items():
        paired = _bootstrap(
            {canonical_method: method_values[canonical_method], baseline: method_values[baseline]},
            baseline,
            int(protocol["statistics"]["bootstrap_repeats"]),
            int(protocol["statistics"]["seed"]),
        )
        canonical_comparisons[label] = paired[f"{canonical_method}_vs_{baseline}"]

    probability = _probability_metrics(rows, anchor_predictions)
    handoff_rows = [
        row for row in read_table(args.dataset / "handoff_examples.parquet")
        if row.get("split") == "test" and bool(row.get("eligible"))
    ]
    handoff_protocol = (
        json.loads(args.handoff_protocol.read_text(encoding="utf-8"))
        if args.handoff_protocol and args.handoff_protocol.is_file() else None
    )
    handoff_probability = _handoff_metrics(
        handoff_rows, handoff_predictions, handoff_protocol
    )
    test_roots = len({row["stat_group_id"] for row in rows})
    rescuable = len({
        row["stat_group_id"] for row in rows
        if not bool(row["y0"]) and any(bool(row[f"y_genuine_l{length}"]) for length in FINITE_LENGTHS)
    })
    baseline_success_roots = len({
        row["stat_group_id"] for row in rows if bool(row["y0"])
    })
    payload = {
        "schema_version": "hb2_test_metrics_v2",
        "selected_visual_model": selected,
        "canonical_seed": "seed0",
        "valid_roots": test_roots,
        "valid_anchors": len(rows),
        "full_branch_coverage": bool(all_test_rows) and all(
            bool(row.get("complete_pair")) for row in all_test_rows
        ),
        "authoritative_anchors": len(all_test_rows),
        "oracle_rescuable_roots": rescuable,
        "baseline_success_roots": baseline_success_roots,
        "methods": aggregates,
        "method_specs": method_specs,
        "full_reference": _full_reference(rows),
        "bootstrap": bootstrap,
        "bootstrap_reference": reference,
        "bootstrap_reference_spec": selection["validation_best_comparator_spec"],
        "canonical_comparisons": canonical_comparisons,
        "probability_metrics_path": str((args.output_dir / "probability_metrics.json").resolve()),
        "handoff_probability_metrics_path": str((args.output_dir / "handoff_probability_metrics.json").resolve()),
        "handoff_readiness_rule": handoff_protocol.get("readiness_rule") if handoff_protocol else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(payload, args.output_dir / "summary.json")
    atomic_json_dump(probability, args.output_dir / "probability_metrics.json")
    atomic_json_dump(handoff_probability, args.output_dir / "handoff_probability_metrics.json")

    method_table = [{"method": name, **metrics} for name, metrics in aggregates.items()]
    write_table(method_table, args.output_dir / "methods.csv")
    write_table(
        [row for name, values in method_values.items() for row in _per_anchor_time(name, values)],
        args.output_dir / "by_anchor_t.csv",
    )
    lambda_rows = []
    for curve_lambda in decision["lambda_curve"]:
        for name, spec in method_specs.items():
            values = _method_rows(
                rows, predictions_for(spec), spec, float(curve_lambda), decision
            )
            lambda_rows.append({
                "method": name,
                "lambda": float(curve_lambda),
                "root_equal_utility": _root_mean(values, "utility"),
                "autonomous_completion_rate": _root_mean(values, "y_autonomous"),
                "mean_helper_steps_actual": _root_mean(values, "helper_steps_actual"),
            })
    write_table(lambda_rows, args.output_dir / "lambda_curve.csv")
    write_table(
        [{**row, "method": name} for name, values in method_values.items() for row in values],
        args.output_dir / "method_rows.parquet",
    )
    print(json.dumps({
        "valid_roots": test_roots,
        "valid_anchors": len(rows),
        "selected": selected,
        "reference": reference,
        "rescuable_roots": rescuable,
    }, indent=2))


if __name__ == "__main__":
    main()
