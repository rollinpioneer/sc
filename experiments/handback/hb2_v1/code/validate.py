"""Root-equal HB2 validation, shared-temperature calibration, and H metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recovery_handback.common import atomic_json_dump, read_table, sha256_file
from recovery_handback.hb2.dataset import AnchorDataset, HandoffDataset
from recovery_handback.hb2.metrics import choose_length, finite_probs, root_equal_mean, root_probability_metrics
from recovery_handback.hb2.models import build_model


def _inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    names = ("global", "local", "proprio", "base_actions", "time", "helper_elapsed")
    return {key: value.to(device) for key, value in batch.items() if key in names}


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = values.max(axis=1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _binary_nll(logits: np.ndarray, truth: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, np.float64)
    truth = np.asarray(truth, np.float64)
    return np.maximum(logits, 0.0) - logits * truth + np.log1p(np.exp(-np.abs(logits)))


def _five_head_root_nll(logits: np.ndarray, rows: list[dict], temperature: float) -> float:
    scaled = np.asarray(logits, np.float64) / float(temperature)
    row_indices = np.arange(len(rows))
    terms = [
        _binary_nll(scaled[:, 0], np.asarray([row["y0"] for row in rows])),
        -_log_softmax(scaled[:, 1:5])[row_indices, [int(row["category_l5"]) for row in rows]],
        -_log_softmax(scaled[:, 5:9])[row_indices, [int(row["category_l20"]) for row in rows]],
        -_log_softmax(scaled[:, 9:13])[row_indices, [int(row["category_l80"]) for row in rows]],
        _binary_nll(scaled[:, 13], np.asarray([row["y_full"] for row in rows])),
    ]
    per_example = np.stack(terms, axis=1).mean(axis=1)
    value = root_equal_mean(per_example, [str(row["stat_group_id"]) for row in rows])
    return float(value) if value is not None else float("inf")


def _multiclass_root_nll(logits: np.ndarray, rows: list[dict], temperature: float) -> float:
    logp = _log_softmax(np.asarray(logits, np.float64) / float(temperature))
    target = np.asarray([int(row["handoff_category"]) for row in rows])
    losses = -logp[np.arange(len(rows)), target]
    value = root_equal_mean(losses, [str(row["stat_group_id"]) for row in rows])
    return float(value) if value is not None else float("inf")


def _fit_temperature(objective, minimum: float, maximum: float) -> tuple[float, float, float]:
    before = float(objective(1.0))
    scored = [(float(objective(value)), float(value)) for value in np.linspace(minimum, maximum, 91)]
    after, temperature = min(scored, key=lambda item: (item[0], abs(item[1] - 1.0)))
    return temperature, before, after


def _m0_predictions(model_dir: Path, rows: list[dict]) -> np.ndarray:
    payload = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    values = []
    for row in rows:
        entry = payload["by_time"].get(str(int(row["anchor_t"])), payload["overall"])
        categories = entry.get("categories", [[0.25] * 4] * 3)
        values.append([
            np.log(entry["p0"] / (1 - entry["p0"])),
            *[np.log(max(value, 1e-7)) for group in categories for value in group],
            np.log(entry["pfull"] / (1 - entry["pfull"])),
        ])
    return np.asarray(values, np.float32)


def _anchor_predictions(model_id: str, model_dir: Path, rows: list[dict], features: Path) -> np.ndarray:
    if model_id == "M0_time":
        return _m0_predictions(model_dir, rows)
    dataset = AnchorDataset(rows, features, split="validation", normalizer_path=features / "normalizer.json")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_id).to(device)
    state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            output.append(model(_inputs(batch, device)).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, 14), np.float32)


def _probability_metrics(records: list[dict]) -> dict:
    targets = {
        "p0": "y0", "p_full": "y_full",
        "p_sys_5": "y_sys_5", "p_sys_20": "y_sys_20", "p_sys_80": "y_sys_80",
        "p_genuine_5": "y_genuine_5", "p_genuine_20": "y_genuine_20",
        "p_genuine_80": "y_genuine_80",
    }
    return {key: root_probability_metrics(records, truth, key) for key, truth in targets.items()}


def _category_diagnostics(records: list[dict], decision: dict) -> dict:
    groups = {
        "natural_failure_rescuable": lambda row: not row["y0"] and any(row[f"y_genuine_{length}"] for length in (5, 20, 80)),
        "natural_failure_all_help_ineffective": lambda row: not row["y0"] and not any(row[f"y_sys_{length}"] for length in (5, 20, 80)),
        "baseline_success": lambda row: bool(row["y0"]),
        "help_interference": lambda row: bool(row["y0"]) and any(not row[f"y_sys_{length}"] for length in (5, 20, 80)),
        "helper_completed": lambda row: any(row[f"category_l{length}"] == 1 for length in (5, 20, 80)),
        "other_success_including_too_fast": lambda row: any(row[f"category_l{length}"] == 2 for length in (5, 20, 80)),
    }
    output = {}
    for name, predicate in groups.items():
        subset = [row for row in records if predicate(row)]
        utilities = []
        for row in subset:
            length = choose_length(
                row, float(decision["primary_lambda"]),
                denominator=float(decision["cost_denominator"]),
                tolerance=float(decision["tie_tolerance"]),
            )
            if length == 0:
                utility = float(row["y0"])
            else:
                utility = (
                    float(row[f"y_genuine_{length}"])
                    - float(decision["primary_lambda"])
                    * float(row[f"helper_steps_l{length}"])
                    / float(decision["cost_denominator"])
                )
            utilities.append(utility)
        output[name] = {
            "anchors": len(subset),
            "roots": len({row["stat_group_id"] for row in subset}),
            "selected_root_equal_utility": root_equal_mean(
                utilities, [row["stat_group_id"] for row in subset]
            ),
        }
    return output


def _anchor_validation(model_id: str, seed_name: str, model_dir: Path, rows: list[dict],
                       features: Path, output_dir: Path, calibration: dict,
                       decision: dict) -> dict:
    valid_rows = [row for row in rows if row.get("split") == "validation" and bool(row.get("complete_pair"))]
    logits = _anchor_predictions(model_id, model_dir, valid_rows, features)
    if len(logits) != len(valid_rows):
        raise ValueError(f"prediction row mismatch for {model_id}/{seed_name}: {len(logits)} != {len(valid_rows)}")
    temperature, before_nll, after_nll = _fit_temperature(
        lambda value: _five_head_root_nll(logits, valid_rows, value),
        float(calibration["min"]), float(calibration["max"]),
    )
    calibrated = finite_probs(logits, temperature)
    uncalibrated = finite_probs(logits, 1.0)
    records = []
    raw_records = []
    for index, row in enumerate(valid_rows):
        base = {
            "example_id": row["example_id"], "stat_group_id": row["stat_group_id"],
            "root_id": row["root_id"], "anchor_t": int(row["anchor_t"]),
            "y0": int(row["y0"]), "y_full": int(row["y_full"]),
        }
        for length in (5, 20, 80):
            base[f"y_sys_{length}"] = int(row[f"y_sys_l{length}"])
            base[f"y_genuine_{length}"] = int(row[f"y_genuine_l{length}"])
            base[f"category_l{length}"] = int(row[f"category_l{length}"])
            base[f"helper_steps_l{length}"] = int(row[f"helper_steps_l{length}"])
        calibrated_row = dict(base)
        calibrated_row.update({key: float(value[index]) for key, value in calibrated.items()})
        raw_row = dict(base)
        raw_row.update({key: float(value[index]) for key, value in uncalibrated.items()})
        records.append(calibrated_row)
        raw_records.append(raw_row)
    summary_path = model_dir / "training_summary.json"
    training_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    checkpoint = model_dir / "best.pt"
    metrics = {
        "model": model_id, "seed": seed_name,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "temperature": temperature,
        "root_equal_five_head_nll_uncalibrated": before_nll,
        "root_equal_five_head_nll_calibrated": after_nll,
        "parameters": training_summary.get("parameters"),
        "history_length": training_summary.get("history_length", 1 if model_id == "M5_single" else 4),
        "probability_uncalibrated": _probability_metrics(raw_records),
        "probability_calibrated": _probability_metrics(records),
        "category_diagnostics": _category_diagnostics(records, decision),
    }
    payload = {"schema_version": "hb2_model_validation_v2", "model": model_id, "seed": seed_name,
               "records": records, "metrics": metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_id}.json" if seed_name == "seed0" else f"{model_id}_{seed_name}.json"
    atomic_json_dump(payload, output_dir / filename)
    return metrics


def _handoff_logits(model_dir: Path, rows: list[dict], features: Path) -> tuple[np.ndarray, dict]:
    dataset = HandoffDataset(rows, features, split="validation", normalizer_path=features / "normalizer.json")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    fallback = "proprio" if state.get("model_id") == "proprio" else "selected_visual_global"
    architecture = str(state.get("architecture") or fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture, head="handoff").to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    output = []
    with torch.no_grad():
        for batch in loader:
            output.append(model(_inputs(batch, device)).cpu().numpy())
    logits = np.concatenate(output, axis=0) if output else np.empty((0, 3), np.float32)
    return logits, state


def _handoff_validation(args, config: dict, rows: list[dict]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid_rows = [row for row in rows if row.get("split") == "validation" and bool(row.get("eligible"))]
    summaries = []
    for name in ("proprio", "selected_visual"):
        model_dir = args.models_root / name / "seed0"
        logits, state = _handoff_logits(model_dir, valid_rows, args.features)
        if len(logits) != len(valid_rows):
            raise ValueError(f"handoff prediction row mismatch for {name}")
        temperature, before_nll, after_nll = _fit_temperature(
            lambda value: _multiclass_root_nll(logits, valid_rows, value),
            float(config["hb2"]["calibration"]["min"]),
            float(config["hb2"]["calibration"]["max"]),
        )
        scaled = logits / temperature
        probabilities = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        raw_probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        raw_probabilities /= raw_probabilities.sum(axis=1, keepdims=True)
        records, raw_records = [], []
        for index, row in enumerate(valid_rows):
            base = {
                "example_id": row["example_id"], "stat_group_id": row["stat_group_id"],
                "root_id": row["root_id"], "handoff_t": int(row["handoff_t"]),
                "helper_length": int(row["helper_length"]), "target": int(row["handoff_category"]),
                "y_complete": int(row["y_complete_after_handoff"]),
                "y_strict": int(row["y_strict_genuine"]),
            }
            records.append({**base,
                "q_failure": float(probabilities[index, 0]),
                "q_complete": float(probabilities[index, 1] + probabilities[index, 2]),
                "q_strict": float(probabilities[index, 2]),
            })
            raw_records.append({**base,
                "q_failure": float(raw_probabilities[index, 0]),
                "q_complete": float(raw_probabilities[index, 1] + raw_probabilities[index, 2]),
                "q_strict": float(raw_probabilities[index, 2]),
            })
        checkpoint = model_dir / "best.pt"
        metrics = {
            "model": name, "architecture": state.get("architecture"),
            "selected_f_model": state.get("selected_f_model"),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "temperature": temperature,
            "root_equal_three_class_nll_uncalibrated": before_nll,
            "root_equal_three_class_nll_calibrated": after_nll,
            "probability_uncalibrated": {
                "q_complete": root_probability_metrics(raw_records, "y_complete", "q_complete"),
                "q_strict": root_probability_metrics(raw_records, "y_strict", "q_strict"),
            },
            "probability_calibrated": {
                "q_complete": root_probability_metrics(records, "y_complete", "q_complete"),
                "q_strict": root_probability_metrics(records, "y_strict", "q_strict"),
            },
        }
        atomic_json_dump({"schema_version": "hb2_handoff_model_validation_v2", "model": name,
                          "records": records, "metrics": metrics}, args.output_dir / f"{name}.json")
        summaries.append(metrics)
    atomic_json_dump({"schema_version": "hb2_handoff_validation_v2", "models": summaries,
                      "primary_visual_model": "selected_visual"}, args.output_dir / "summary.json")
    experiment_root = Path(config["output_root"])
    f_protocol_path = experiment_root / "config/frozen_protocol.json"
    f_protocol = json.loads(f_protocol_path.read_text(encoding="utf-8"))
    models = {}
    for summary in summaries:
        models[summary["model"]] = {
            "seed0": {
                "architecture": summary["architecture"],
                "selected_f_model": summary["selected_f_model"],
                "checkpoint": summary["checkpoint"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "temperature": summary["temperature"],
                "validation_root_equal_nll_uncalibrated": summary["root_equal_three_class_nll_uncalibrated"],
                "validation_root_equal_nll_calibrated": summary["root_equal_three_class_nll_calibrated"],
            }
        }
    handoff_feature_dir = Path(args.features)
    train_rows = [
        row for row in rows if row.get("split") == "train" and bool(row.get("eligible"))
    ]
    prior_complete = root_equal_mean(
        [int(row["y_complete_after_handoff"]) for row in train_rows],
        [row["stat_group_id"] for row in train_rows],
    )
    prior_strict = root_equal_mean(
        [int(row["y_strict_genuine"]) for row in train_rows],
        [row["stat_group_id"] for row in train_rows],
    )
    handoff_protocol = {
        "schema_version": "hb2_frozen_handoff_protocol_v2",
        "task": "square",
        "code_commit": f_protocol["code_commit"],
        "semantic_pair_id": f_protocol["semantic_pair_id"],
        "canonical_seed": 0,
        "models": models,
        "primary_visual_model": "selected_visual",
        "selected_f_model": f_protocol["selected_visual_model"],
        "training_root_equal_priors": {
            "q_complete": prior_complete,
            "q_strict": prior_strict,
        },
        "readiness_rule": {
            "minimum_test_roots": 20,
            "minimum_positive_roots_per_output": 5,
            "minimum_negative_roots_per_output": 5,
            "criterion": "selected_visual_mean_brier_below_proprio_and_frozen_train_prior",
        },
        "input_support": {
            "distribution": "observed_fixed_length_handoffs",
            "helper_lengths": [5, 20, 80],
            "arbitrary_online_exit_validated": False,
        },
        "input_schema": str((handoff_feature_dir / "input_schema.json").resolve()),
        "normalizer": str((handoff_feature_dir / "normalizer.json").resolve()),
        "artifacts": {
            "f_protocol": {"path": str(f_protocol_path.resolve()), "sha256": sha256_file(f_protocol_path)},
            "input_schema": {"path": str((handoff_feature_dir / "input_schema.json").resolve()),
                             "sha256": sha256_file(handoff_feature_dir / "input_schema.json")},
            "normalizer": {"path": str((handoff_feature_dir / "normalizer.json").resolve()),
                           "sha256": sha256_file(handoff_feature_dir / "normalizer.json")},
            "feature_manifest": {"path": str((handoff_feature_dir / "handoff.manifest.json").resolve()),
                                 "sha256": sha256_file(handoff_feature_dir / "handoff.manifest.json")},
            "validation_summary": {"path": str((args.output_dir / "summary.json").resolve()),
                                   "sha256": sha256_file(args.output_dir / "summary.json")},
        },
        "validation_locked": True,
    }
    atomic_json_dump(handoff_protocol, experiment_root / "config/frozen_handoff_protocol.json")
    print(json.dumps({"models": [item["model"] for item in summaries]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head", choices=("anchor", "handoff"), default="anchor")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.head == "handoff":
        _handoff_validation(args, config, read_table(args.dataset / "handoff_examples.parquet"))
        return
    rows = read_table(args.dataset / "anchor_examples.parquet")
    summaries = []
    for model_root in sorted(args.models_root.glob("M*")):
        for seed_dir in sorted(model_root.glob("seed*")):
            if (seed_dir / "best.pt").is_file():
                summaries.append(_anchor_validation(model_root.name, seed_dir.name, seed_dir, rows,
                                                    args.features, args.output_dir,
                                                    config["hb2"]["calibration"],
                                                    config["hb2"]["decision"]))
    atomic_json_dump({"schema_version": "hb2_validation_v2", "models": summaries}, args.output_dir / "summary.json")
    print(json.dumps({"models": [f"{item['model']}/{item['seed']}" for item in summaries]}, indent=2))


if __name__ == "__main__":
    main()
