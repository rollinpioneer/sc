"""Verify four-root runtime parity, then execute each frozen one-shot choice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.hb2.features import (
    CAMERA_KEYS,
    _anchor_history,
    _handoff_history,
    _proprio,
)
from recovery_handback.hb2.metrics import choose_length, finite_probs


def _first_anchor_per_root(rows: list[dict]) -> list[dict]:
    selected = {}
    for row in sorted(rows, key=lambda item: (str(item["root_id"]), int(item["anchor_t"]))):
        selected.setdefault(str(row["root_id"]), row)
    return list(selected.values())


def _max_difference(expected: dict[str, np.ndarray], actual: dict[str, torch.Tensor]) -> dict[str, float]:
    output = {}
    for key, value in expected.items():
        observed = actual[key].detach().cpu().numpy()
        output[key] = float(np.max(np.abs(np.asarray(value, np.float32) - observed)))
    return output


def _prediction_mode(args, config: dict) -> None:
    from recovery_handback.hb2.runtime import RuntimePredictor

    anchors = read_table(args.pilot_roots / "anchors.parquet")
    selected = _first_anchor_per_root(anchors)
    if len(selected) != 4:
        raise ValueError(f"runtime probe requires exactly four pilot roots, got {len(selected)}")
    predictor = RuntimePredictor(
        args.config, args.protocol, args.handoff_protocol, device=args.device
    )
    cache = predictor.featurizer.featurize_rows(selected, head="anchor", batch_size=4)
    pm, ps, am, action_std = predictor.f_normalizer
    result = {
        "schema_version": "hb2_runtime_probe_v2",
        "status": "running",
        "pilot_roots": len(selected),
        "checks": [],
        "camera_order": list(CAMERA_KEYS),
        "action_source": "persisted rollout suggestions at exact absolute times",
        "single_query_per_task": True,
        "online_adaptive_switching_tested": False,
    }
    selections = []
    tolerance = float(args.tolerance)
    for index, row in enumerate(selected):
        frames, actions, frame_times = _anchor_history(row, int(config["horizon_steps"]))
        proprio = np.stack([_proprio(frame) for frame in frames])
        runtime_inputs, _ = predictor._features(
            frames, proprio, actions, int(row["anchor_t"])
        )
        expected = {
            "global": cache["global"][index:index + 1].astype(np.float32),
            "local": cache["local"][index:index + 1].astype(np.float32),
            "proprio": ((cache["proprio"][index:index + 1] - pm) / ps).astype(np.float32),
            "base_actions": ((cache["base_actions"][index:index + 1] - am) / action_std).astype(np.float32),
            "time": cache["time"][index:index + 1].astype(np.float32),
            "helper_elapsed": cache["helper_elapsed"][index:index + 1].astype(np.float32),
        }
        differences = _max_difference(expected, runtime_inputs)
        with torch.inference_mode():
            offline_logits = predictor.model(runtime_inputs).cpu().numpy()
        offline_probabilities = finite_probs(offline_logits, predictor.temperature)
        offline = {key: float(value[0]) for key, value in offline_probabilities.items()}
        offline_length = choose_length(
            offline,
            float(predictor.protocol["decision"]["primary_lambda"]),
            denominator=float(predictor.protocol["decision"]["cost_denominator"]),
            tolerance=float(predictor.protocol["decision"]["tie_tolerance"]),
        )
        online = predictor.predict_before_takeover(
            frames, proprio, actions, int(row["anchor_t"])
        )
        probability_error = max(
            abs(offline[key] - float(online[key])) for key in offline
        )
        check = {
            "root_id": str(row["root_id"]),
            "anchor_id": str(row["anchor_id"]),
            "anchor_t": int(row["anchor_t"]),
            "finite_inputs": bool(
                np.isfinite(proprio).all()
                and np.isfinite(actions).all()
                and all(
                    np.isfinite(np.asarray(frame[key])).all()
                    for frame in frames for key in CAMERA_KEYS
                )
            ),
            "feature_max_abs_difference": differences,
            "probability_max_abs_difference": probability_error,
            "offline_selected_length": offline_length,
            "online_selected_length": int(online["selected_length"]),
            "selected_length_equal": offline_length == int(online["selected_length"]),
            "backbone_seconds": float(online["backbone_seconds"]),
            "prediction_head_seconds": float(online["prediction_head_seconds"]),
            "inference_seconds": float(online["inference_seconds"]),
        }
        check["parity_passed"] = (
            max(differences.values()) <= tolerance
            and probability_error <= tolerance
            and check["selected_length_equal"]
        )
        result["checks"].append(check)
        selections.append({
            "root_id": str(row["root_id"]),
            "anchor_id": str(row["anchor_id"]),
            "anchor_t": int(row["anchor_t"]),
            "selected_length": int(online["selected_length"]),
        })

    handoff_data = Path(config["output_root"]) / "data/hb2_pilot/handoff_examples.parquet"
    eligible = [
        row for row in read_table(handoff_data) if bool(row.get("eligible"))
    ] if handoff_data.is_file() else []
    if eligible and args.handoff_protocol:
        row = eligible[0]
        frames, actions, _ = _handoff_history(row, int(config["horizon_steps"]))
        proprio = np.stack([_proprio(frame) for frame in frames])
        handoff_prediction = predictor.predict_at_handoff(
            frames,
            proprio,
            actions,
            int(row["handoff_t"]),
            int(row["helper_length"]),
        )
        result["handoff_interface"] = {
            "status": "ready",
            "example_id": str(row["example_id"]),
            "handoff_t": int(row["handoff_t"]),
            "elapsed_helper_steps": int(row["helper_length"]),
            "prediction": handoff_prediction,
        }
    else:
        result["handoff_interface"] = {
            "status": "not_estimable",
            "reason": "no eligible pilot handoff or protocol unavailable",
        }
    result["status"] = (
        "parity_ready" if all(check["parity_passed"] for check in result["checks"])
        else "parity_failed"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, args.output_dir / "runtime_probe.json")
    atomic_json_dump({
        "schema_version": "hb2_runtime_probe_selections_v1",
        "role": "hb2_pilot",
        "selections": selections,
    }, args.output_dir / "selected_branches.json")
    print(json.dumps({
        "status": result["status"],
        "pilot_roots": len(selected),
        "selected_lengths": [row["selected_length"] for row in selections],
    }, indent=2))
    if result["status"] != "parity_ready" or result["handoff_interface"]["status"] != "ready":
        raise SystemExit(2)


def _execution_mode(args, config: dict) -> None:
    from recovery_handback.execution.paired_branch import run_branch

    if not args.policy_pair:
        raise ValueError("--policy-pair is required with --execute-selections")
    selection_payload = json.loads(args.execute_selections.read_text(encoding="utf-8"))
    selections = {
        str(row["anchor_id"]): row for row in selection_payload["selections"]
    }
    anchors = {
        str(row["anchor_id"]): row
        for row in read_table(args.pilot_roots / "anchors.parquet")
    }
    pair = json.loads(args.policy_pair.read_text(encoding="utf-8"))
    records = []
    for anchor_id, selection in selections.items():
        anchor = anchors[anchor_id]
        length = int(selection["selected_length"])
        slug = anchor_id.replace(":", "__").replace("/", "_")
        result = run_branch(
            config,
            anchor,
            pair,
            length,
            args.output_dir / "branches" / slug / "selected.json",
            base_device=args.base_device,
            repair_device=args.repair_device,
        )
        records.append({
            "root_id": str(anchor["root_id"]),
            "anchor_id": anchor_id,
            "selected_length": length,
            "engineering_ok": bool(result["engineering_ok"]),
            "system_success": bool(result["system_success"]),
            "genuine_handoff_success": bool(result["genuine_handoff_success"]),
            "prefix_env_steps": int(result["prefix_env_steps"]),
            "continuation_env_steps": int(result["continuation_env_steps"]),
            "base_policy_calls": int(result["base_policy_calls"]),
            "repair_policy_calls": int(result["repair_policy_calls"]),
            "helper_steps_actual": int(result["helper_steps_actual"]),
            "repair_calls_after_handoff": int(result["repair_calls_after_handoff"]),
            "changed_action_steps": int(result["changed_action_steps"]),
            "wall_seconds": float(result["wall_seconds"]),
            "exception_reason": result["exception_reason"],
        })
    execution = {
        "schema_version": "hb2_runtime_probe_execution_v1",
        "pilot_roots": len(records),
        "all_engineering_ok": all(row["engineering_ok"] for row in records),
        "single_intervention_only": True,
        "repair_calls_after_handoff_total": sum(
            row["repair_calls_after_handoff"] for row in records
        ),
        "prefix_replay_steps_total": sum(row["prefix_env_steps"] for row in records),
        "records": records,
    }
    atomic_json_dump(execution, args.output_dir / "runtime_probe_execution.json")
    probe_path = args.output_dir / "runtime_probe.json"
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        probe["control_execution"] = execution
        probe["status"] = (
            "runtime_probe_complete"
            if probe.get("status") == "parity_ready" and execution["all_engineering_ok"]
            else "runtime_probe_failed"
        )
        atomic_json_dump(probe, probe_path)
    print(json.dumps({
        "all_engineering_ok": execution["all_engineering_ok"],
        "pilot_roots": len(records),
        "repair_calls_after_handoff_total": execution["repair_calls_after_handoff_total"],
    }, indent=2))
    if not execution["all_engineering_ok"] or execution["repair_calls_after_handoff_total"] != 0:
        raise SystemExit(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--handoff-protocol", type=Path)
    parser.add_argument("--pilot-roots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-pair", type=Path)
    parser.add_argument("--execute-selections", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--base-device", default="cuda")
    parser.add_argument("--repair-device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.execute_selections:
        _execution_mode(args, config)
    else:
        if not args.protocol:
            raise ValueError("--protocol is required for prediction parity")
        _prediction_mode(args, config)


if __name__ == "__main__":
    main()
