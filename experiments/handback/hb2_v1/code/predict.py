"""Create label-free predictions from frozen HB2 protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recovery_handback.common import atomic_json_dump, read_table, sha256_file
from recovery_handback.hb2.dataset import AnchorDataset, HandoffDataset
from recovery_handback.hb2.metrics import choose_length, finite_probs
from recovery_handback.hb2.models import build_model


def _inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    names = ("global", "local", "proprio", "base_actions", "time", "helper_elapsed")
    return {key: value.to(device) for key, value in batch.items() if key in names}


def _verify_checkpoint(entry: dict) -> Path:
    path = Path(entry["checkpoint"])
    expected = entry.get("checkpoint_sha256") or entry.get("sha256")
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected and sha256_file(path) != expected:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    return path


def _m0_logits(checkpoint: Path, rows: list[dict]) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = []
    for row in rows:
        entry = payload["by_time"].get(str(int(row["anchor_t"])), payload["overall"])
        categories = entry["categories"]
        values.append([
            np.log(entry["p0"] / (1 - entry["p0"])),
            *[np.log(max(value, 1e-7)) for group in categories for value in group],
            np.log(entry["pfull"] / (1 - entry["pfull"])),
        ])
    return np.asarray(values, np.float32)


def _anchor_logits(
    model_id: str, checkpoint: Path, rows: list[dict], features: Path, role: str
) -> np.ndarray:
    if model_id == "M0_time":
        return _m0_logits(checkpoint, rows)
    dataset = AnchorDataset(
        rows, features, split=role, normalizer_path=features / "normalizer.json",
        inference=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_id).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            output.append(model(_inputs(batch, device)).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, 14), np.float32)


def _handoff_logits(
    architecture: str, checkpoint: Path, rows: list[dict], features: Path, role: str
) -> np.ndarray:
    dataset = HandoffDataset(
        rows, features, split=role, normalizer_path=features / "normalizer.json",
        inference=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture, head="handoff").to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in loader:
            output.append(model(_inputs(batch, device)).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, 3), np.float32)


def _system_choice(probabilities: dict[str, float], decision: dict) -> int:
    scores = [(float(probabilities["p0"]), 0)]
    for length in (5, 20, 80):
        score = (
            float(probabilities[f"p_sys_{length}"])
            - float(decision["primary_lambda"]) * length / float(decision["cost_denominator"])
        )
        scores.append((score, length))
    best = max(score for score, _ in scores)
    return min(
        length
        for score, length in scores
        if best - score <= float(decision["tie_tolerance"])
    )


def _predict_anchor(args, protocol: dict, summary: dict) -> None:
    metadata_path = args.dataset / "prediction_anchor_metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError("label-free prediction anchor metadata is required")
    rows = [
        row for row in read_table(metadata_path)
        if row.get("split") == args.role and bool(row.get("complete_pair"))
    ]
    decision = protocol["decision"]
    for model_id, seeds in protocol["models"].items():
        for seed_name, entry in seeds.items():
            checkpoint = _verify_checkpoint(entry)
            logits = _anchor_logits(model_id, checkpoint, rows, args.features, args.role)
            if len(logits) != len(rows):
                raise ValueError(f"anchor prediction row mismatch for {model_id}/{seed_name}")
            probabilities = finite_probs(logits, float(entry["temperature"]))
            records = []
            for index, row in enumerate(rows):
                scalar = {key: float(value[index]) for key, value in probabilities.items()}
                records.append({
                    "example_id": str(row["example_id"]),
                    "stat_group_id": str(row["stat_group_id"]),
                    "root_id": str(row["root_id"]),
                    "anchor_t": int(row["anchor_t"]),
                    "model": model_id,
                    "seed": seed_name,
                    "temperature": float(entry["temperature"]),
                    **scalar,
                    "selected_length": choose_length(
                        scalar,
                        float(decision["primary_lambda"]),
                        denominator=float(decision["cost_denominator"]),
                        tolerance=float(decision["tie_tolerance"]),
                    ),
                    "selected_system_length": _system_choice(scalar, decision),
                })
            name = f"{model_id}_{seed_name}.json"
            output = args.output_dir / name
            atomic_json_dump({
                "schema_version": "hb2_label_free_anchor_predictions_v2",
                "model": model_id,
                "seed": seed_name,
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "records": records,
            }, output)
            summary["anchor_models"].append({
                "model": model_id, "seed": seed_name, "path": str(output.resolve()),
                "records": len(records),
            })


def _predict_handoff(args, protocol: dict, summary: dict) -> None:
    if not args.handoff_protocol:
        summary["exit_model_ready"] = False
        return
    handoff_protocol = json.loads(args.handoff_protocol.read_text(encoding="utf-8"))
    metadata_path = args.dataset / "prediction_handoff_metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError("label-free prediction handoff metadata is required")
    rows = [
        row for row in read_table(metadata_path)
        if row.get("split") == args.role and bool(row.get("eligible"))
    ]
    feature_dir = args.features / "handoff"
    if not (feature_dir / "handoff.npz").is_file():
        summary["exit_model_ready"] = False
        summary["handoff_feature_issue"] = str(feature_dir / "handoff.npz")
        return
    for model_name, seeds in handoff_protocol["models"].items():
        for seed_name, entry in seeds.items():
            checkpoint = _verify_checkpoint(entry)
            logits = _handoff_logits(
                str(entry["architecture"]), checkpoint, rows, feature_dir, args.role
            )
            if len(logits) != len(rows):
                raise ValueError(f"handoff prediction row mismatch for {model_name}/{seed_name}")
            scaled = logits / float(entry["temperature"])
            probabilities = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            records = [{
                "example_id": str(row["example_id"]),
                "stat_group_id": str(row["stat_group_id"]),
                "root_id": str(row["root_id"]),
                "handoff_t": int(row["handoff_t"]),
                "helper_length": int(row["helper_length"]),
                "model": model_name,
                "seed": seed_name,
                "temperature": float(entry["temperature"]),
                "q_failure": float(probabilities[index, 0]),
                "q_complete": float(probabilities[index, 1] + probabilities[index, 2]),
                "q_strict": float(probabilities[index, 2]),
            } for index, row in enumerate(rows)]
            name = f"handoff_{model_name}_{seed_name}.json"
            output = args.output_dir / name
            atomic_json_dump({
                "schema_version": "hb2_label_free_handoff_predictions_v2",
                "head": "handoff",
                "model": model_name,
                "seed": seed_name,
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "records": records,
            }, output)
            summary["handoff_models"].append({
                "model": model_name, "seed": seed_name, "path": str(output.resolve()),
                "records": len(records),
            })
    summary["exit_model_ready"] = bool(summary["handoff_models"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--handoff-protocol", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--role", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "hb2_label_free_predictions_v2",
        "role": args.role,
        "protocol_sha256": sha256_file(args.protocol),
        "anchor_models": [],
        "handoff_models": [],
    }
    _predict_anchor(args, protocol, summary)
    _predict_handoff(args, protocol, summary)
    atomic_json_dump(summary, args.output_dir / "summary.json")
    print(json.dumps({
        "anchor_models": len(summary["anchor_models"]),
        "handoff_models": len(summary["handoff_models"]),
        "exit_model_ready": summary["exit_model_ready"],
    }, indent=2))


if __name__ == "__main__":
    main()
