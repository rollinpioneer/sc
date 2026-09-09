"""Predict frozen HB2 outcomes before evaluation joins observed branch labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.hb2.dataset import AnchorDataset
from recovery_handback.hb2.features import ObservationFeaturizer
from recovery_handback.hb2.metrics import finite_probs
from recovery_handback.hb2.models import build_model


def _inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key in ("global", "local", "proprio", "base_actions", "time")}


def _m0_logits(checkpoint: Path, rows: list[dict]) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = []
    for row in rows:
        entry = payload["by_time"].get(str(int(row["anchor_t"])), payload["overall"])
        cats = entry.get("categories", [[.25] * 4] * 3)
        values.append([np.log(entry["p0"] / (1-entry["p0"])), *sum(([np.log(max(x, 1e-7)) for x in cats[i]] for i in range(3)), []),
                       np.log(entry["pfull"] / (1-entry["pfull"]))])
    return np.asarray(values, np.float32)


def _logits(model_id: str, checkpoint: Path, rows: list[dict], features: Path, role: str) -> np.ndarray:
    if model_id == "M0_time": return _m0_logits(checkpoint, rows)
    ds = AnchorDataset(rows, features, split=role, normalizer_path=features / "normalizer.json")
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_id).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"]); model.eval()
    output = []
    with torch.no_grad():
        for batch in loader: output.append(model(_inputs(batch, device)).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, 14), np.float32)


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
    rows = [row for row in read_table(args.dataset / "anchor_examples.parquet") if row.get("split") == args.role and bool(row.get("complete_pair"))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "hb2_predictions_v1", "role": args.role, "models": []}
    model_entries = protocol.get("models", {})
    for model_id, seeds in model_entries.items():
        for seed, entry in seeds.items():
            checkpoint = Path(entry["checkpoint"])
            if not checkpoint.is_file(): continue
            logits = _logits(model_id, checkpoint, rows, args.features, args.role)
            temperature = float(protocol.get("selection", {}).get("canonical", {}).get("temperature", 1.0)) if int(seed.replace("seed", "")) == 0 else 1.0
            probs = finite_probs(logits, temperature)
            records = []
            for index, row in enumerate(rows):
                record = {"example_id": row["example_id"], "stat_group_id": row["stat_group_id"], "root_id": row["root_id"],
                          "anchor_t": row["anchor_t"], "y0": int(row["y0"]), "y_full": int(row["y_full"]),
                          "category_l5": int(row["category_l5"]), "category_l20": int(row["category_l20"]), "category_l80": int(row["category_l80"]),
                          "y_genuine_5": int(row["y_genuine_l5"]), "y_genuine_20": int(row["y_genuine_l20"]), "y_genuine_80": int(row["y_genuine_l80"]),
                          "helper_steps_l5": int(row["helper_steps_l5"]), "helper_steps_l20": int(row["helper_steps_l20"]), "helper_steps_l80": int(row["helper_steps_l80"]),
                          "model": model_id, "seed": seed, "temperature": temperature}
                record.update({key: float(value[index]) for key, value in probs.items()})
                record["selected_length"] = None
                records.append(record)
            name = f"{model_id}_{seed}.json"
            atomic_json_dump({"model": model_id, "seed": seed, "checkpoint": str(checkpoint.resolve()), "records": records}, args.output_dir / name)
            summary["models"].append({"model": model_id, "seed": seed, "path": str((args.output_dir / name).resolve())})
    atomic_json_dump(summary, args.output_dir / "summary.json")
    print(json.dumps({"models": len(summary["models"]), "rows": len(rows)}, indent=2))


if __name__ == "__main__": main()

