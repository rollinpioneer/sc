"""Validation metrics, temperature calibration, and H validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.hb2.dataset import AnchorDataset, HandoffDataset
from recovery_handback.hb2.metrics import finite_probs, root_probability_metrics
from recovery_handback.hb2.models import build_model


def _inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key in ("global", "local", "proprio", "base_actions", "time")}


def _temperature(logits: np.ndarray, rows: list[dict]) -> float:
    truth = []
    for row in rows:
        truth.append([int(row["y0"]), int(row["category_l5"]), int(row["category_l20"]), int(row["category_l80"]), int(row["y_full"])])
    truth = np.asarray(truth)
    best = (float("inf"), 1.0)
    for temperature in np.linspace(0.5, 5.0, 91):
        scaled = logits / temperature
        terms = []
        for index in (0, 4):
            p = np.clip(1 / (1 + np.exp(-np.clip(scaled[:, index], -60, 60))), 1e-7, 1 - 1e-7)
            y = truth[:, 0 if index == 0 else 4]
            terms.append(-(y * np.log(p) + (1-y) * np.log1p(-p)))
        for index, start in enumerate((1, 5, 9)):
            values = scaled[:, start:start+4]
            logp = values - np.logaddexp.reduce(values, axis=1, keepdims=True)
            terms.append(-logp[np.arange(len(rows)), truth[:, index + 1]])
        loss = float(np.mean(np.stack(terms, axis=1)))
        if loss < best[0]: best = (loss, float(temperature))
    return best[1]


def _m0_predictions(model_dir: Path, rows: list[dict]) -> np.ndarray:
    payload = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    values = []
    for row in rows:
        entry = payload["by_time"].get(str(int(row["anchor_t"])), payload["overall"])
        logits = [np.log(entry["p0"] / (1-entry["p0"])),
                  *[np.log(max(v, 1e-7)) for v in entry.get("categories", [[.25]*4]*3)[0]],
                  *[np.log(max(v, 1e-7)) for v in entry.get("categories", [[.25]*4]*3)[1]],
                  *[np.log(max(v, 1e-7)) for v in entry.get("categories", [[.25]*4]*3)[2]],
                  np.log(entry["pfull"] / (1-entry["pfull"]))]
        values.append(logits)
    return np.asarray(values, np.float32)


def _model_predictions(model_id: str, model_dir: Path, rows: list[dict], features: Path) -> np.ndarray:
    if model_id == "M0_time": return _m0_predictions(model_dir, rows)
    ds = AnchorDataset(rows, features, split="validation", normalizer_path=features / "normalizer.json")
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model = build_model(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"]); model.eval(); device = next(model.parameters()).device
    output = []
    with torch.no_grad():
        for batch in loader:
            output.append(model(_inputs(batch, device)).cpu().numpy())
    # Dataset is complete/split-filtered in the same stable order as rows below.
    return np.concatenate(output, axis=0) if output else np.empty((0, 14), np.float32)


def _write_model_validation(model_id: str, model_dir: Path, rows: list[dict], features: Path, output_dir: Path) -> dict:
    valid_rows = [row for row in rows if row.get("split") == "validation" and bool(row.get("complete_pair"))]
    logits = _model_predictions(model_id, model_dir, valid_rows, features)
    temperature = _temperature(logits, valid_rows)
    probs = finite_probs(logits, temperature)
    records = []
    for index, row in enumerate(valid_rows):
        record = {"example_id": row["example_id"], "stat_group_id": row["stat_group_id"], "root_id": row["root_id"],
                  "anchor_t": row["anchor_t"], "y0": row["y0"], "y_full": row["y_full"],
                  "category_l5": row["category_l5"], "category_l20": row["category_l20"], "category_l80": row["category_l80"],
                  "temperature": temperature}
        record.update({key: float(value[index]) for key, value in probs.items()})
        for length in (5, 20, 80): record[f"y_genuine_{length}"] = row[f"y_genuine_l{length}"]
        records.append(record)
    metrics = {"model": model_id, "checkpoint": str((model_dir / "best.pt").resolve()), "temperature": temperature}
    for key, truth in (("p0", "y0"), ("p_full", "y_full"), ("p_genuine_5", "y_genuine_5"),
                       ("p_genuine_20", "y_genuine_20"), ("p_genuine_80", "y_genuine_80")):
        metrics[key] = root_probability_metrics(records, truth, key)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({"model": model_id, "records": records, "metrics": metrics}, output_dir / f"{model_id}.json")
    return metrics


def _handoff_validation(args, config: dict, rows: list[dict]) -> None:
    output_dir = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True)
    for view in ("proprio", "selected_visual"):
        model_dir = args.models_root / view / "seed0"
        ds_rows = [row for row in rows if row.get("split") == "validation" and bool(row.get("eligible"))]
        ds = HandoffDataset(rows, args.features, split="validation", normalizer_path=args.features / "normalizer.json")
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        model = build_model(view, head="handoff").to("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(state["state_dict"]); model.eval()
        device = next(model.parameters()).device; logits = []
        with torch.no_grad():
            for batch in loader: logits.append(model(_inputs(batch, device)).cpu().numpy())
        logits = np.concatenate(logits) if logits else np.empty((0, 3))
        records = [{"example_id": row["example_id"], "stat_group_id": row["stat_group_id"], "target": row["handoff_category"]}
                   for row in ds_rows]
        for index, record in enumerate(records):
            p = np.exp(logits[index] - np.max(logits[index])); p /= p.sum()
            record.update({"q_complete": float(p[1] + p[2]), "q_strict": float(p[2]), "q_failure": float(p[0])})
        atomic_json_dump({"model": view, "records": records}, output_dir / f"{view}.json")
    atomic_json_dump({"schema_version": "hb2_handoff_validation_v1", "models": ["proprio", "selected_visual"]}, output_dir / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head", choices=("anchor", "handoff"), default="anchor")
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.head == "handoff":
        _handoff_validation(args, config, read_table(args.dataset / "handoff_examples.parquet")); return
    rows = read_table(args.dataset / "anchor_examples.parquet")
    metrics = []
    for path in sorted(args.models_root.glob("M*/seed0")):
        if (path / "best.pt").is_file(): metrics.append(_write_model_validation(path.parent.name, path, rows, args.features, args.output_dir))
    atomic_json_dump({"schema_version": "hb2_validation_v1", "models": metrics}, args.output_dir / "summary.json")
    print(json.dumps({"models": [metric["model"] for metric in metrics]}, indent=2))


if __name__ == "__main__": main()

