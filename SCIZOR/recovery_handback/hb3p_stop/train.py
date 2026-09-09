"""Train the small stop/continue MLP and select a threshold from root-level OOF scores."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb3p.metrics import paired_bootstrap
from recovery_handback.hb3p_stop.io import write_jsonl
from recovery_handback.hb3p_stop.model import StopContinueMLP, save_normalizer


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalizer(features: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features[indices].mean(axis=0).astype(np.float32)
    std = np.maximum(features[indices].std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def _fit(features: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, *, seed: int, epochs: int, device: str):
    _seed(seed)
    mean, std = _normalizer(features, train_indices)
    model = StopContinueMLP().to(device)
    x = torch.from_numpy(((features[train_indices] - mean) / std).astype(np.float32)).to(device)
    y = torch.from_numpy(labels[train_indices].astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = float(y.sum().item())
    negatives = float(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("every training fold must contain STOP and CONTINUE labels")
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=positive_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, mean, std


def _scores(model, features: np.ndarray, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(((features[indices] - mean) / std).astype(np.float32)).to(device)
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(x)).cpu().numpy().astype(np.float64)


def _threshold(oof_scores: np.ndarray, rows: list[dict]) -> tuple[float, dict]:
    candidates = sorted({0.0, 1.0, *[float(value) for value in oof_scores]})
    values = []
    for threshold in candidates:
        decisions = oof_scores >= threshold
        realized = np.asarray([
            float(row["utility_l80"] if decision else row["utility_l60"])
            for row, decision in zip(rows, decisions)
        ])
        values.append((float(realized.mean()), float(threshold), int(decisions.sum())))
    # Prefer STOP on exact ties, then fewer CONTINUE decisions.
    best = max(values, key=lambda item: (item[0], -item[2], item[1]))
    selected = {
        "threshold": best[1],
        "oof_mean_realized_utility": best[0],
        "oof_continue_count": best[2],
        "candidate_count": len(candidates),
        "selection_rule": "maximize root-mean paired realized utility; ties prefer fewer CONTINUE decisions",
    }
    return best[1], selected


def train(dataset_dir: Path, output_dir: Path, *, seed: int = 20260909, epochs: int = 300, device: str = "cpu") -> dict:
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    with np.load(dataset_dir / "paired_dataset.npz", allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
    rows = [json.loads(line) for line in (dataset_dir / "paired_labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(labels) or features.shape != (len(rows), 65) or set(labels.tolist()) != {0, 1}:
        raise ValueError("dataset must contain a finite [N,65] matrix with both stop and continue labels")
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    fold_rows = []
    for fold in sorted(set(folds.tolist())):
        train_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)
        model, mean, std = _fit(features, labels, train_idx, seed + fold, epochs, device)
        oof[test_idx] = _scores(model, features, test_idx, mean, std, device)
        fold_rows.append({"fold": fold, "train_roots": int(len(train_idx)), "holdout_roots": int(len(test_idx))})
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF prediction coverage is incomplete")
    threshold, selection = _threshold(oof, rows)
    final_indices = np.arange(len(rows))
    model, mean, std = _fit(features, labels, final_indices, seed, epochs, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "stop_continue.pt"
    torch.save({
        "schema_version": "hb3p_stop_continue_model_v1",
        "input_dim": 65,
        "hidden": [64, 32],
        "state_dict": model.state_dict(),
        "seed": int(seed),
        "epochs": int(epochs),
        "threshold": float(threshold),
        "label_rule": "paired_l80_utility_gt_l60_utility",
    }, checkpoint)
    normalizer_path = output_dir / "normalizer.json"
    save_normalizer(normalizer_path, mean, std, fit_indices=final_indices.tolist())
    predictions = []
    for index, row in enumerate(rows):
        predictions.append({
            "root_id": row["root_id"],
            "fold": int(folds[index]),
            "label": int(labels[index]),
            "oof_continue_score": float(oof[index]),
            "oof_decision": "CONTINUE" if oof[index] >= threshold else "STOP",
            "utility_l60": row["utility_l60"],
            "utility_l80": row["utility_l80"],
            "realized_oof_utility": row["utility_l80"] if oof[index] >= threshold else row["utility_l60"],
        })
    write_jsonl(predictions, output_dir / "oof_predictions.jsonl")
    chosen = np.asarray([score >= threshold for score in oof])
    y = labels.astype(bool)
    atomic_json_dump({
        "schema_version": "hb3p_stop_continue_training_summary_v1",
        "dataset_manifest_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
        "dataset_matrix_sha256": sha256_file(dataset_dir / "paired_dataset.npz"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "normalizer": str(normalizer_path.resolve()),
        "normalizer_sha256": sha256_file(normalizer_path),
        "roots": len(rows),
        "continue_labels": int(labels.sum()),
        "stop_labels": int((labels == 0).sum()),
        "folds": fold_rows,
        "threshold_selection": selection,
        "oof_label_accuracy": float(np.mean(chosen == y)),
        "oof_continue_true_positive": int(np.sum(chosen & y)),
        "oof_continue_false_positive": int(np.sum(chosen & ~y)),
        "oof_continue_false_negative": int(np.sum(~chosen & y)),
        "oof_continue_true_negative": int(np.sum(~chosen & ~y)),
        "input_protocol": "9-D proprio + four 7-D base-action frames + current normalized time; no images or privileged labels",
        "pilot_warning": "40-root source set is insufficient for a formal non-inferiority claim",
    }, output_dir / "training_summary.json")
    return {"output_dir": str(output_dir.resolve()), "roots": len(rows), "threshold": threshold, "oof_accuracy": float(np.mean(chosen == y))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(train(args.dataset_dir, args.output_dir, seed=args.seed, epochs=args.epochs, device=args.device), indent=2))


if __name__ == "__main__":
    main()
